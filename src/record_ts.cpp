/** Frozen function backend
 *
 * This file implements the backend for the `FrozenFunction` feature.
 * Calling ``jitc_freeze_start`` allows recording of ``ThreadState`` operations
 * by swapping out the current thread state with a ``RecordThreadState``, which
 * records all operations performed on itself. Ending the recording process by
 * calling `jit_freeze_stop`, produces a ``Recording``, which can be replayed
 * later.
 *
 * The ``Recording`` relies on three main data structures:
 *
 * - ``recording.recorded_variables``: A vector of variables that have been
 *   encountered when recording. We track variables by the memory regions they
 *   refer to, therefore only evaluated variables can be tracked. During
 *   recording, we map the pointers to their ``RecordedVariables`` using the
 *   ``ptr_to_slot`` hash map. For that reason pointers have to refer to the
 *   start of a memory region, allocated with ``jic_malloc``.
 *
 * - ``recording.operations``: A vector of operations, that have been recorded.
 *   Each operation stores its type, some additional information, such as the
 *   executed kernel, and a tuple of indices into the ``recording.dependencies``
 *   vector. This records the accesses the operation has performed on variables.
 *
 * - ``recording.dependencies``: A vector of ``AccessInfo`` which are used used
 *   to record how operations access different variables. Every entry records an
 *   index (``slot``) into the ``recording.recorded_variables`` vector,
 *   representing the memory region it refers to. Additionally, it stores the
 *   type of access (Input or Output) and the variable type as which the
 *   variable was accessed. The latter is required since some operations might
 *   interpret the same memory with different types. For example, a ``memcpy``
 *   operation might access the an ``UInt32`` array as an array of ``UInt8``
 *   bytes.
 *
 * To identify memory regions by their pointers and map them to indices in
 * ``recording.recorded_variables``, we employ the ``ptr_to_slot`` hash map.
 *
 * When recording a new operation, it is pushed on the back of the
 * ``recording.operations`` array. If the operation accesses a variable that
 * already exists i.e. it is the result of another operation or the input of the
 * Frozen Function, ``get_variable`` should be used to lookup the index into the
 * ``recorded_variables``, representing the variable. The function will throw an
 * exception if the pointer has not yet been encountered.
 * On the other hand, if the variable is either a result of the operation or it
 * is accessed by that operation for the first time (see aggregate),
 * ``add_variable`` should be used. This will either push a new
 * ``RecordedVariable`` to the ``recorded_variables`` vector and insert an index
 * to it into ``ptr_to_slot`` or lookup the one already known to
 * ``ptr_to_slot``. The index returned by either function can be used to record
 * how the memory region/variable is accessed by the operation. To this end, a
 * ``AccessInfo`` struct is constructed, with at least its ``slot`` being set to
 * the recorded variable index and its `type` to either ``ParamType::Input`` or
 * ``ParamType::Output``. Using ``add_param`` the struct is then pushed onto the
 * ``dependencies`` vector, while updating the variable type and state of the
 * referenced ``RecordedVariable``. We provide wrapper functions for recording
 * both input and output accesses to variables, which take the pointer address
 * and an optional variable type.
 *
 * Example in aggregate operation:
 * ```
 * uint32_t slot = add_variable(p.src);
 *
 * AccessInfo info;
 * info.slot           = slot;
 * info.type           = ParamType::Input;
 * info.pointer_access = p.size == 8;
 * info.extra.offset   = p.offset;
 * info.test_uninit    = false;
 * add_param(info);
 * ```
 *
 * Example of recording `reduce_dot`:
 * This code is executed in the `record_reduce_dot` function and shows how to
 * record an operation. The functions ``add_out_param``, ``add_in_param`` as
 * well as the fields ``m_recording`` are part of the ``RecordThreadState``.
 * ```
 * uint32_t start = m_recording.dependencies.size();
 * add_out_param(out, type);
 * add_in_param(ptr_1, type);
 * add_in_param(ptr_2, type);
 * uint32_t end = m_recording.dependencies.size();
 *
 * Operation op;
 * op.type             = OpType::ReduceDot;
 * op.dependency_range = std::pair(start, end);
 * op.size             = size;
 * m_recording.operations.push_back(op);
 * ```
 * In order to replay the recorded ThreadState, we have to first call the
 * ``jit_freeze_dry_run`` function. This will determine if the recording has to
 * be re-traced, by replaying the recording without executing the operations or
 * allocating memory. If that function returns with ``1``, the
 * ``jitc_freeze_replay`` function can be called to actually replay the
 * recording.
 *
 *
 *
 * The following is a list of complications, that might occur when trying to
 * record frozen functions:
 * - Since we allow replaying frozen functions with differently sized input
 *   variables, changes to the width will not be tracked by the function
 *   interface key. In order to still allow using the width of one variable to
 *   define the width of another, for example,
 *   ``dr.gather(t, x, dr.width(x))``, we implement a heuristic that determines
 *   the kernel launch size dependent on the sizes of its inputs. The heuristic
 *   might fall back to the recorded kernel size, which might not be correct.
 *   Introducing a ``symbolic_width`` function can resolve this.
 * - Some operations might require dry running the recording before being able
 *   to replay them. One example, is the ``reduce_expanded`` case, which results
 *   in different kernels depending on the size of its input size. During dry
 *   running, the size of the variable is inferred in the same way as when
 *   replaying, without actually executing any operations. In these cases, we
 *   set the ``requires_dry_run`` flag in the recording. The ``Compress``
 *   operation however creates an output which's size depends on the content of
 *   the compressed array. We therefore return false when encountering a
 *   ``Compress`` operation in dry run mode, which should result in a
 *   re-recording of the frozen function.
 * - Recording the ``poke`` operation is currently unsupported, as it will write
 *   to some offset pointer in a memory region, which cannot be tracked.
 * - Exceptions that occur in the ``record_*`` functions are captured instead of
 *   propagated to the calling function. Letting the exception propagate from a
 *   function such as \ref launch will leak variables, since they are still held
 *   by the \ref schedule. Therefore, all other operations might still be
 *   executed and the content of the variables will have changed.
 */

#include "record_ts.h"
#include "call.h"
#include "common.h"
#include "drjit-core/jit.h"
#include "eval.h"
#include "internal.h"
#include "llvm.h"
#include "log.h"
#include "malloc.h"
#include "profile.h"
#include "util.h"
#include "var.h"

const char *op_type_name[(int) OpType::Count]{
    "Barrier",        "KernelLaunch",      "MemsetAsync", "Expand",
    "ReduceExpanded", "Compress",          "MemcpyAsync", "Mkperm",
    "BlockReduce",    "BlockPrefixReduce", "ReduceDot",   "Aggregate",
    "Free",
};

static bool dry_run = false;

/**
 * Represents a variable during replay.
 * It is created from the RecordVariable at the top of the replay function.
 */
struct ReplayVariable {
    void *data = nullptr;
    // Tracks the capacity, this allocation has been allocated for
    size_t alloc_size = 0;
    // Tracks the size in bytes, of this allocation
    size_t data_size = 0;
    uint32_t index;
    RecordedVarInit init;

    ReplayVariable(RecordedVariable &rv) : index(rv.index), init(rv.init) {
        if (init == RecordedVarInit::Captured) {
            // Copy the variable, so that it isn't changed by this recording.
            // Captured variables are only used for vcall offset buffers, which
            // are not supposed to change between replay calls.

            Variable *v = jitc_var(index);
            alloc_size  = v->size * type_size[v->type];
            data_size   = alloc_size;
            if (!dry_run) {
                index = jitc_var_copy(index);
                data  = v->data;
            }
        }
    }

    /// Initializes the \ref ReplayVariable from a function input.
    void init_from_input(Variable *input_variable) {
        data       = input_variable->data;
        alloc_size = type_size[input_variable->type] * input_variable->size;
        data_size = alloc_size;
    }

    /**
     * Calculate the number of elements from the size in bytes given some
     * variable type.
     */
    uint32_t size(VarType vtype) { return size(type_size[(uint32_t) vtype]); }
    /**
     * Calculate the number of elements given some type size.
     * This depends on how an operation accesses a variable.
     * For example, a memcpy operation might access a variable as an array of
     * \ref uint8_t types, whereas a kernel can access the same variable as an
     * array of ``uint64_t``. This changes the size of the variable when
     * inferring the size of the kernel launch.
     */
    uint32_t size(uint32_t tsize) {
        if (tsize == 0)
            jitc_raise("replay(): Invalid variable type!");
        size_t size = (data_size / (size_t) tsize);

        if (size == 0)
            jitc_raise("replay(): Error, determining size of variable! init "
                       "%u, dsize=%zu",
                       (uint32_t) init, data_size);

        if (size * (size_t) tsize != data_size)
            jitc_raise("replay(): Error, determining size of variable!");

        return (uint32_t) size;
    }

    void alloc(JitBackend backend, uint32_t size, VarType type) {
        alloc(backend, size, type_size[(uint32_t) type]);
    }
    void alloc(JitBackend backend, uint32_t size, uint32_t isize) {
        size_t dsize = ((size_t) size) * ((size_t) isize);
        return alloc(backend, dsize);
    }
    /**
     * Allocates the data for this replay variable. If this is attempted twice,
     * we test Whether the allocated size is sufficient and re-allocate the
     * memory if necessary.
     *
     * In dry run mode, this does not perform allocations, but just changes the
     * \c data_size of the \c ReplayVariable, so that subsequent operations can
     * infer the size of this variable.
     */
    void alloc(JitBackend backend, size_t dsize) {
        AllocType alloc_type =
            backend == JitBackend::CUDA ? AllocType::Device : AllocType::Host;

        if (!data) {
            alloc_size = dsize;
            jitc_log(LogLevel::Debug, "    allocating output of size %zu.",
                     dsize);
            if (!dry_run)
                data = jitc_malloc(alloc_type, dsize);
        } else if (alloc_size < dsize) {
            alloc_size = dsize;
            jitc_log(LogLevel::Debug, "    re-allocating output of size %zu.",
                     dsize);
            if (!dry_run) {
                jitc_free(data);
                data = jitc_malloc(alloc_type, dsize);
            }
        } else {
            // Do not reallocate if the size is enough
        }

        data_size = dsize;
    }

    void free() {
        jitc_free(data);
        data       = nullptr;
        data_size  = 0;
        alloc_size = 0;
    }
};

/// Kernel parameter buffer
static std::vector<void *> kernel_params;

/// ThreadState used during replay
static ThreadState *ts = nullptr;

/// Temporary variables used for replaying a recording.
static std::vector<ReplayVariable> replay_variables;

static ProfilerRegion pr_operation("Replay Operation");
static ProfilerRegion pr_kernel_launch("KernelLaunch");
static ProfilerRegion pr_barrier("Barrier");
static ProfilerRegion pr_memset_async("MemsetAsync");
static ProfilerRegion pr_reduce_expanded("ReduceExpanded");
static ProfilerRegion pr_expand("Expand");
static ProfilerRegion pr_compress("Compress");
static ProfilerRegion pr_memcpy_async("MemcpyAsync");
static ProfilerRegion pr_mkperm("Mkperm");
static ProfilerRegion pr_block_reduce("BlockReduce");
static ProfilerRegion pr_block_prefix_reduce("BlockPrefixReduce");
static ProfilerRegion pr_reduce_dot("ReduceDot");
static ProfilerRegion pr_aggregate("Aggregate");
static ProfilerRegion pr_free("Free");
static ProfilerRegion pr_output("Output");

int Recording::replay(const uint32_t *replay_inputs, uint32_t *replay_outputs) {

    ts = thread_state(backend);

#ifndef NDEBUG
    n_kernels = 0;
#endif

    if (dynamic_cast<RecordThreadState *>(ts) != nullptr)
        jitc_raise("replay(): Tried to replay while recording!");

#if defined(DRJIT_ENABLE_OPTIX)
    OptixShaderBindingTable *tmp_sbt = ts->optix_sbt;
#endif
    scoped_set_context_maybe guard2(ts->context);

    // Initialize replay_variables
    replay_variables.clear();
    replay_variables.reserve(recorded_variables.size());
    for (RecordedVariable &rv : recorded_variables) {
        replay_variables.push_back(ReplayVariable(rv));
    }

    jitc_log(LogLevel::Debug, "replay(): inputs");
    // Populate with input variables
    for (uint32_t i = 0; i < inputs.size(); ++i) {
        Variable *input_variable = jitc_var(replay_inputs[i]);
        replay_variables[inputs[i]].init_from_input(input_variable);
        jitc_log(LogLevel::Debug, "    input %u: r%u maps to slot s%u", i,
                 replay_inputs[i], inputs[i]);
    }

    // The main loop that executes each operation
    for (uint32_t i = 0; i < operations.size(); ++i) {
        Operation &op = operations[i];
        ProfilerPhase profiler(pr_operation);
        if (!op.enabled)
            continue;

        switch (op.type) {
            case OpType::KernelLaunch:
                if (!replay_launch(op))
                    return false;
                break;
            case OpType::Barrier: {
                ProfilerPhase profiler(pr_barrier);
                if (!dry_run)
                    ts->barrier();
            }; break;
            case OpType::MemsetAsync:
                if (!replay_memset_async(op))
                    return false;
                break;
            case OpType::ReduceExpanded:
                if (!replay_reduce_expanded(op))
                    return false;
                break;
            case OpType::Expand:
                if (!replay_expand(op))
                    return false;
                break;
            case OpType::Compress:
                if (!replay_compress(op))
                    return false;
                break;
            case OpType::MemcpyAsync:
                if (!replay_memcpy_async(op))
                    return false;
                break;
            case OpType::Mkperm:
                if (!replay_mkperm(op))
                    return false;
                break;
            case OpType::BlockReduce:
                if (!replay_block_reduce(op))
                    return false;
                break;
            case OpType::BlockPrefixReduce:
                if (!replay_block_prefix_reduce(op))
                    return false;
                break;
            case OpType::ReduceDot:
                if (!replay_reduce_dot(op))
                    return false;
                break;
            case OpType::Aggregate:
                if (!replay_aggregate(op))
                    return false;
                break;
            case OpType::Free: {
                ProfilerPhase profiler(pr_free);

                uint32_t i         = op.dependency_range.first;
                AccessInfo info     = dependencies[i];
                ReplayVariable &rv = replay_variables[info.slot];

                rv.free();

            }; break;
            default:
                jitc_fail(
                    "An unknown operation with type %u has been recorded!",
                    (uint32_t) op.type);
                break;
        }
    }

#if defined(DRJIT_ENABLE_OPTIX)
    ts->optix_sbt = tmp_sbt;
#endif

    if (dry_run)
        return true;

    ProfilerPhase profiler(pr_output);
    // Create output variables
    jitc_log(LogLevel::Debug, "replay(): creating outputs");
    for (uint32_t i = 0; i < outputs.size(); ++i) {
        AccessInfo info     = outputs[i];
        uint32_t slot      = info.slot;
        ReplayVariable &rv = replay_variables[slot];

        if (rv.init == RecordedVarInit::Input) {
            // Use input variable
            jitc_log(LogLevel::Debug,
                     "    output %u: from slot s%u = input[%u]", i, slot,
                     rv.index);
            jitc_assert(rv.data, "replay(): freed an input variable "
                                 "that is passed through!");
            uint32_t var_index = replay_inputs[rv.index];
            jitc_var_inc_ref(var_index);
            replay_outputs[i] = var_index;
        } else if (rv.init == RecordedVarInit::Captured) {
            jitc_log(LogLevel::Debug,
                     "    output %u: from slot s%u = captured r%u", i, slot,
                     rv.index);
            jitc_assert(rv.data, "replay(): freed an input variable "
                                 "that is passed through!");

            jitc_var_inc_ref(rv.index);

            replay_outputs[i] = rv.index;
        } else {
            jitc_log(LogLevel::Debug, "    output %u: from slot s%u", i, slot);
            if (!rv.data)
                jitc_fail("replay(): freed slot %u used for output.", slot);
            replay_outputs[i] = jitc_var_mem_map(backend, info.vtype, rv.data,
                                          rv.size(info.vtype), true);
        }
    }

    // Set \c rv.data to nullptr for next step, where we free all remaining
    // temporary variables
    for (uint32_t i = 0; i < outputs.size(); ++i) {
        AccessInfo info     = outputs[i];
        uint32_t slot      = info.slot;
        ReplayVariable &rv = replay_variables[slot];

        rv.data = nullptr;
    }

    for (ReplayVariable &rv : replay_variables) {
        if (rv.init == RecordedVarInit::Captured) {
            jitc_var_dec_ref(rv.index);
            rv.data = 0;
        } else if (rv.init == RecordedVarInit::None && rv.data) {
            rv.free();
        }
    }

    return true;
}

void RecordThreadState::barrier() {
    if (!paused()) {
        uint32_t start = m_recording.dependencies.size();

        Operation op;
        op.type             = OpType::Barrier;
        op.dependency_range = std::pair(start, start);
        m_recording.operations.push_back(op);
    }

    pause_scope pause(this);
    return m_internal->barrier();
}

/**
 * This function is called every time a pointer is freed using \ref
 * jitc_free. It records the operation and removes the mapping from that
 * pointer to the recorded variable.
 * If the pointer is reused later by another call to \ref jitc_malloc, the
 * \ref RecordThreadState.add_variable function will create a new variable
 * and mapping from the pointer to it.
 */
void RecordThreadState::notify_free(const void *ptr) {
    if (has_variable(ptr)) {
        jitc_log(LogLevel::Debug, "record(): jitc_free(ptr=%p)", ptr);

        uint32_t start = m_recording.dependencies.size();
        add_in_param(ptr, VarType::Void, false);
        uint32_t end = m_recording.dependencies.size();

        Operation op;
        op.type             = OpType::Free;
        op.dependency_range = std::pair(start, end);

        /// Removes the pointer from the \c ptr_to_slot mapping. The next
        /// operation using this memory region will have to add it using \c
        /// add_variable.
        ptr_to_slot.erase(ptr);
    }
}

/**
 * Adds an input of the recording.
 * This is adds the slot of that variable to the \ref Recording.inputs
 * vector.
 */
void RecordThreadState::add_input(uint32_t input) {
    try {
        uint32_t input_index = m_recording.inputs.size();
        Variable *v          = jitc_var(input);

        uint32_t slot = add_variable(v->data);
        RecordedVariable &rv = m_recording.recorded_variables[slot];
        rv.state             = RecordedVarState::Input;
        rv.init              = RecordedVarInit::Input;
        rv.index             = input_index;
        rv.type              = (VarType) v->type;

        jitc_log(LogLevel::Debug,
                 "record(): Adding variable %u <%p> input %u to slot s%u",
                 input, v->data, input_index, slot);
        m_recording.inputs.push_back(slot);
    } catch (...) {
        record_exception();
    }
}
/**
 * Adds an output to the recording.
 * The output can be seen as a final operation, which also has to infer the
 * size of its input variables.
 * Therefore, we record the full \ref AccessInfo for each output variable.
 */
void RecordThreadState::add_output(uint32_t output) {
    try {
        uint32_t output_index = m_recording.outputs.size();
        Variable *v           = jitc_var(output);
        uint32_t slot;
        if (!has_variable(v->data)) {
            slot = capture_variable(output);
        } else {
            slot = get_variable(v->data);
        }

        jitc_log(LogLevel::Trace,
                 "record(): Adding variable %u output %u to slot s%u", output,
                 output_index, slot);
        AccessInfo info;
        info.slot  = slot;
        info.vtype = (VarType) v->type;
        m_recording.outputs.push_back(info);
    } catch (...) {
        record_exception();
    }
}

bool RecordThreadState::pause() {
    bool tmp = m_paused;
    m_paused = true;
    return tmp;
}
bool RecordThreadState::resume() {
    bool tmp = m_paused;
    m_paused = false;
    return tmp;
}

Task *RecordThreadState::launch(Kernel kernel, KernelKey *key,
                                XXH128_hash_t hash, uint32_t size,
                                std::vector<void *> *kernel_params,
                                const std::vector<uint32_t> *kernel_param_ids) {
    if (!paused()) {
        try {
            record_launch(kernel, key, hash, size, kernel_params,
                          kernel_param_ids);
        } catch (...) {
            record_exception();
        }
    }
    pause_scope pause(this);
    return m_internal->launch(kernel, key, hash, size, kernel_params, nullptr);
}

void RecordThreadState::record_launch(
    Kernel kernel, KernelKey *key, XXH128_hash_t hash, uint32_t size,
    std::vector<void *> *kernel_params,
    const std::vector<uint32_t> *kernel_param_ids) {
    uint32_t kernel_param_offset = backend == JitBackend::CUDA ? 1 : 3;

    size_t input_size = 0;
    size_t ptr_size   = 0;

#ifndef NDEBUG
    jitc_log(LogLevel::Debug, "record(): recording kernel %u %016llx",
             m_recording.n_kernels++, (unsigned long long) hash.high64);
#else
    jitc_log(LogLevel::Debug, "record(): recording kernel %016llx",
             (unsigned long long) hash.high64);
#endif

    uint32_t start = m_recording.dependencies.size();
    for (uint32_t param_index = 0; param_index < kernel_param_ids->size();
         param_index++) {

        bool pointer_access = false;
        uint32_t index      = kernel_param_ids->at(param_index);
        Variable *v         = jitc_var(index);

        // Note, the ptr might not come from the variable but the \c
        // ScheduledVariable if it is an output.
        void *ptr = kernel_params->at(kernel_param_offset + param_index);
        ParamType param_type = (ParamType) v->param_type;

        if (param_type == ParamType::Input &&
            (VarType) v->type != VarType::Pointer) {
            input_size = std::max(input_size, (size_t) v->size);
        }

        // In case the variable is a pointer, we follow the pointer to the
        // source and record the source size.
        // NOTE: this means that `v` is now the source variable
        if ((VarType) v->type == VarType::Pointer) {
            jitc_assert(v->is_literal(),
                        "record(): Recording non-literal pointers are "
                        "not yet supported!");
            jitc_assert(param_type != ParamType::Output,
                        "record(): A pointer, pointing to a kernel "
                        "ouptut is not yet supported!");

            // Follow pointer
            uint32_t ptr_index = index;
            index              = v->dep[3];
            v                  = jitc_var(index);
            if (v->data != ptr)
                jitc_fail("record(): Tried to record variable r%u, "
                          "pointing to r%u, but their memory address "
                          "did not match! (%p != %p)",
                          ptr_index, index, ptr, v->data);

            pointer_access = true;
            ptr_size       = std::max(ptr_size, (size_t) v->size);
        }

        uint32_t slot;
        if (param_type == ParamType::Input) {
            if (has_variable(ptr)) {
                slot = get_variable(ptr);
            } else {
                slot = capture_variable(index);
            }

        } else if (param_type == ParamType::Output) {
            slot = add_variable(ptr);
        } else
            jitc_fail("Parameter Type not supported!");

        if (pointer_access) {
            jitc_log(LogLevel::Debug,
                     " %s recording param %u = var(%u, points to r%u, "
                     "size=%u, data=%p, type=%s) at s%u",
                     param_type == ParamType::Output ? "<-" : "->", param_index,
                     kernel_param_ids->at(param_index), index, v->size, ptr,
                     type_name[(uint32_t) v->type], slot);
        } else {
            jitc_log(LogLevel::Debug,
                     " %s recording param %u = var(%u, size=%u, "
                     "data=%p, type=%s) at s%u",
                     param_type == ParamType::Output ? "<-" : "->", param_index,
                     kernel_param_ids->at(param_index), v->size, ptr,
                     type_name[(uint32_t) v->type], slot);
        }

        jitc_log(LogLevel::Debug, "    label=%s", jitc_var_label(index));

        AccessInfo info;
        info.slot           = slot;
        info.type           = param_type;
        info.pointer_access = pointer_access;
        info.vtype          = (VarType) v->type;
        add_param(info);
    }
    uint32_t end = m_recording.dependencies.size();

    Operation op;
    op.type             = OpType::KernelLaunch;
    op.dependency_range = std::pair(start, end);

    op.kernel.kernel   = kernel;
    op.kernel.key      = (KernelKey *) std::malloc(sizeof(KernelKey));
    size_t str_size    = buffer.size() + 1;
    op.kernel.key->str = (char *) malloc_check(str_size);
    std::memcpy(op.kernel.key->str, key->str, str_size);
    op.kernel.key->device = key->device;
    op.kernel.key->flags  = key->flags;
    op.kernel.key->high64 = key->high64;

    op.size = size;

    // If this kernel uses optix, we have to copy the shader binding table for
    // replaying
#if defined(DRJIT_ENABLE_OPTIX)
    if (uses_optix) {
        op.uses_optix = true;

        pause_scope pause(this);
        // Copy SBT
        op.sbt = new OptixShaderBindingTable();
        std::memcpy(op.sbt, optix_sbt, sizeof(OptixShaderBindingTable));

        // Copy hit groups
        size_t hit_group_size = optix_sbt->hitgroupRecordStrideInBytes *
                                optix_sbt->hitgroupRecordCount;
        op.sbt->hitgroupRecordBase =
            jitc_malloc(AllocType::Device, hit_group_size);
        jitc_memcpy(backend, op.sbt->hitgroupRecordBase,
                    optix_sbt->hitgroupRecordBase, hit_group_size);

        // Copy miss groups
        size_t miss_group_size =
            optix_sbt->missRecordStrideInBytes * optix_sbt->missRecordCount;
        op.sbt->missRecordBase =
            jitc_malloc(AllocType::Device, miss_group_size);
        jitc_memcpy(backend, op.sbt->missRecordBase, optix_sbt->missRecordBase,
                    miss_group_size);
    }
#endif

    // Record max_input_size if we have only pointer inputs.
    // Therefore, if max_input_size > 0 we know this at replay.
    if (input_size == 0) {
        jitc_log(LogLevel::Debug, "    input_size(pointers)=%zu", ptr_size);
        op.input_size = ptr_size;
    } else {
        jitc_log(LogLevel::Debug, "    input_size(direct)=%zu", input_size);
        op.input_size = input_size;
    }

    // Reset input size if ratio/fraction is not valid
    if (op.input_size > 0) {
        if (op.size > op.input_size && op.size % op.input_size != 0)
            op.input_size = 0;
        if (op.size < op.input_size && op.input_size % op.size != 0)
            op.input_size = 0;
    }

    if (op.input_size) {
        if (op.size > op.input_size)
            jitc_log(LogLevel::Debug, "    size=input_size*%zu",
                     op.size / op.input_size);
        else if (op.size < op.input_size)
            jitc_log(LogLevel::Debug, "    size=input_size/%zu",
                     op.input_size / op.size);
    } else {
        jitc_log(LogLevel::Debug, "    input size could not be determined "
                                  "by input and is recorded as is.");
    }

    m_recording.operations.push_back(op);

    // Re-assign optix specific variables to internal thread state since they
    // might have changed
#if defined(DRJIT_ENABLE_OPTIX)
    m_internal->optix_pipeline = optix_pipeline;
    m_internal->optix_sbt      = optix_sbt;
#endif
}

int Recording::replay_launch(Operation &op) {
    ProfilerPhase profiler(pr_kernel_launch);

    // Reconstruct the \ref kernel_params for this launch given the allocations
    // when replaying.
    kernel_params.clear();

    if (backend == JitBackend::CUDA) {
        // First parameter contains kernel size. Assigned later.
        kernel_params.push_back(nullptr);
    } else {
        // First 3 parameters reserved for: kernel ptr, size, ITT identifier
        for (int i = 0; i < 3; ++i)
            kernel_params.push_back(nullptr);
    }

    // Infer kernel launch size

    // We first infer the size of the input variable, given how they are
    // accessed by the kernel (i.e. as what type). While doing this, we also
    // record the maximum size of variables accessed directly and through
    // pointers separately. This will then be used when inferring the kernel
    // launch size.

    jitc_log(LogLevel::Debug, "replay(): inferring input size");

    // Size of direct input variables
    uint32_t input_size = 0;
    // Size of variables referenced by pointers
    uint32_t ptr_size = 0;

    for (uint32_t j = op.dependency_range.first; j < op.dependency_range.second;
         ++j) {
        AccessInfo info     = dependencies[j];
        ReplayVariable &rv = replay_variables[info.slot];

        if (info.type == ParamType::Input) {
            uint32_t size = rv.size(info.vtype);
            jitc_log(LogLevel::Debug, "    s%u size=%u", info.slot, size);

            if (rv.data == nullptr && !dry_run)
                jitc_raise("replay(): Kernel input variable s%u not allocated!",
                           info.slot);

            if (!info.pointer_access)
                input_size = std::max(input_size, size);
            else
                ptr_size = std::max(ptr_size, size);
        }
    }

    // Given the maximum size of the input variables (accessed directly and
    // through pointers) we can infer the kernel launch size. We assume that the
    // launch size is either a multiple of the maximum input variable (directly
    // accessed) or if the kernel has no direct input variable, a multiple or
    // fraction of the largest variable accessed through a pointer. If the
    // launch size could not be inferred, we use the recorded size.

    uint32_t launch_size = 0;
    if (op.input_size > 0) {
        launch_size = input_size != 0 ? input_size : ptr_size;
        // Apply the factor
        if (op.size > op.input_size && (op.size % op.input_size == 0)) {
            uint32_t ratio = op.size / op.input_size;

            jitc_log(LogLevel::Debug,
                     "replay(): Inferring launch size by heuristic, "
                     "launch_size=%u, ratio=%u",
                     launch_size, ratio);
            launch_size = launch_size * ratio;
        } else if (op.input_size % op.size == 0) {
            uint32_t fraction = op.input_size / op.size;
            jitc_log(LogLevel::Debug,
                     "replay(): Inferring launch size by heuristic, "
                     "launch_size(%u), fraction=%u",
                     launch_size, fraction);
            launch_size = launch_size / fraction;
        }
    }
    if (launch_size == 0) {
        jitc_log(LogLevel::Debug, "replay(): Could not infer launch "
                                  "size, using recorded size");
        launch_size = op.size;
    }

    // Allocate output variables for kernel launch. The assumption here is that
    // for every kernel launch, the inputs are already allocated. Therefore we
    // only allocate output variables, which have the same size as the kernel.
    for (uint32_t j = op.dependency_range.first; j < op.dependency_range.second;
         ++j) {
        AccessInfo info     = dependencies[j];
        ReplayVariable &rv = replay_variables[info.slot];

        if (info.type == ParamType::Input) {
            uint32_t size = rv.size(info.vtype);
            jitc_log(LogLevel::Debug, " -> param s%u is_pointer=%u size=%u",
                     info.slot, info.pointer_access, size);
        } else {
            jitc_log(LogLevel::Debug, " <- param s%u is_pointer=%u", info.slot,
                     info.pointer_access);
        }

        if (info.type == ParamType::Output) {
            rv.alloc(backend, launch_size, info.vtype);
        }
        jitc_assert(rv.data != nullptr || dry_run,
                    "replay(): Encountered nullptr in kernel parameters.");
        kernel_params.push_back(rv.data);
    }

    // Change kernel size in `kernel_params`
    if (backend == JitBackend::CUDA)
        kernel_params[0] = (void *) (uintptr_t) launch_size;

    if (!dry_run) {
#ifndef NDEBUG
        jitc_log(LogLevel::Debug,
                 "replay(): launching kernel %u %016llx [%u]%s", n_kernels++,
                 (unsigned long long) op.kernel.hash.high64, launch_size,
                 op.uses_optix ? " uses optix" : "");
#else
        jitc_log(LogLevel::Debug, "replay(): launching kernel %016llx [%u]%s",
                 (unsigned long long) op.kernel.hash.high64, launch_size,
                 op.uses_optix ? " uses optix" : "");
#endif

        std::vector<uint32_t> kernel_calls;
        Kernel kernel = op.kernel.kernel;
#if defined(DRJIT_ENABLE_OPTIX)
        if (op.uses_optix) {
            uses_optix    = true;
            ts->optix_sbt = op.sbt;
        }
#endif
        ts->launch(kernel, op.kernel.key, op.kernel.hash, launch_size,
                   &kernel_params, nullptr);
#if defined(DRJIT_ENABLE_OPTIX)
        if (op.uses_optix)
            uses_optix = false;
#endif
    }

    return true;
}

void RecordThreadState::record_expand(uint32_t index) {
    Variable *v = jitc_var(index);

    uint32_t dst_slot        = get_variable(v->data);
    const RecordedVariable &rv = m_recording.recorded_variables[dst_slot];
    if (rv.last_memset == 0)
        jitc_fail("record(): Could not infer last memset operation of r%u s%u, "
                  "to construct expand operation!",
                  index, dst_slot);
    Operation &memset = m_recording.operations[rv.last_memset - 1];
    memset.enabled    = false;

    Operation op;
    uint32_t start = m_recording.dependencies.size();
    add_out_param(dst_slot, v->type);
    if (rv.last_memcpy) {
        Operation &memcpy = m_recording.operations[rv.last_memcpy - 1];
        memcpy.enabled    = false;

        uint32_t dependency_index = memcpy.dependency_range.first;
        AccessInfo src_info        = m_recording.dependencies[dependency_index];

        add_in_param(src_info.slot);

        jitc_log(LogLevel::Debug, "record(): expand(dst=s%u, src=s%u)",
                 dst_slot, src_info.slot);

        op.size = memcpy.size / type_size[v->type];
    } else {
        // Case where in jitc_var_expand, v->is_literal && v->literal ==
        // identity
        uint64_t identity =
            jitc_reduce_identity((VarType) v->type, (ReduceOp) v->reduce_op);

        jitc_log(LogLevel::Debug,
                 "record(): expand(dst=s%u, src=literal 0x%llx)", dst_slot,
                 (unsigned long long) identity);

        op.size = v->size;
    }
    uint32_t end = m_recording.dependencies.size();

    op.type             = OpType::Expand;
    op.dependency_range = std::pair(start, end);
    op.data             = memset.data;
    m_recording.operations.push_back(op);

    m_recording.requires_dry_run = true;
}

int Recording::replay_expand(Operation &op) {
    ProfilerPhase profiler(pr_expand);

    jitc_log(LogLevel::Debug, "replay(): expand");

    uint32_t dependency_index = op.dependency_range.first;
    bool memcpy = op.dependency_range.second == dependency_index + 2;

    AccessInfo dst_info    = dependencies[dependency_index];
    ReplayVariable &dst_rv = replay_variables[dst_info.slot];
    VarType vt             = dst_info.vtype;
    uint32_t tsize         = type_size[(uint32_t) vt];

    uint32_t size;
    void *src_ptr = 0;
    if (memcpy) {
        AccessInfo src_info     = dependencies[dependency_index + 1];
        ReplayVariable &src_rv = replay_variables[src_info.slot];
        size                   = src_rv.size(vt);
        jitc_log(LogLevel::Debug, "jitc_memcpy_async(dst=%p, src=%p, size=%zu)",
                 dst_rv.data, src_rv.data, (size_t) size * tsize);

        src_ptr = src_rv.data;
    } else {
        // Case where in jitc_var_expand, v->is_literal &&
        // v->literal == identity
        size = op.size;
        jitc_log(LogLevel::Debug,
                 "jitc_memcpy_async(dst=%p, src= literal 0x%llx, "
                 "size=%zu)",
                 dst_rv.data, (unsigned long long) op.data,
                 (size_t) size * tsize);
    }

    if (size != op.size)
        return false;

    auto [workers, replication_per_worker] =
        jitc_llvm_expand_replication_factor(size, tsize);

    size_t new_size = size * (size_t) replication_per_worker * (size_t) workers;

    dst_rv.alloc(backend, new_size, dst_info.vtype);

    if (!dry_run)
        ts->memset_async(dst_rv.data, new_size, tsize, &op.data);

    if (!dry_run && memcpy)
        ts->memcpy_async(dst_rv.data, src_ptr, (size_t) size * tsize);

    dst_rv.data_size = size * type_size[(uint32_t) dst_info.vtype];

    return true;
}

/// LLVM: Notify the thread state, that a variable has been expanded using
/// \c jitc_var_expand. This is required to record the ThreadState.
void RecordThreadState::notify_expand(uint32_t index){
    // Reductions in LLVM might be split into three operations. First the
    // variable is expanded by its size times the number of workers + 1 Then the
    // kernel writes into the expanded variable with some offset, and finally
    // the variable is reduced. The expand operation allocates a new memory
    // region and copies the old content into it. We catch this case if the
    // input variable of a kernel has a reduce_op associated with it.
    if (!paused()){
        try {
            record_expand(index);
        } catch (...) {
            record_exception();
        }
    }
}

/// LLVM: reduce a variable that was previously expanded due to
/// dr.ReduceOp.Expand
void RecordThreadState::reduce_expanded(VarType vt, ReduceOp reduce_op,
                                        void *data, uint32_t exp,
                                        uint32_t size) {
    if (!paused()) {
        try {
            record_reduce_expanded(vt, reduce_op, data, exp, size);
        } catch (...) {
            record_exception();
        }
    }
    pause_scope pause(this);
    return m_internal->reduce_expanded(vt, reduce_op, data, exp, size);
}

void RecordThreadState::record_reduce_expanded(VarType vt, ReduceOp reduce_op,
                                               void *data, uint32_t exp,
                                               uint32_t size) {
    jitc_log(LogLevel::Debug,
             "record(): reduce_expanded(vt=%s, op=%u, data=%p, exp=%u, "
             "size=%u)",
             type_name[(uint32_t) vt], (uint32_t) reduce_op, data, exp, size);

    uint32_t start = m_recording.dependencies.size();
    add_out_param(data, vt);
    uint32_t end = m_recording.dependencies.size();

    Operation op;
    op.type             = OpType::ReduceExpanded;
    op.dependency_range = std::pair(start, end);
    op.rtype            = reduce_op;
    op.size             = size;
    m_recording.operations.push_back(op);
}

int Recording::replay_reduce_expanded(Operation &op) {
    ProfilerPhase profiler(pr_reduce_expanded);

    uint32_t dependency_index = op.dependency_range.first;
    AccessInfo data_info       = dependencies[dependency_index];

    ReplayVariable &data_var = replay_variables[data_info.slot];

    VarType vt       = data_info.vtype;
    ReduceOp rop     = op.rtype;
    uint32_t size    = data_var.size(vt);
    uint32_t tsize   = type_size[(uint32_t) vt];
    uint32_t workers = pool_size() + 1;

    uint32_t replication_per_worker = size == 1u ? (64u / tsize) : 1u;

    jitc_log(
        LogLevel::Debug,
        "replay(): reduce_expanded(vt=%s, op=%u, data=%p, exp=%u, size=%u)",
        type_name[(uint32_t) vt], (uint32_t) rop, data_var.data,
        replication_per_worker * workers, size);

    if (!dry_run)
        ts->reduce_expanded(vt, rop, data_var.data,
                            replication_per_worker * workers, size);

    return true;
}

/// Perform a synchronous copy operation
void RecordThreadState::memcpy(void *dst, const void *src, size_t size) {
    jitc_log(LogLevel::Debug, "record(): memcpy(dst=%p, src=%p, size=%zu)", dst,
             src, size);
    pause_scope pause(this);
    return m_internal->memcpy(dst, src, size);
}

/// Perform an asynchronous copy operation
void RecordThreadState::memcpy_async(void *dst, const void *src, size_t size) {
    jitc_log(LogLevel::Debug,
             "record(): memcpy_async(dst=%p, src=%p, size=%zu)", dst, src,
             size);
    bool has_var = has_variable(src);
    try {
        if (!paused() && has_var) {

            uint32_t src_id;
            src_id = get_variable(src);

            uint32_t dst_id = add_variable(dst);
            m_recording.recorded_variables[dst_id].last_memcpy =
                m_recording.operations.size() + 1;

            uint32_t start = m_recording.dependencies.size();
            add_in_param(src_id);
            add_out_param(dst_id, m_recording.recorded_variables[src_id].type);
            uint32_t end = m_recording.dependencies.size();

            Operation op;
            op.type             = OpType::MemcpyAsync;
            op.dependency_range = std::pair(start, end);
            op.size             = size;
            m_recording.operations.push_back(op);
        }
    } catch (...) {
        record_exception();
    }
    {
        pause_scope pause(this);
        m_internal->memcpy_async(dst, src, size);
    }
    try {
        if (!paused() && !has_var) {
            // If we did not know the source variable, this memcpy might be
            // coming from \c jitc_call_upload.
            // If that is the case, we have to capture the offset buffer.
            // Since the pointer might be used, for example by an aggregate call
            // (nested calls), we have to overwrite the RecordedVariable.
            CallData *call = nullptr;
            for (CallData *tmp : calls_assembled) {
                if (tmp->offset == dst) {
                    call = tmp;
                    break;
                }
            }
            if (call) {
                capture_call_offset(dst, size);
                jitc_log(LogLevel::Debug, "    captured call offset");
            } else {
                jitc_raise(
                    "record(): Tried to record a memcpy_async operation, "
                    "but the source pointer %p was not known.",
                    src);
            }
        }
    } catch (...) {
        record_exception();
    }
}

int Recording::replay_memcpy_async(Operation &op) {
    ProfilerPhase profiler(pr_memcpy_async);

    uint32_t dependency_index = op.dependency_range.first;
    AccessInfo src_info        = dependencies[dependency_index];
    AccessInfo dst_info        = dependencies[dependency_index + 1];

    ReplayVariable &src_var = replay_variables[src_info.slot];
    ReplayVariable &dst_var = replay_variables[dst_info.slot];

    dst_var.alloc(backend, src_var.data_size);

    jitc_log(LogLevel::Debug,
             "replay(): memcpy_async(dst=%p, src=%p, size=%zu)", dst_var.data,
             src_var.data, src_var.data_size);

    if (!dry_run)
        ts->memcpy_async(dst_var.data, src_var.data, src_var.data_size);

    return true;
}

/// Fill a device memory region with constants of a given type
void RecordThreadState::memset_async(void *ptr, uint32_t size, uint32_t isize,
                                     const void *src) {
    if (!paused()) {
        try {
            record_memset_async(ptr, size, isize, src);
        } catch (...) {
            record_exception();
        }
    }
    pause_scope pause(this);
    return m_internal->memset_async(ptr, size, isize, src);
}

void RecordThreadState::record_memset_async(void *ptr, uint32_t size,
                                            uint32_t isize, const void *src) {
    jitc_log(LogLevel::Debug,
             "record(): memset_async(ptr=%p, size=%u, "
             "isize=%u, src=%p)",
             ptr, size, isize, src);
    jitc_assert(isize <= 8,
                "record(): Tried to call memset_async with isize=%u, "
                "only isize<=8 is supported!",
                isize);

    uint32_t ptr_id = add_variable(ptr);
    m_recording.recorded_variables[ptr_id].last_memset =
        m_recording.operations.size() + 1;

    uint32_t start = m_recording.dependencies.size();
    add_out_param(ptr_id, VarType::Void);
    uint32_t end = m_recording.dependencies.size();

    Operation op;
    op.type             = OpType::MemsetAsync;
    op.dependency_range = std::pair(start, end);
    op.size             = size;
    op.input_size       = isize;
    std::memcpy(&op.data, src, isize);

    m_recording.operations.push_back(op);
}

int Recording::replay_memset_async(Operation &op) {
    ProfilerPhase profiler(pr_memset_async);

    uint32_t dependency_index = op.dependency_range.first;

    AccessInfo ptr_info = dependencies[dependency_index];

    ReplayVariable &ptr_var = replay_variables[ptr_info.slot];
    ptr_var.alloc(backend, op.size, op.input_size);

    uint32_t size = ptr_var.size(op.input_size);

    jitc_log(LogLevel::Debug,
             "replay(): memset_async(ptr=%p, size=%u, "
             "isize=%zu, src=%p)",
             ptr_var.data, size, op.input_size, &op.data);
    if (!dry_run)
        ts->memset_async(ptr_var.data, size, op.input_size, &op.data);

    return true;
}

/// Mask compression
uint32_t RecordThreadState::compress(const uint8_t *in, uint32_t size,
                                     uint32_t *out) {
    if (!paused()) {
        try {
            record_compress(in, size, out);
        } catch (...) {
            record_exception();
        }
    }
    pause_scope pause(this);
    return m_internal->compress(in, size, out);
}

void RecordThreadState::record_compress(const uint8_t *in, uint32_t size,
                                        uint32_t *out) {
    jitc_assert(has_variable(in),
                "record(): Input variable has not been recorded!");
    jitc_log(LogLevel::Debug, "record(): compress(in=%p, size=%u, out=%p)", in,
             size, out);

    uint32_t start = m_recording.dependencies.size();
    add_in_param(in);
    add_out_param(out, VarType::UInt32);
    uint32_t end = m_recording.dependencies.size();

    Operation op;
    op.type             = OpType::Compress;
    op.dependency_range = std::pair(start, end);
    op.size             = size;
    m_recording.operations.push_back(op);
}

int Recording::replay_compress(Operation &op) {
    ProfilerPhase profiler(pr_compress);

    uint32_t dependency_index = op.dependency_range.first;

    AccessInfo in_info  = dependencies[dependency_index];
    AccessInfo out_info = dependencies[dependency_index + 1];

    ReplayVariable &in_rv  = replay_variables[in_info.slot];
    ReplayVariable &out_rv = replay_variables[out_info.slot];

    uint32_t size = in_rv.size(in_info.vtype);
    out_rv.alloc(backend, size, out_info.vtype);

    if (dry_run) {
        jitc_log(LogLevel::Warn,
                 "replay(): Could not infer the size of the result "
                 "to a dr.compress call during dry run. The "
                 "function will be re-recorded, which causes some "
                 "overhead. This occured most likely due to a "
                 "scatter_reduce call in a LLVM kernel in the same "
                 "frozen function. To avoid this try to split the "
                 "function into two frozen functions.");
        // We return false (the dry run failed), which will trigger
        // a re-recording of the frozen function. This will be
        // slower than not freezing the function but cannot be
        // avoided.
        return false;
    }

    jitc_log(LogLevel::Debug, "replay(): compress(in=%p, size=%u, out=%p)",
             in_rv.data, size, (uint32_t *) out_rv.data);

    uint32_t out_size =
        ts->compress((uint8_t *) in_rv.data, size, (uint32_t *) out_rv.data);

    out_rv.data_size = out_size * type_size[(uint32_t) out_info.vtype];

    return true;
}

/// Compute a permutation to reorder an integer array into discrete groups
uint32_t RecordThreadState::mkperm(const uint32_t *values, uint32_t size,
                                   uint32_t bucket_count, uint32_t *perm,
                                   uint32_t *offsets) {
    if (!paused()) {
        try {
            record_mkperm(values, size, bucket_count, perm, offsets);
        } catch (...) {
            record_exception();
        }
    }
    pause_scope pause(this);
    return m_internal->mkperm(values, size, bucket_count, perm, offsets);
}

void RecordThreadState::record_mkperm(const uint32_t *values, uint32_t size,
                                      uint32_t bucket_count, uint32_t *perm,
                                      uint32_t *offsets) {
    if (has_variable(values)) {
        jitc_log(LogLevel::Debug,
                 "record(): mkperm(values=%p, size=%u, "
                 "bucket_count=%u, perm=%p, offsets=%p)",
                 values, size, bucket_count, perm, offsets);

        uint32_t start = m_recording.dependencies.size();
        add_in_param(values);
        add_out_param(perm, VarType::UInt32);
        add_out_param(offsets, VarType::UInt32);
        uint32_t end = m_recording.dependencies.size();

        Operation op;
        op.type             = OpType::Mkperm;
        op.dependency_range = std::pair(start, end);
        op.size             = size;
        op.bucket_count     = bucket_count;
        m_recording.operations.push_back(op);
    }
}

int Recording::replay_mkperm(Operation &op) {
    ProfilerPhase profiler(pr_mkperm);

    uint32_t dependency_index = op.dependency_range.first;
    AccessInfo values_info     = dependencies[dependency_index];
    AccessInfo perm_info       = dependencies[dependency_index + 1];
    AccessInfo offsets_info    = dependencies[dependency_index + 2];

    ReplayVariable &values_var  = replay_variables[values_info.slot];
    ReplayVariable &perm_var    = replay_variables[perm_info.slot];
    ReplayVariable &offsets_var = replay_variables[offsets_info.slot];

    uint32_t size         = values_var.size(values_info.vtype);
    uint32_t bucket_count = op.bucket_count;

    perm_var.alloc(backend, size, perm_info.vtype);

    offsets_var.alloc(backend, bucket_count * 4 + 1, offsets_info.vtype);

    jitc_log(LogLevel::Debug,
             "replay(): mkperm(values=%p, size=%u, "
             "bucket_count=%u, perm=%p, offsets=%p)",
             values_var.data, size, bucket_count, perm_var.data,
             offsets_var.data);

    if (!dry_run)
        ts->mkperm((uint32_t *) values_var.data, size, bucket_count,
                   (uint32_t *) perm_var.data, (uint32_t *) offsets_var.data);

    return true;
}

/// Sum over elements within blocks
void RecordThreadState::block_reduce(VarType vt, ReduceOp op, uint32_t size,
                                     uint32_t block_size, const void *in,
                                     void *out) {
    if (!paused()) {
        try {
            record_block_reduce(vt, op, size, block_size, in, out);
        } catch (...) {
            record_exception();
        }
    }
    pause_scope pause(this);
    return m_internal->block_reduce(vt, op, size, block_size, in, out);
}

void RecordThreadState::record_block_reduce(VarType vt, ReduceOp rop,
                                            uint32_t size, uint32_t block_size,
                                            const void *in, void *out) {
    jitc_log(LogLevel::Debug,
             "record(): block_reduce(vt=%u, op=%u, size=%u, block_size=%u, "
             "in=%p, out=%p)",
             (uint32_t) vt, (uint32_t) rop, size, block_size, in, out);

    uint32_t start = m_recording.dependencies.size();
    add_in_param(in, vt);
    add_out_param(out, vt);
    uint32_t end = m_recording.dependencies.size();

    Operation op;
    op.type             = OpType::BlockReduce;
    op.dependency_range = std::pair(start, end);
    op.size             = size;
    op.input_size       = block_size;
    op.rtype            = rop;
    m_recording.operations.push_back(op);

    m_recording.requires_dry_run = true;
}

int Recording::replay_block_reduce(Operation &op) {
    ProfilerPhase profiler(pr_block_reduce);

    uint32_t dependency_index = op.dependency_range.first;
    AccessInfo in_info         = dependencies[dependency_index];
    AccessInfo out_info        = dependencies[dependency_index + 1];

    ReplayVariable &in_var  = replay_variables[in_info.slot];
    ReplayVariable &out_var = replay_variables[out_info.slot];

    uint32_t size = in_var.size(in_info.vtype);

    uint32_t block_size = op.input_size;
    if (op.input_size == op.size)
        block_size = size;

    if ((size % block_size) != 0) {
        if (dry_run)
            return false;
        jitc_fail("replay(): The size (%u) of the argument to a "
                  "block_sum has to be divisible by the block_size (%u)!",
                  size, block_size);
    }

    uint32_t output_size = size / block_size;

    out_var.alloc(backend, output_size, out_info.vtype);

    jitc_log(LogLevel::Debug,
             "replay(): block_reduce(vt=%u, op=%u, size=%u, block_size=%u, "
             "in=%p, out=%p)",
             (uint32_t) out_info.vtype, (uint32_t) op.rtype, size, block_size,
             in_var.data, out_var.data);

    if (!dry_run)
        ts->block_reduce(out_info.vtype, op.rtype, size, block_size,
                         in_var.data, out_var.data);

    return true;
}

void RecordThreadState::block_prefix_reduce(VarType vt, ReduceOp op,
                                            uint32_t size, uint32_t block_size,
                                            bool exclusive, bool reverse,
                                            const void *in, void *out) {
    if (!paused()) {
        try {
            record_block_prefix_reduce(vt, op, size, block_size, exclusive,
                                       reverse, in, out);
        } catch (...) {
            record_exception();
        }
    }
    pause_scope pause(this);
    return m_internal->block_prefix_reduce(vt, op, size, block_size, exclusive,
                                           reverse, in, out);
}

void RecordThreadState::record_block_prefix_reduce(VarType vt, ReduceOp rop,
                                                   uint32_t size,
                                                   uint32_t block_size,
                                                   bool exclusive, bool reverse,
                                                   const void *in, void *out) {
    jitc_log(
        LogLevel::Debug,
        "record(): block_prefix_reduce(vt=%u, op=%u, size=%u, block_size=%u, "
        "exclusive=%u, reverse=%u, in=%p, out=%p)",
        (uint32_t) vt, (uint32_t) rop, size, block_size, exclusive, reverse, in,
        out);

    uint32_t start = m_recording.dependencies.size();
    add_in_param(in, vt);
    add_out_param(out, vt);
    uint32_t end = m_recording.dependencies.size();

    Operation op;
    op.type             = OpType::BlockPrefixReduce;
    op.dependency_range = std::pair(start, end);
    op.size             = size;
    op.input_size       = block_size;
    op.prefix_reduce    = { /*rtype=*/rop, /*exclusive=*/exclusive,
                         /*reverse=*/reverse };
    m_recording.operations.push_back(op);

    m_recording.requires_dry_run = true;
}

int Recording::replay_block_prefix_reduce(Operation &op) {
    ProfilerPhase profiler(pr_block_reduce);

    uint32_t dependency_index = op.dependency_range.first;
    AccessInfo in_info         = dependencies[dependency_index];
    AccessInfo out_info        = dependencies[dependency_index + 1];

    ReplayVariable &in_var  = replay_variables[in_info.slot];
    ReplayVariable &out_var = replay_variables[out_info.slot];

    uint32_t size = in_var.size(in_info.vtype);

    uint32_t block_size = op.input_size;
    if (op.input_size == op.size)
        block_size = size;

    if ((size % block_size) != 0) {
        if (dry_run)
            return false;
        jitc_fail("replay(): The size (%u) of the argument to a "
                  "block_sum has to be divisible by the block_size (%u)!",
                  size, block_size);
    }

    uint32_t output_size = size;

    out_var.alloc(backend, output_size, out_info.vtype);

    jitc_log(LogLevel::Debug,
             "replay(): block_prefix_reduce(vt=%u, op=%u, size=%u, block_size=%u, "
             "exclusive=%u, reverse=%u, in=%p, out=%p)",
             (uint32_t) out_info.vtype, (uint32_t) op.prefix_reduce.rtype, size,
             block_size, op.prefix_reduce.exclusive, op.prefix_reduce.reverse,
             in_var.data, out_var.data);

    if (!dry_run)
        ts->block_prefix_reduce(out_info.vtype, op.prefix_reduce.rtype, size,
                                block_size, op.prefix_reduce.exclusive,
                                op.prefix_reduce.reverse, in_var.data,
                                out_var.data);

    return true;
}

/// Compute a dot product of two equal-sized arrays
void RecordThreadState::reduce_dot(VarType type, const void *ptr_1,
                                   const void *ptr_2, uint32_t size,
                                   void *out) {
    if (!paused()) {
        try {
            record_reduce_dot(type, ptr_1, ptr_2, size, out);
        } catch (...) {
            record_exception();
        }
    }
    pause_scope pause(this);
    return m_internal->reduce_dot(type, ptr_1, ptr_2, size, out);
}

void RecordThreadState::record_reduce_dot(VarType type, const void *ptr_1,
                                          const void *ptr_2, uint32_t size,
                                          void *out) {
    ProfilerPhase profiler(pr_reduce_dot);

    jitc_log(
        LogLevel::Debug,
        "record(): reduce_dot(type=%s, ptr_1=%p, ptr_2=%p, size=%u, out=%p)",
        type_name[(uint32_t) type], ptr_1, ptr_2, size, out);

    uint32_t start = m_recording.dependencies.size();
    add_out_param(out, type);
    add_in_param(ptr_1, type);
    add_in_param(ptr_2, type);
    uint32_t end = m_recording.dependencies.size();

    Operation op;
    op.type             = OpType::ReduceDot;
    op.dependency_range = std::pair(start, end);
    op.size             = size;
    m_recording.operations.push_back(op);
}

int Recording::replay_reduce_dot(Operation &op) {
    ProfilerPhase profiler(pr_reduce_dot);

    uint32_t dependency_index = op.dependency_range.first;
    AccessInfo out_info        = dependencies[dependency_index];
    AccessInfo ptr_1_info      = dependencies[dependency_index + 1];
    AccessInfo ptr_2_info      = dependencies[dependency_index + 2];

    ReplayVariable &out_var   = replay_variables[out_info.slot];
    ReplayVariable &ptr_1_var = replay_variables[ptr_1_info.slot];
    ReplayVariable &ptr_2_var = replay_variables[ptr_2_info.slot];

    uint32_t size1 = ptr_1_var.size(ptr_1_info.vtype);
    uint32_t size2 = ptr_2_var.size(ptr_2_info.vtype);

    // Technically not required, but serves as a soft assert.
    if (dry_run && size1 != size2)
        return false;

    out_var.alloc(backend, 1, out_info.vtype);

    jitc_log(
        LogLevel::Debug,
        "replay(): reduce_dot(type=%s, ptr_1=%p, ptr_2=%p, size=%u, out=%p)",
        type_name[(uint32_t) out_info.vtype], ptr_1_var.data, ptr_2_var.data,
        size1, out_var.data);

    if (!dry_run)
        ts->reduce_dot(out_info.vtype, ptr_1_var.data, ptr_2_var.data, size1,
                       out_var.data);

    return true;
}

void RecordThreadState::aggregate(void *dst, AggregationEntry *agg,
                                  uint32_t size) {
    if (!paused()) {
        try {
            record_aggregate(dst, agg, size);
        } catch (...) {
            record_exception();
        }
    }
    pause_scope pause(this);
    m_internal->aggregate(dst, agg, size);
}

void RecordThreadState::record_aggregate(void *dst, AggregationEntry *agg,
                                         uint32_t size) {
    jitc_log(LogLevel::Debug, "record(): aggregate(dst=%p, size=%u)", dst,
             size);

    uint32_t start = m_recording.dependencies.size();

    add_out_param(dst, VarType::UInt8);

    for (uint32_t i = 0; i < size; ++i) {
        AggregationEntry &p = agg[i];

        // There are three cases, we might have to handle.
        // 1. The input is a pointer (size = 8 and it is known to the malloc
        // cache)
        // 2. The input is an evaluated variable (size < 0)
        // 3. The variable is a literal (size > 0 and it is not a
        // pointer to a known allocation).

        bool is_ptr;
        auto it = state.alloc_used.find((uintptr_t) p.src);
        if (it == state.alloc_used.end())
            is_ptr = false;
        else
            is_ptr = true;

        if ((p.size == 8 && is_ptr) || p.size < 0) {
            // Pointer or evaluated

            bool has_var = has_variable(p.src);

            // NOTE: Offset buffers of nested calls might be used by this
            // aggregate call, before the offset buffer is uploaded. We defer
            // the offset buffer capture to the memcpy_async operation.
            uint32_t slot = add_variable(p.src);

            AccessInfo info;
            info.slot           = slot;
            info.type           = ParamType::Input;
            info.pointer_access = p.size == 8;
            info.extra.offset   = p.offset;
            info.test_uninit    = false;
            add_param(info);

            jitc_log(LogLevel::Debug,
                     "    entry: var at slot s%u %s src=%p, size=%i, offset=%u",
                     slot, has_var ? "" : "deferred capture", p.src, p.size,
                     p.offset);
        } else {
            // Literal
            AccessInfo info;
            std::memcpy(&info.extra.data, &p.src, sizeof(uint64_t));
            info.extra.offset    = p.offset;
            info.extra.type_size = p.size;
            info.type            = ParamType::Register;
            info.pointer_access  = false;
            add_param(info);

            jitc_log(LogLevel::Debug, "    entry: literal, size=%i, offset=%u",
                     p.size, p.offset);
        }
    }

    uint32_t end = m_recording.dependencies.size();

    Operation op;
    op.type             = OpType::Aggregate;
    op.dependency_range = std::pair(start, end);
    op.size             = size;
    m_recording.operations.push_back(op);
}

int Recording::replay_aggregate(Operation &op) {
    ProfilerPhase profiler(pr_aggregate);

    jitc_log(LogLevel::Debug, "replay(): aggregate");

    uint32_t i = op.dependency_range.first;

    AccessInfo dst_info     = dependencies[i++];
    ReplayVariable &dst_rv = replay_variables[dst_info.slot];

    AggregationEntry *agg = nullptr;

    size_t agg_size = sizeof(AggregationEntry) * op.size;

    // In dry-run mode, we allocate a buffer on the CPU to reuse the code for
    // inferring the size of generated buffers. The pointer is freed at the end
    // of this function.
    if (dry_run)
        agg = (AggregationEntry *) malloc_check(agg_size);
    else if (backend == JitBackend::CUDA)
        agg = (AggregationEntry *) jitc_malloc(AllocType::HostPinned, agg_size);
    else
        agg = (AggregationEntry *) malloc_check(agg_size);

    AggregationEntry *p = agg;

    for (; i < op.dependency_range.second; ++i) {
        AccessInfo param = dependencies[i];

        if (param.type == ParamType::Input) {
            ReplayVariable &rv = replay_variables[param.slot];
            jitc_log(LogLevel::Debug, " -> s%u is_pointer=%u offset=%u",
                     param.slot, param.pointer_access, param.extra.offset);

            if (rv.init == RecordedVarInit::Captured) {
                jitc_log(LogLevel::Debug, "    captured");
                jitc_log(LogLevel::Debug, "    label=%s",
                         jitc_var_label(rv.index));
                jitc_log(LogLevel::Debug, "    data=%s",
                         jitc_var_str(rv.index));
            }

            p->size   = param.pointer_access
                            ? 8
                            : -(int) type_size[(uint32_t) param.vtype];
            p->offset = param.extra.offset;
            p->src    = rv.data;
        } else {
            jitc_log(LogLevel::Debug, " -> literal: offset=%u, size=%u",
                     param.extra.offset, param.extra.type_size);
            p->size   = param.extra.type_size;
            p->offset = param.extra.offset;
            p->src    = (void *) param.extra.data;
        }

        p++;
    }

    AggregationEntry *last = p - 1;
    uint32_t data_size =
        last->offset + (last->size > 0 ? last->size : -last->size);
    // Restore to full alignment
    data_size = (data_size + 7) / 8 * 8;

    dst_rv.alloc(backend, data_size, VarType::UInt8);

    jitc_assert(dst_rv.data != nullptr || dry_run,
                "replay(): Error allocating dst.");

    jitc_log(LogLevel::Debug, "    aggregate(dst=%p, agg=%p, size=%u)",
             dst_rv.data, agg, (uint32_t) (p - agg));

    if (!dry_run)
        ts->aggregate(dst_rv.data, agg, (uint32_t) (p - agg));

    if (dry_run)
        free(agg);

    return true;
}

/// Asynchronously update a single element in memory
void RecordThreadState::poke(void *dst, const void *src, uint32_t size) {
    // At the time of writing, \c poke is only called from \c jit_var_write.
    // \c src will therefore not be a pointer, allocated with \c jitc_malloc,
    // and we cannot track it.
    jitc_raise("RecordThreadState::poke(): this function cannot be recorded!");
    pause_scope pause(this);
    return m_internal->poke(dst, src, size);
}

// Enqueue a function to be run on the host once backend computation is done
void RecordThreadState::enqueue_host_func(void (*callback)(void *),
                                          void *payload) {
    jitc_raise("RecordThreadState::enqueue_host_func(): this function cannot "
               "be recorded!");
    pause_scope pause(this);
    return m_internal->enqueue_host_func(callback, payload);
}

void Recording::validate() {
    for (uint32_t i = 0; i < recorded_variables.size(); i++) {
        RecordedVariable &rv = recorded_variables[i];
        if (rv.state == RecordedVarState::Uninitialized) {
#ifndef NDEBUG
            jitc_raise("record(): Variable at slot s%u %p was left in an "
                       "uninitialized state!",
                       i, rv.ptr);
#else
            jitc_raise("record(): Variable at slot s%u was left in an "
                       "uninitialized state!",
                       i);
#endif
        }
    }
}

bool Recording::check_kernel_cache() {
    for (uint32_t i = 0; i < operations.size(); i++) {
        Operation &op = operations[i];
        if (op.type == OpType::KernelLaunch) {
            // Test if this kernel is still in the cache
            auto it = state.kernel_cache.find(*op.kernel.key);
            if (it == state.kernel_cache.end())
                return false;
        }
    }
    return true;
}

void jitc_freeze_start(JitBackend backend, const uint32_t *inputs,
                       uint32_t n_inputs) {

    if (jitc_flags() & (uint32_t) JitFlag::FreezingScope)
        jitc_fail("Tried to record a thread_state while inside another "
                  "FreezingScope!");

    // Increment scope, can be used to track missing inputs
    jitc_new_scope(backend);

    ThreadState *ts              = thread_state(backend);
    RecordThreadState *record_ts = new RecordThreadState(ts);

    if (backend == JitBackend::CUDA) {
        thread_state_cuda = record_ts;
    } else {
        thread_state_llvm = record_ts;
    }

    for (uint32_t i = 0; i < n_inputs; ++i) {
        record_ts->add_input(inputs[i]);
    }

    jitc_set_flag(JitFlag::FreezingScope, true);
}
Recording *jitc_freeze_stop(JitBackend backend, const uint32_t *outputs,
                            uint32_t n_outputs) {
    if (RecordThreadState *rts =
            dynamic_cast<RecordThreadState *>(thread_state(backend));
        rts != nullptr) {
        ThreadState *internal = rts->m_internal;

        // Perform reassignments to internal thread-state of possibly changed
        // variables
        internal->scope = rts->scope;

        jitc_assert(rts->record_stack.empty(),
                    "Kernel recording ended while still recording loop!");

        jitc_set_flag(JitFlag::FreezingScope, false);
        if (rts->m_exception) {
            std::rethrow_exception(rts->m_exception);
        }

        for (uint32_t i = 0; i < n_outputs; ++i) {
            rts->add_output(outputs[i]);
        }

        if (backend == JitBackend::CUDA) {
            thread_state_cuda = internal;
        } else {
            thread_state_llvm = internal;
        }
        Recording *recording = new Recording(std::move(rts->m_recording));
        recording->validate();
        delete rts;

        return recording;
    } else {
        jitc_fail(
            "jit_record_stop(): Tried to stop recording a thread state "
            "for backend %u, while no recording was started for this backend. "
            "Try to start the recording with jit_record_start.",
            (uint32_t) backend);
    }
}

/**
 * \brief
 *     This captures the offset buffer of a vcall in a kernel.
 *
 * The offset buffer describes where in the data buffer of that vcall the
 * variables or pointers to variables, for that vcall are stored.
 * It should not change between invocations and we should therefore be able
 * to capture it and reuse it when replaying the kernel.
 *
 * \param dsize
 *     The size of the offset buffer in bytes.
 */
uint32_t RecordThreadState::capture_call_offset(const void *ptr, size_t dsize) {
    jitc_log(LogLevel::Debug, "capture_call_offset(ptr=%p, dsize=%zu)", ptr, dsize);
    uint32_t size = dsize / type_size[(uint32_t) VarType::UInt64];

    AllocType atype =
        backend == JitBackend::CUDA ? AllocType::Device : AllocType::HostAsync;
    uint64_t *data = (uint64_t *) jitc_malloc(atype, dsize);
    jitc_memcpy(backend, data, ptr, dsize);

    uint32_t data_var =
        jitc_var_mem_map(backend, VarType::UInt64, data, size, true);

    RecordedVariable rv;
#ifndef NDEBUG
    rv.ptr = ptr;
#endif
    rv.state = RecordedVarState::Captured;
    rv.init  = RecordedVarInit::Captured;
    rv.index = data_var;

    uint32_t slot;
    auto it = ptr_to_slot.find(ptr);
    if (it == ptr_to_slot.end()) {
        slot = m_recording.recorded_variables.size();

        m_recording.recorded_variables.push_back(rv);

        ptr_to_slot.insert({ ptr, slot });
    } else {
        slot                  = it.value();
        RecordedVariable &old = m_recording.recorded_variables[slot];
        if (old.init != RecordedVarInit::None)
            jitc_fail("record(): Tried to overwrite an initialized variable "
                      "with an offset buffer!");

        m_recording.recorded_variables[slot] = rv;
    }

    return slot;
}

/**
 * This function tries to capture a variable that is not known to the
 * recording \c ThreadState.
 * This is unsupported for now and raises an exception.
 */
uint32_t RecordThreadState::capture_variable(uint32_t index,
                                             const void * /*ptr*/,
                                             bool /*remember*/, bool test_scope,
                                             bool /*overwrite*/) {

    pause_scope pause(this);
    Variable *v = jitc_var(index);
    if (v->scope < m_internal->scope && test_scope) {
        jitc_raise("record(): Variable r%u[%u] -> %p, label=%s, was created "
                   "before recording was started, but it was "
                   "not specified as an input variable! This can happen if a "
                   "input type is not fully traversable, for example when not "
                   "specifying a member in DRJIT_STRUCT, but using it in the "
                   "frozen function.",
                   index, v->size, v->data, jitc_var_label(index));
    }

    jitc_raise("record(): Variable r%u[%u] -> %p, label=%s, was created while "
               "recording, but it was not created by a supported operation. "
               "This can happen if a memory region was created outside of "
               "Dr.Jit but mapped to a Dr.Jit variable.",
               index, v->size, v->data, jitc_var_label(index));

    return 0;
}

/**
 * Add information about a variable, deduplicating it and returning the slot
 * in the `variables` field of the recording.
 * Information is combined when the variable has already been added.
 * This is used by the input variables of a kernel.
 */
uint32_t RecordThreadState::add_variable(const void *ptr) {
    auto it = ptr_to_slot.find(ptr);

    if (it == ptr_to_slot.end()) {
        uint32_t slot = m_recording.recorded_variables.size();

        RecordedVariable rv;
#ifndef NDEBUG
        rv.ptr = ptr;
#endif
        m_recording.recorded_variables.push_back(rv);

        ptr_to_slot.insert({ ptr, slot });

        return slot;
    } else {
        uint32_t slot = it.value();

        return slot;
    }
}

/// Return the slot index given the data pointer of a variable.
/// This fails if the variable has not been previously added.
uint32_t RecordThreadState::get_variable(const void *ptr) {
    auto it = ptr_to_slot.find(ptr);

    if (it == ptr_to_slot.end())
        jitc_raise("Failed to find the slot corresponding to the variable "
                   "with data at %p",
                   ptr);

    return it.value();
}

/// Test if the ThreadState knows this \c ptr
bool RecordThreadState::has_variable(const void *ptr) {
    auto it = ptr_to_slot.find(ptr);

    return it != ptr_to_slot.end();
}

/**
 * Adds a parameter access to the \ref dependencies vector.
 * This also modifies the state of the \ref RecordVariable that was
 * accessed.
 */
void RecordThreadState::add_param(AccessInfo info) {
    RecordedVariable &rv = m_recording.recorded_variables[info.slot];
    if (info.type == ParamType::Output) {

        jitc_log(LogLevel::Debug, " <- param s%u", info.slot);

        if (info.vtype != VarType::Void)
            rv.type = info.vtype;

        rv.state = RecordedVarState::OpOutput;

    } else if (info.type == ParamType::Input) {

        jitc_log(LogLevel::Debug, " -> param s%u", info.slot);

        if (info.test_uninit && rv.state == RecordedVarState::Uninitialized)
            jitc_raise("record(): Variable at slot s%u was read by "
                       "operation o%u, but it had not yet been initialized! "
                       "This can occur if the variable was not part of "
                       "the input but is used by a recorded operation, for "
                       "example if it was not specified as a member in a "
                       "DRJIT_STRUCT but used in the frozen function.",
                       info.slot, (uint32_t) m_recording.operations.size());

        if (info.vtype == VarType::Void)
            info.vtype = rv.type;
    }

    m_recording.dependencies.push_back(info);
}
/// Helper function for recording input parameters given the slot.
void RecordThreadState::add_in_param(uint32_t slot, VarType vtype,
                                     bool test_uninit) {
    AccessInfo info;
    info.type        = ParamType::Input;
    info.slot        = slot;
    info.test_uninit = test_uninit;
    info.vtype       = vtype;
    add_param(info);
}
/// Helper function recording input access given the pointer.
void RecordThreadState::add_in_param(const void *ptr, VarType vtype,
                                     bool test_uninit) {
    uint32_t slot = get_variable(ptr);
    add_in_param(slot, vtype, test_uninit);
}
/// Helper function recording an output access, given the slot and \ref VarType
void RecordThreadState::add_out_param(uint32_t slot, VarType vtype) {
    AccessInfo info;
    info.type  = ParamType::Output;
    info.slot  = slot;
    info.vtype = vtype;
    add_param(info);
}
/// Helper function recording an output access, given the pointer and \ref
/// VarType
void RecordThreadState::add_out_param(const void *ptr, VarType vtype) {
    uint32_t slot = add_variable(ptr);
    add_out_param(slot, vtype);
}
/// Helper function recording an output access, given the pointer and the
/// uint32_t representation of a \ref VarType
void RecordThreadState::add_out_param(uint32_t slot, uint32_t vtype) {
    add_out_param(slot, (VarType) vtype);
}

void jitc_freeze_abort(JitBackend backend) {
    if (RecordThreadState *rts =
            dynamic_cast<RecordThreadState *>(thread_state(backend));
        rts != nullptr) {

        ThreadState *internal = rts->m_internal;

        // Perform reassignments to internal thread-state of possibly changed
        // variables
        internal->scope = rts->scope;

        if (backend == JitBackend::CUDA) {
            thread_state_cuda = internal;
        } else {
            thread_state_llvm = internal;
        }

        delete rts;

        jitc_set_flag(JitFlag::FreezingScope, false);
    }
}

void jitc_freeze_destroy(Recording *recording) {
    for (RecordedVariable &rv : recording->recorded_variables) {
        if (rv.init == RecordedVarInit::Captured) {
            jitc_var_dec_ref(rv.index);
        }
    }
#if defined(DRJIT_ENABLE_OPTIX)
    for (Operation &op : recording->operations) {
        if (op.uses_optix){
            jitc_free(op.sbt->hitgroupRecordBase);
            jitc_free(op.sbt->missRecordBase);
            delete op.sbt;
        }
    }
#endif
    delete recording;
}

int jitc_freeze_pause(JitBackend backend) {

    if (RecordThreadState *rts =
            dynamic_cast<RecordThreadState *>(thread_state(backend));
        rts != nullptr) {
        jitc_set_flag(JitFlag::FreezingScope, false);
        return rts->pause();
    } else {
        jitc_fail(
            "jit_freeze_pause(): Tried to pause recording a thread state "
            "for backend %u, while no recording was started for this backend. "
            "Try to start the recording with jit_freeze_start.",
            (uint32_t) backend);
    }
}
int jitc_freeze_resume(JitBackend backend) {
    if (RecordThreadState *rts =
            dynamic_cast<RecordThreadState *>(thread_state(backend));
        rts != nullptr) {
        jitc_set_flag(JitFlag::FreezingScope, true);
        return rts->resume();
    } else {
        jitc_fail(
            "jit_freeze_resume(): Tried to resume recording a thread state "
            "for backend %u, while no recording was started for this backend. "
            "Try to start the recording with jit_freeze_start.",
            (uint32_t) backend);
    }
}

void jitc_freeze_replay(Recording *recording, const uint32_t *inputs,
                        uint32_t *outputs) {
    dry_run = false;
    recording->replay(inputs, outputs);
}

int jitc_freeze_dry_run(Recording *recording, const uint32_t *inputs) {
    int result = true;
    // Check if all kernels are still present in the kernel cache
    if (!recording->check_kernel_cache())
        return false;
    if (recording->requires_dry_run) {
        jitc_log(LogLevel::Debug, "Replaying in dry-run mode");
        dry_run = true;
        result  = recording->replay(inputs, nullptr);
        dry_run = false;
    }

    return result;
}
