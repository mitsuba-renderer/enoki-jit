/*
    src/vcall.cpp -- Code generation for virtual function calls

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "log.h"
#include "var.h"
#include "eval.h"
#include "registry.h"
#include "util.h"
#include "op.h"
#include "profiler.h"
#include "vcall.h"


static std::vector<VCall *> vcalls_assembled;

static void jitc_var_vcall_collect_data(
    tsl::robin_map<uint64_t, uint32_t, UInt64Hasher> &data_map,
    uint32_t &data_offset, uint32_t inst_id, uint32_t index,
    bool &use_self, bool &use_optix);

void jitc_vcall_set_self(JitBackend backend, uint32_t value, uint32_t index) {
    ThreadState *ts = thread_state(backend);

    if (ts->vcall_self_index) {
        jitc_var_dec_ref(ts->vcall_self_index);
        ts->vcall_self_index = 0;
    }

    ts->vcall_self_value = value;

    if (value) {
        if (index) {
            jitc_var_inc_ref(index);
            ts->vcall_self_index = index;
        } else {
            Variable v;
            v.kind = VarKind::VCallSelf;
            v.backend = (uint32_t) backend;
            v.size = 1u;
            v.type = (uint32_t) VarType::UInt32;
            v.placeholder = true;
            ts->vcall_self_index = jitc_var_new(v, true);
        }
    }
}

void jitc_vcall_self(JitBackend backend, uint32_t *value, uint32_t *index) {
    ThreadState *ts = thread_state(backend);
    *value = ts->vcall_self_value;
    *index = ts->vcall_self_index;
}

/// Weave a virtual function call into the computation graph
uint32_t jitc_var_vcall(const char *name, uint32_t self, uint32_t mask_,
                        uint32_t n_inst, const uint32_t *inst_id, uint32_t n_in,
                        const uint32_t *in, uint32_t n_out_nested,
                        const uint32_t *out_nested, const uint32_t *checkpoints,
                        uint32_t *out) {

    const uint32_t checkpoint_mask = 0x7fffffff;

#if defined(DRJIT_ENABLE_NVTX) || defined(DRJIT_ENABLE_ITTNOTIFY)
    std::string profile_name = std::string("jit_var_vcall: ") + name;
    ProfilerRegion profiler_region(profile_name.c_str());
    ProfilerPhase profiler(profiler_region);
#endif

    // =====================================================
    // 1. Various sanity checks
    // =====================================================

    if (n_inst == 0)
        jitc_raise("jit_var_vcall(): must have at least one instance!");

    if (n_out_nested % n_inst != 0)
        jitc_raise("jit_var_vcall(): list of all output indices must be a "
                   "multiple of the instance count!");

    uint32_t n_out = n_out_nested / n_inst, size = 0,
             in_size_initial = 0, out_size_initial = 0;

    bool placeholder = false, dirty = false;

    JitBackend backend;
    /* Check 'self' */ {
        const Variable *self_v = jitc_var(self);
        size = self_v->size;
        placeholder |= (bool) self_v->placeholder;
        dirty |= self_v->is_dirty();
        backend = (JitBackend) self_v->backend;
        if ((VarType) self_v->type != VarType::UInt32)
            jitc_raise("jit_var_vcall(): 'self' argument must be of type "
                       "UInt32 (was: %s)", type_name[self_v->type]);
    }

    size = std::max(size, jitc_var(mask_)->size);

    for (uint32_t i = 0; i < n_in; ++i) {
        const Variable *v = jitc_var(in[i]);
        if (v->vcall_iface) {
            if (!v->dep[0])
                jitc_raise("jit_var_vcall(): placeholder variable r%u does not "
                           "reference another input!", in[i]);
            Variable *v2 = jitc_var(v->dep[0]);
            placeholder |= (bool) v2->placeholder;
            dirty |= v2->is_dirty();
            size = std::max(size, v2->size);
        } else if (!v->is_literal()) {
            jitc_raise("jit_var_vcall(): input variable r%u must either be a "
                       "value or placeholder wrapping another variable!", in[i]);
        }
        if (v->size != 1)
            jitc_raise("jit_var_vcall(): size of input variable r%u must be 1!", in[i]);
        in_size_initial += type_size[v->type];
    }

    for (uint32_t i = 0; i < n_out_nested; ++i) {
        const Variable *v  = jitc_var(out_nested[i]),
                       *v0 = jitc_var(out_nested[i % n_out]);
        size = std::max(size, v->size);
        if (v->type != v0->type)
            jitc_raise(
                "jit_var_vcall(): output types don't match between instances!");

        if (i < n_out)
            out_size_initial += type_size[v->type];
    }

    for (uint32_t i = 0; i < n_in; ++i) {
        const Variable *v = jitc_var(in[i]);
        if (v->size != 1 && v->size != size)
            jitc_raise(
                "jit_var_vcall(): input/output arrays have incompatible size!");
    }

    for (uint32_t i = 0; i < n_out_nested; ++i) {
        const Variable *v = jitc_var(out_nested[i]);
        if (v->size != 1 && v->size != size)
            jitc_raise(
                "jit_var_vcall(): input/output arrays have incompatible size!");
    }

    ThreadState *ts = thread_state(backend);
    if (ts->side_effects_recorded.size() !=
        (checkpoints[n_inst] & checkpoint_mask))
        jitc_raise("jitc_var_vcall(): side effect queue doesn't have the "
                   "expected size!");

    if (dirty) {
        jitc_eval(ts);

        dirty = jitc_var(self)->is_dirty();
        for (uint32_t i = 0; i < n_in; ++i) {
            const Variable *v = jitc_var(in[i]);
            if (v->vcall_iface)
                dirty |= jitc_var(v->dep[0])->is_dirty();
        }

        if (unlikely(dirty))
            jitc_raise("jit_var_vcall(): inputs remain dirty after evaluation!");
    }

    // =====================================================
    // 3. Apply any masks on the stack, ignore NULL args
    // =====================================================

    Ref mask;
    {
        uint32_t zero = 0;
        Ref null_instance = steal(jitc_var_literal(backend, VarType::UInt32, &zero, 1, 0)),
            is_non_null   = steal(jitc_var_neq(self, null_instance)),
            mask_2        = steal(jitc_var_and(mask_, is_non_null));
        mask = steal(jitc_var_mask_apply(mask_2, size));
    }

    // =====================================================
    // 3. Stash information about inputs and outputs
    // =====================================================

    std::unique_ptr<VCall> vcall(new VCall());
    vcall->backend = backend;
    vcall->name = strdup(name);
    vcall->n_inst = n_inst;
    vcall->inst_id = std::vector<uint32_t>(inst_id, inst_id + n_inst);
    vcall->inst_hash.resize(n_inst);
    vcall->in.reserve(n_in);
    vcall->in_nested.reserve(n_in);
    vcall->out.reserve(n_out);
    vcall->out_nested.reserve(n_out_nested);
    vcall->in_size_initial = in_size_initial;
    vcall->out_size_initial = out_size_initial;
    vcall->in_count_initial = n_in;
    vcall->checkpoints.reserve(n_inst + 1);
    vcall->side_effects = std::vector<uint32_t>(
        ts->side_effects_recorded.begin() + (checkpoints[0] & checkpoint_mask),
        ts->side_effects_recorded.begin() + (checkpoints[n_inst] & checkpoint_mask));
    ts->side_effects_recorded.resize(checkpoints[0] & checkpoint_mask);

    // =====================================================
    // 4. Collect evaluated data accessed by the instances
    // =====================================================

    vcall->data_offset.reserve(n_inst);

    // Collect accesses to evaluated variables/pointers
    uint32_t data_size = 0, inst_id_max = 0;
    bool use_optix = false;
    for (uint32_t i = 0; i < n_inst; ++i) {
        if (unlikely(checkpoints[i] > checkpoints[i + 1]))
            jitc_raise("jitc_var_vcall(): checkpoints parameter is not "
                       "monotonically increasing!");

        uint32_t id = inst_id[i];
        vcall->data_offset.push_back(data_size);

        for (uint32_t j = 0; j < n_out; ++j)
            jitc_var_vcall_collect_data(vcall->data_map, data_size, i,
                                        out_nested[j + i * n_out],
                                        vcall->use_self, use_optix);

        for (uint32_t j = checkpoints[i]; j != checkpoints[i + 1]; ++j)
            jitc_var_vcall_collect_data(vcall->data_map, data_size, i,
                                        vcall->side_effects[j - checkpoints[0]],
                                        vcall->use_self, use_optix);

        // Restore to full alignment
        data_size = (data_size + 7) / 8 * 8;
        inst_id_max = std::max(id, inst_id_max);
    }

    // Allocate memory + wrapper variables for call offset and data arrays
    vcall->offset_size = (inst_id_max + 1) * sizeof(uint64_t);

    AllocType at =
        backend == JitBackend::CUDA ? AllocType::Device : AllocType::HostAsync;
    vcall->offset = (uint64_t *) jitc_malloc(at, vcall->offset_size);
    uint8_t *data_d = (uint8_t *) jitc_malloc(at, data_size);

    Ref data_buf, data_v,
        offset_buf = steal(jitc_var_mem_map(
            backend, VarType::UInt64, vcall->offset, inst_id_max + 1, 1)),
        offset_v =
            steal(jitc_var_pointer(backend, vcall->offset, offset_buf, 0));

    char temp[128];
    snprintf(temp, sizeof(temp), "VCall: %s [call offsets]", name);
    jitc_var_set_label(offset_buf, temp);

    if (data_size) {
        data_buf = steal(
            jitc_var_mem_map(backend, VarType::UInt8, data_d, data_size, 1));
        snprintf(temp, sizeof(temp), "VCall: %s [call data]", name);
        jitc_var_set_label(data_buf, temp);

        data_v = steal(jitc_var_pointer(backend, data_d, data_buf, 0));

        VCallDataRecord *rec = (VCallDataRecord *)
            jitc_malloc(backend == JitBackend::CUDA ? AllocType::HostPinned
                                                    : AllocType::Host,
                        sizeof(VCallDataRecord) * vcall->data_map.size());

        VCallDataRecord *p = rec;

        for (auto kv : vcall->data_map) {
            uint32_t index = (uint32_t) kv.first, offset = kv.second;
            if (offset == (uint32_t) -1)
                continue;

            const Variable *v = jitc_var(index);
            bool is_pointer = (VarType) v->type == VarType::Pointer;
            p->offset = offset;
            p->size = is_pointer ? 0u : type_size[v->type];
            p->src = is_pointer ? (const void *) v->literal : v->data;
            p++;
        }

        std::sort(rec, p,
                  [](const VCallDataRecord &a, const VCallDataRecord &b) {
                      return a.offset < b.offset;
                  });

        jitc_vcall_prepare(backend, data_d, rec, (uint32_t)(p - rec));
    } else {
        vcall->data_map.clear();
    }

    // =====================================================
    // 5. Create special variable encoding the function call
    // =====================================================

    Ref vcall_v;

    if (data_size)
        vcall_v = steal(jitc_var_new_node_4(
            backend, VarKind::Dispatch, VarType::Void, size, placeholder, self,
            jitc_var(self), mask, jitc_var(mask), offset_v, jitc_var(offset_v),
            data_v, jitc_var(data_v)));
    else
        vcall_v = steal(
            jitc_var_new_node_3(backend, VarKind::Dispatch, VarType::Void, size,
                                placeholder, self, jitc_var(self), mask,
                                jitc_var(mask), offset_v, jitc_var(offset_v)));

    vcall->id = vcall_v;

    uint32_t n_devirt = 0, flags = jitc_flags();

    bool vcall_optimize = flags & (uint32_t) JitFlag::VCallOptimize,
         vcall_inline   = flags & (uint32_t) JitFlag::VCallInline;

    std::vector<bool> uniform(n_out, true);
    for (uint32_t i = 0; i < n_inst; ++i) {
        for (uint32_t j = 0; j < n_out; ++j) {
            uint32_t index = out_nested[i * n_out + j];
            uniform[j] = uniform[j] && index == out_nested[j];

            /* Hold a reference to the nested computation until the cleanup
               callback later below is invoked. */
            jitc_var_inc_ref(index);
            vcall->out_nested.push_back(index);
        }

        vcall->checkpoints.push_back(checkpoints[i] - checkpoints[0]);
    }
    vcall->checkpoints.push_back(checkpoints[n_inst] - checkpoints[0]);


    if (vcall_optimize) {
        for (uint32_t j = 0; j < n_out; ++j) {
            if (!uniform[j])
                continue;

            /* Only devirtualize literals unless inlining is requested */
            if (!jitc_var(out_nested[j])->is_literal() &&
                !((n_inst == 1) && vcall_inline))
                continue;

            Ref result_v = steal(jitc_var_and(out_nested[j], mask));
            Variable *v = jitc_var(result_v);

            if ((bool) v->placeholder != placeholder || v->size != size) {
                if (v->ref_count != 1) {
                    result_v = steal(jitc_var_copy(result_v));
                    v = jitc_var(result_v);
                }
                jitc_lvn_drop(result_v, v);
                v->placeholder = placeholder;
                v->size = size;
                jitc_lvn_put(result_v, v);
            }

            out[j] = result_v.release();
            n_devirt++;

            for (uint32_t i = 0; i < n_inst; ++i) {
                uint32_t &index_2 = vcall->out_nested[i * n_out + j];
                jitc_var_dec_ref(index_2);
                index_2 = 0;
            }
        }
    }

    uint32_t se_count = checkpoints[n_inst] - checkpoints[0];

    jitc_log(InfoSym,
             "jit_var_vcall(r%u, self=r%u): call (\"%s\") with %u instance%s, %u "
             "input%s, %u output%s (%u devirtualized), %u side effect%s, %u "
             "byte%s of call data, %u elements%s%s", (uint32_t) vcall_v, self, name, n_inst,
             n_inst == 1 ? "" : "s", n_in, n_in == 1 ? "" : "s", n_out,
             n_out == 1 ? "" : "s", n_devirt, se_count, se_count == 1 ? "" : "s",
             data_size, data_size == 1 ? "" : "s", size,
             (n_devirt == n_out && se_count == 0) ? " (optimized away)" : "",
             placeholder ? " (part of a recorded computation)" : "");

    // =====================================================
    // 6. Create output variables
    // =====================================================

    auto var_callback = [](uint32_t index, int free, void *ptr) {
        if (!ptr)
            return;

        VCall *vcall_2 = (VCall *) ptr;
        if (!free) {
            // Disable callback
            state.extra[index].callback_data = nullptr;
        }

        // An output variable is no longer referenced. Find out which one.
        uint32_t n_out_2 = (uint32_t) vcall_2->out.size(),
                 offset  = (uint32_t) -1;

        for (uint32_t i = 0; i < n_out_2; ++i) {
            if (vcall_2->out[i] == index)
                offset = i;
        }

        if (offset == (uint32_t) -1)
            jitc_fail("jit_var_vcall(): expired output variable %u could not "
                      "be located!", index);

        // Inform recursive computation graphs via reference counting
        for (uint32_t j = 0; j < vcall_2->n_inst; ++j) {
            uint32_t &index_2 = vcall_2->out_nested[n_out_2 * j + offset];
            jitc_var_dec_ref(index_2);
            index_2 = 0;
        }
        vcall_2->out[offset] = 0;

        // Check if any input parameters became irrelevant
        for (uint32_t i = 0; i < vcall_2->in.size(); ++i) {
            uint32_t index_2 = vcall_2->in_nested[i];
            if (index_2 &&
                state.variables.find(index_2) == state.variables.end()) {
                Extra *e = &state.extra[vcall_2->id];
                if (unlikely(e->dep[i] != vcall_2->in[i]))
                    jitc_fail("jit_var_vcall(): internal error! (1)");
                jitc_var_dec_ref(vcall_2->in[i]);
                if (state.extra.find(vcall_2->id) == state.extra.end())
                    jitc_fail("jit_var_vcall(): internal error! (2)");
                e = &state.extra[vcall_2->id]; // may have changed
                e->dep[i] = 0;
                vcall_2->in[i] = 0;
                vcall_2->in_nested[i] = 0;
            }
        }
    };

    for (uint32_t i = 0; i < n_out; ++i) {
        uint32_t index = vcall->out_nested[i];
        if (!index) {
            vcall->out.push_back(0);
            continue;
        }

        const Variable *v = jitc_var(index);
        uint32_t index_2 = jitc_var_new_node_1(
            backend, VarKind::Nop, (VarType) v->type, size, placeholder,
            vcall_v, jitc_var(vcall_v), i);

        jitc_var(index_2)->extra = 1;
        Extra &extra = state.extra[index_2];

        if (vcall_optimize) {
            extra.callback = var_callback;
            extra.callback_data = vcall.get();
            extra.callback_internal = true;
        }

        vcall->out.push_back(index_2);
        snprintf(temp, sizeof(temp), "VCall: %s [out %u]", name, i);
        jitc_var_set_label(index_2, temp);
        out[i] = index_2;
    }

    // =====================================================
    // 7. Optimize calling conventions by reordering args
    // =====================================================

    for (uint32_t i = 0; i < n_in; ++i) {
        uint32_t index = in[i];
        Variable *v = jitc_var(index);
        if (!v->vcall_iface)
            continue;

        // Ignore unreferenced inputs
        if (vcall_optimize && v->ref_count == 2 /* 1 each from collect_indices + wrap_vcall */) {
            auto& on = vcall->out_nested;
            auto it  = std::find(on.begin(), on.end(), index);
            // Only skip if this variable isn't also an output
            if (it == on.end())
                continue;
        }

        vcall->in_nested.push_back(index);

        uint32_t index_2 = v->dep[0];
        vcall->in.push_back(index_2);
        jitc_var_inc_ref(index_2);
    }

    auto comp = [](uint32_t i0, uint32_t i1) {
        int s0 = i0 ? type_size[jitc_var(i0)->type] : 0;
        int s1 = i1 ? type_size[jitc_var(i1)->type] : 0;
        return s0 > s1;
    };

    std::sort(vcall->in.begin(), vcall->in.end(), comp);
    std::sort(vcall->in_nested.begin(), vcall->in_nested.end(), comp);

    std::sort(vcall->out.begin(), vcall->out.end(), comp);
    for (uint32_t i = 0; i < n_inst; ++i)
        std::sort(vcall->out_nested.begin() + (i + 0) * n_out,
                  vcall->out_nested.begin() + (i + 1) * n_out, comp);

    // =====================================================
    // 8. Install code generation and deallocation callbacks
    // =====================================================

    size_t dep_size = vcall->in.size() * sizeof(uint32_t);

    {
        Variable *vcall_var = jitc_var(vcall_v);
        vcall_var->extra = 1;
        vcall_var->optix = use_optix;
    }

    Extra *e_special = &state.extra[vcall_v];
    e_special->n_dep = (uint32_t) vcall->in.size();
    e_special->dep = (uint32_t *) malloc_check(dep_size);

    // Steal input dependencies from placeholder arguments
    if (dep_size)
        memcpy(e_special->dep, vcall->in.data(), dep_size);

    e_special->callback = [](uint32_t, int free, void *ptr) {
        if (free)
            delete (VCall *) ptr;
    };
    e_special->callback_internal = true;
    e_special->callback_data = vcall.release();

    snprintf(temp, sizeof(temp), "VCall: %s", name);
    jitc_var_set_label(vcall_v, temp);

    Ref se_v;
    if (se_count) {
        /* The call has side effects. Create a dummy variable to
           ensure that they are evaluated */
        se_v = steal(jitc_var_new_node_1(backend, VarKind::Nop, VarType::Void,
                                         size, placeholder, vcall_v,
                                         jitc_var(vcall_v)));

        snprintf(temp, sizeof(temp), "VCall: %s [side effects]", name);
        jitc_var_set_label(se_v, temp);
    }

    vcall_v.reset();

    return se_v.release();
}

static ProfilerRegion profiler_region_vcall_assemble("jit_var_vcall_assemble");


/// Called by the JIT compiler when compiling a virtual function call
void jitc_var_vcall_assemble(VCall *vcall, uint32_t self_reg, uint32_t mask_reg,
                             uint32_t offset_reg, uint32_t data_reg) {
    uint32_t vcall_reg = jitc_var(vcall->id)->reg_index;

    ProfilerPhase profiler(profiler_region_vcall_assemble);

    // =====================================================
    // 1. Need to backup state before we can JIT recursively
    // =====================================================

    struct JitBackupRecord {
        ScheduledVariable sv;
        uint32_t param_type : 2;
        uint32_t output_flag : 1;
        uint32_t reg_index;
        uint32_t param_offset;
    };

    std::vector<JitBackupRecord> backup;
    backup.reserve(schedule.size());

    for (const ScheduledVariable &sv : schedule) {
        const Variable *v = jitc_var(sv.index);
        backup.push_back(JitBackupRecord{ sv, v->param_type, v->output_flag,
                                          v->reg_index, v->param_offset });
    }

    int32_t alloca_size_backup = alloca_size;
    int32_t alloca_align_backup = alloca_align;
    alloca_size = alloca_align = -1;

    // =====================================================
    // 2. Determine calling conventions (input/output size,
    //    and alignment). Less state may need to be passed
    //    compared to when the vcall was first created.
    // =====================================================

    uint32_t in_size = 0, in_align = 1,
             out_size = 0, out_align = 1,
             n_in = (uint32_t) vcall->in.size(),
             n_out = (uint32_t) vcall->out.size(),
             n_in_active = 0,
             n_out_active = 0;

    for (uint32_t i = 0; i < n_in; ++i) {
        auto it = state.variables.find(vcall->in[i]);
        if (it == state.variables.end())
            continue;

        uint32_t size = type_size[it.value().type],
                 offset = in_size;
        Variable *v = &it.value();
        in_size += size;
        in_align = std::max(size, in_align);
        n_in_active++;

        // Transfer parameter offset to instances
        auto it2 = state.variables.find(vcall->in_nested[i]);
        if (it2 == state.variables.end())
            continue;

        Variable *v2 = &it2.value();
        v2->param_offset = offset;
        v2->reg_index = v->reg_index;
    }

    for (uint32_t i = 0; i < n_out; ++i) {
        auto it = state.variables.find(vcall->out_nested[i]);
        if (it == state.variables.end())
            continue;

        Variable *v = &it.value();
        uint32_t size = type_size[v->type];
        out_size += size;
        out_align = std::max(size, out_align);
        n_out_active++;
    }

    // Input and output are packed into the same array in LLVM mode
    if (vcall->backend == JitBackend::LLVM)
        in_size = (in_size + out_align - 1) / out_align * out_align;

    // =====================================================
    // 3. Compile code for all instances and collapse
    // =====================================================

    ThreadState *ts = thread_state(vcall->backend);

    vcall->callables_set = {};
    for (uint32_t i = 0; i < vcall->n_inst; ++i) {
        XXH128_hash_t hash = jitc_assemble_func(
            ts, vcall->name, i, in_size, in_align, out_size, out_align,
            vcall->data_offset[i], vcall->data_map, n_in, vcall->in.data(),
            n_out, vcall->out_nested.data() + n_out * i,
            vcall->checkpoints[i + 1] - vcall->checkpoints[i],
            vcall->side_effects.data() + vcall->checkpoints[i],
            vcall->use_self);
        vcall->inst_hash[i] = hash;
        vcall->callables_set.insert(hash);
    }

    size_t se_count = vcall->side_effects.size();
    vcall->clear_side_effects();

    // =====================================================
    // 4. Restore previously backed-up JIT state
    // =====================================================

    schedule.clear();
    for (const JitBackupRecord &b : backup) {
        Variable *v = jitc_var(b.sv.index);
        v->param_type = b.param_type;
        v->output_flag = b.output_flag;
        v->reg_index = b.reg_index;
        v->param_offset = b.param_offset;
        schedule.push_back(b.sv);
    }

    alloca_size = alloca_size_backup;
    alloca_align = alloca_align_backup;

    if (vcall->backend == JitBackend::CUDA)
        jitc_var_vcall_assemble_cuda(vcall, vcall_reg, self_reg, mask_reg,
                                     offset_reg, data_reg, n_out, in_size,
                                     in_align, out_size, out_align);
    else
        jitc_var_vcall_assemble_llvm(vcall, vcall_reg, self_reg, mask_reg,
                                     offset_reg, data_reg, n_out, in_size,
                                     in_align, out_size, out_align);

    jitc_log(
        InfoSym,
        "jit_var_vcall_assemble(): indirect call (\"%s\") to %zu/%u instances, "
        "passing %u/%u inputs (%u/%u bytes), %u/%u outputs (%u/%u bytes), %zu side effects",
        vcall->name, vcall->callables_set.size(), vcall->n_inst, n_in_active,
        vcall->in_count_initial, in_size, vcall->in_size_initial, n_out_active,
        n_out, out_size, vcall->out_size_initial, se_count);

    jitc_var_inc_ref(vcall->id);
    vcalls_assembled.push_back(vcall);
}

/// Collect scalar / pointer variables referenced by a computation
void jitc_var_vcall_collect_data(tsl::robin_map<uint64_t, uint32_t, UInt64Hasher> &data_map,
                                 uint32_t &data_offset, uint32_t inst_id,
                                 uint32_t index, bool &use_self, bool &use_optix) {
    uint64_t key = (uint64_t) index + (((uint64_t) inst_id) << 32);
    auto it_and_status = data_map.emplace(key, (uint32_t) -1);
    if (!it_and_status.second)
        return;

    const Variable *v = jitc_var(index);

    if ((VarKind) v->kind == VarKind::VCallSelf)
        use_self = true;

#if defined(DRJIT_ENABLE_OPTIX)
    if ((JitBackend) v->backend == JitBackend::CUDA)
        use_optix |= v->optix;
#endif

    if (v->vcall_iface) {
        return;
    } else if (v->is_data() || (VarType) v->type == VarType::Pointer) {
        uint32_t tsize = type_size[v->type];
        uint32_t offset = (data_offset + tsize - 1) / tsize * tsize;
        it_and_status.first.value() = offset;
        data_offset = offset + tsize;

        if (v->size != 1)
            jitc_raise(
                "jit_var_vcall(): the virtual function call associated with "
                "instance %u accesses an evaluated variable r%u of type "
                "%s and size %u. However, only *scalar* (size == 1) "
                "evaluated variables can be accessed while recording "
                "virtual function calls",
                inst_id, index, type_name[v->type], v->size);
    } else {
        for (uint32_t i = 0; i < 4; ++i) {
            uint32_t index_2 = v->dep[i];
            if (!index_2)
                break;

            jitc_var_vcall_collect_data(data_map, data_offset, inst_id, index_2,
                                        use_self, use_optix);
        }
        if (unlikely(v->extra)) {
            auto it = state.extra.find(index);
            if (it == state.extra.end())
                jitc_fail("jit_var_vcall_collect_data(): could not find "
                          "matching 'extra' record!");

            const Extra &extra = it->second;
            for (uint32_t i = 0; i < extra.n_dep; ++i) {
                uint32_t index_2 = extra.dep[i];
                if (index_2 == 0)
                    continue; // not break
                jitc_var_vcall_collect_data(data_map, data_offset, inst_id,
                                            index_2, use_self, use_optix);
            }
        }
    }
}

void jitc_vcall_upload(ThreadState *ts) {
    AllocType at = ts->backend == JitBackend::CUDA ? AllocType::HostPinned
                                                   : AllocType::Host;

    for (VCall *vcall : vcalls_assembled) {
        uint64_t *data = (uint64_t *) jitc_malloc(at, vcall->offset_size);
        memset(data, 0, vcall->offset_size);

        for (uint32_t i = 0; i < vcall->n_inst; ++i) {
            auto it = globals_map.find(GlobalKey(vcall->inst_hash[i], true));
            if (it == globals_map.end())
                jitc_fail("jitc_vcall_upload(): could not find callable!");

            // high part: instance data offset, low part: callable index
            data[vcall->inst_id[i]] =
                (((uint64_t) vcall->data_offset[i]) << 32) |
                it->second.callable_index;
        }

        jitc_memcpy_async(ts->backend, vcall->offset, data, vcall->offset_size);

        // Free call offset table asynchronously
        if (vcall->backend == JitBackend::CUDA) {
            jitc_free(data);
        } else {
            Task *new_task = task_submit_dep(
                nullptr, &jitc_task, 1, 1,
                [](uint32_t, void *payload) { jit_free(*((void **) payload)); },
                &data, sizeof(void *), nullptr, 1);
            task_release(jitc_task);
            jitc_task = new_task;
        }
    }

    for (VCall *vcall : vcalls_assembled)
        jitc_var_dec_ref(vcall->id);
    vcalls_assembled.clear();
}

// Compute a permutation to reorder an array of registered pointers
VCallBucket *jitc_var_vcall_reduce(JitBackend backend, const char *domain,
                                   uint32_t index, uint32_t *bucket_count_out) {
    auto it = state.extra.find(index);
    if (it != state.extra.end()) {
        auto &v = it.value();
        if (v.vcall_bucket_count) {
            *bucket_count_out = v.vcall_bucket_count;
            return v.vcall_buckets;
        }
    }

    uint32_t bucket_count;
    if (domain)
        bucket_count = jitc_registry_get_max(backend, domain) + 1;
    else
        bucket_count = *bucket_count_out + 1;

    if (unlikely(bucket_count == 1)) {
        *bucket_count_out = 0;
        return nullptr;
    }

    // Ensure input index array is fully evaluated
    jitc_var_eval(index);

    uint32_t size = jitc_var(index)->size;

    if (domain)
        jitc_log(Debug, "jit_vcall(r%u, domain=\"%s\")", index, domain);
    else
        jitc_log(Debug, "jitc_var_vcall_reduce(r%u)", index);

    size_t perm_size    = (size_t) size * (size_t) sizeof(uint32_t),
           offsets_size = (size_t(bucket_count) * 4 + 1) * sizeof(uint32_t);

    if (backend == JitBackend::LLVM)
        perm_size += jitc_llvm_vector_width * sizeof(uint32_t);

    uint8_t *offsets = (uint8_t *) jitc_malloc(
        backend == JitBackend::CUDA ? AllocType::HostPinned : AllocType::Host, offsets_size);
    uint32_t *perm = (uint32_t *) jitc_malloc(
        backend == JitBackend::CUDA ? AllocType::Device : AllocType::HostAsync, perm_size);

    // Compute permutation
    const uint32_t *self = (const uint32_t *) jitc_var_ptr(index);
    uint32_t unique_count = jitc_mkperm(backend, self, size,
                                        bucket_count, perm, (uint32_t *) offsets),
             unique_count_out = unique_count;

    // Register permutation variable with JIT backend and transfer ownership
    uint32_t perm_var = jitc_var_mem_map(backend, VarType::UInt32, perm, size, 1);

    Variable v2;
    v2.kind = (uint32_t) VarKind::Data;
    v2.type = (uint32_t) VarType::UInt32;
    v2.backend = (uint32_t) backend;
    v2.dep[3] = perm_var;
    v2.retain_data = true;
    v2.unaligned = 1;

    struct InputBucket {
        uint32_t id, offset, size, unused;
    };

    InputBucket *input_buckets = (InputBucket *) offsets;

    std::sort(
        input_buckets,
        input_buckets + unique_count,
        [](const InputBucket &b1, const InputBucket &b2) {
            return b1.size > b2.size;
        }
    );

    for (uint32_t i = 0; i < unique_count; ++i) {
        InputBucket bucket = input_buckets[i];

        // Create variable for permutation subrange
        v2.data = perm + bucket.offset;
        v2.size = bucket.size;

        jitc_var_inc_ref(perm_var);

        uint32_t index2 = jitc_var_new(v2);

        VCallBucket bucket_out;

        if (domain)
            bucket_out.ptr = jitc_registry_get_ptr(backend, domain, bucket.id);
        else
            bucket_out.ptr = nullptr;

        bucket_out.index = index2;
        bucket_out.id = bucket.id;

        memcpy(input_buckets + i, &bucket_out, sizeof(VCallBucket));

        jitc_trace("jit_var_vcall_reduce(): registered variable %u: bucket %u "
                   "(" DRJIT_PTR ") of size %u.", index2, bucket_out.id,
                   (uintptr_t) bucket_out.ptr, bucket.size);
    }

    jitc_var_dec_ref(perm_var);

    *bucket_count_out = unique_count_out;

    jitc_var(index)->extra = true;
    Extra &extra = state.extra[index];
    extra.vcall_bucket_count = unique_count_out;
    extra.vcall_buckets = (VCallBucket *) offsets;
    return extra.vcall_buckets;
}
