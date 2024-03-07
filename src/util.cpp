/*
    src/util.cpp -- Parallel reductions and miscellaneous utility routines.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <condition_variable>
#include "internal.h"
#include "util.h"
#include "var.h"
#include "eval.h"
#include "log.h"
#include "call.h"
#include "profile.h"

#if defined(_MSC_VER)
#  pragma warning (disable: 4146) // unary minus operator applied to unsigned type, result still unsigned
#endif

const char *reduction_name[(int) ReduceOp::Count] = { "none", "sum", "mul",
                                                      "min", "max", "and", "or" };

/// Helper function: enqueue parallel CPU task (synchronous or asynchronous)
template <typename Func>
void jitc_submit_cpu(KernelType type, Func &&func, uint32_t width,
                     uint32_t size = 1) {

    struct Payload { Func f; };
    Payload payload{ std::forward<Func>(func) };

    static_assert(std::is_trivially_copyable<Payload>::value &&
                  std::is_trivially_destructible<Payload>::value, "Internal error!");

    Task *new_task = task_submit_dep(
        nullptr, &jitc_task, 1, size,
        [](uint32_t index, void *payload) { ((Payload *) payload)->f(index); },
        &payload, sizeof(Payload), nullptr, 0);

    if (unlikely(jit_flag(JitFlag::LaunchBlocking))) {
        unlock_guard guard(state.lock);
        task_wait(new_task);
    }

    if (unlikely(jit_flag(JitFlag::KernelHistory))) {
        KernelHistoryEntry entry = {};
        entry.backend = JitBackend::LLVM;
        entry.type = type;
        entry.size = width;
        entry.input_count = 1;
        entry.output_count = 1;
        task_retain(new_task);
        entry.task = new_task;
        state.kernel_history.append(entry);
    }

    task_release(jitc_task);
    jitc_task = new_task;
}

void jitc_submit_gpu(KernelType type, CUfunction kernel, uint32_t block_count,
                     uint32_t thread_count, uint32_t shared_mem_bytes,
                     CUstream stream, void **args, void **extra,
                     uint32_t width) {

    KernelHistoryEntry entry = {};

    uint32_t flags = jit_flags();

    if (unlikely(flags & (uint32_t) JitFlag::KernelHistory)) {
        cuda_check(cuEventCreate((CUevent *) &entry.event_start, CU_EVENT_DEFAULT));
        cuda_check(cuEventCreate((CUevent *) &entry.event_end, CU_EVENT_DEFAULT));
        cuda_check(cuEventRecord((CUevent) entry.event_start, stream));
    }

    cuda_check(cuLaunchKernel(kernel, block_count, 1, 1, thread_count, 1, 1,
                              shared_mem_bytes, stream, args, extra));

    if (unlikely(flags & (uint32_t) JitFlag::LaunchBlocking))
        cuda_check(cuStreamSynchronize(stream));

    if (unlikely(flags & (uint32_t) JitFlag::KernelHistory)) {
        entry.backend = JitBackend::CUDA;
        entry.type = type;
        entry.size = width;
        entry.input_count = 1;
        entry.output_count = 1;
        cuda_check(cuEventRecord((CUevent) entry.event_end, stream));

        state.kernel_history.append(entry);
    }
}

/// Fill a device memory region with constants of a given type
void jitc_memset_async(JitBackend backend, void *ptr, uint32_t size_,
                       uint32_t isize, const void *src) {
    ThreadState *ts = thread_state(backend);
    ts->jitc_memset_async(ptr, size_, isize, src);
}

/// Perform a synchronous copy operation
void jitc_memcpy(JitBackend backend, void *dst, const void *src, size_t size) {
    ThreadState *ts = thread_state(backend);

    // Temporarily release the lock while copying
    jitc_sync_thread(ts);
    ts->jitc_memcpy(dst, src, size);
}

/// Perform an asynchronous copy operation
void jitc_memcpy_async(JitBackend backend, void *dst, const void *src, size_t size) {
    ThreadState *ts = thread_state(backend);

    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        cuda_check(cuMemcpyAsync((CUdeviceptr) dst, (CUdeviceptr) src, size,
                                 ts->stream));
    } else {
        jitc_submit_cpu(
            KernelType::Other,
            [dst, src, size](uint32_t) {
                memcpy(dst, src, size);
            },

            (uint32_t) size
        );
    }
}

using Reduction = void (*) (const void *ptr, uint32_t start, uint32_t end, void *out);

template <typename Value>
static Reduction jitc_reduce_create(ReduceOp op) {
    using UInt = uint_with_size_t<Value>;

    switch (op) {
        case ReduceOp::Add:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) JIT_NO_UBSAN {
                const Value *ptr = (const Value *) ptr_;
                Value result = 0;
                for (uint32_t i = start; i != end; ++i)
                    result += ptr[i];
                *((Value *) out) = result;
            };

        case ReduceOp::Mul:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) JIT_NO_UBSAN {
                const Value *ptr = (const Value *) ptr_;
                Value result = 1;
                for (uint32_t i = start; i != end; ++i)
                    result *= ptr[i];
                *((Value *) out) = result;
            };

        case ReduceOp::Max:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const Value *ptr = (const Value *) ptr_;
                Value result = std::is_integral<Value>::value
                                   ?  std::numeric_limits<Value>::min()
                                   : -std::numeric_limits<Value>::infinity();
                for (uint32_t i = start; i != end; ++i)
                    result = std::max(result, ptr[i]);
                *((Value *) out) = result;
            };

        case ReduceOp::Min:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const Value *ptr = (const Value *) ptr_;
                Value result = std::is_integral<Value>::value
                                   ?  std::numeric_limits<Value>::max()
                                   :  std::numeric_limits<Value>::infinity();
                for (uint32_t i = start; i != end; ++i)
                    result = std::min(result, ptr[i]);
                *((Value *) out) = result;
            };

        case ReduceOp::Or:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const UInt *ptr = (const UInt *) ptr_;
                UInt result = 0;
                for (uint32_t i = start; i != end; ++i)
                    result |= ptr[i];
                *((UInt *) out) = result;
            };

        case ReduceOp::And:
            return [](const void *ptr_, uint32_t start, uint32_t end, void *out) {
                const UInt *ptr = (const UInt *) ptr_;
                UInt result = (UInt) -1;
                for (uint32_t i = start; i != end; ++i)
                    result &= ptr[i];
                *((UInt *) out) = result;
            };

        default: jitc_raise("jit_reduce_create(): unsupported reduction type!");
    }
}

static Reduction jitc_reduce_create(VarType type, ReduceOp op) {
    using half = drjit::half;
    switch (type) {
        case VarType::Int8:    return jitc_reduce_create<int8_t  >(op);
        case VarType::UInt8:   return jitc_reduce_create<uint8_t >(op);
        case VarType::Int16:   return jitc_reduce_create<int16_t >(op);
        case VarType::UInt16:  return jitc_reduce_create<uint16_t>(op);
        case VarType::Int32:   return jitc_reduce_create<int32_t >(op);
        case VarType::UInt32:  return jitc_reduce_create<uint32_t>(op);
        case VarType::Int64:   return jitc_reduce_create<int64_t >(op);
        case VarType::UInt64:  return jitc_reduce_create<uint64_t>(op);
        case VarType::Float16: return jitc_reduce_create<half    >(op);
        case VarType::Float32: return jitc_reduce_create<float   >(op);
        case VarType::Float64: return jitc_reduce_create<double  >(op);
        default: jitc_raise("jit_reduce_create(): unsupported data type!");
    }
}

void jitc_reduce(JitBackend backend, VarType type, ReduceOp op, const void *ptr,
                uint32_t size, void *out) {
    ThreadState *ts = thread_state(backend);

    jitc_log(Debug, "jit_reduce(" DRJIT_PTR ", type=%s, op=%s, size=%u)",
            (uintptr_t) ptr, type_name[(int) type],
            reduction_name[(int) op], size);

    uint32_t tsize = type_size[(int) type];

    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        const Device &device = state.devices[ts->device];
        CUfunction func = jitc_cuda_reductions[(int) op][(int) type][device.id];
        if (!func)
            jitc_raise("jit_reduce(): no existing kernel for type=%s, op=%s!",
                      type_name[(int) type], reduction_name[(int) op]);

        uint32_t thread_count = 1024,
                 shared_size = thread_count * tsize,
                 block_count;

        device.get_launch_config(&block_count, nullptr, size, thread_count);

        if (size <= 1024) {
            // This is a small array, do everything in just one reduction.
            void *args[] = { &ptr, &size, &out };

            jitc_submit_gpu(KernelType::Reduce, func, 1, thread_count,
                            shared_size, ts->stream, args, nullptr, size);
        } else {
            void *temp = jitc_malloc(AllocType::Device, size_t(block_count) * tsize);

            // First reduction
            void *args_1[] = { &ptr, &size, &temp };

            jitc_submit_gpu(KernelType::Reduce, func, block_count, thread_count,
                            shared_size, ts->stream, args_1, nullptr, size);

            // Second reduction
            void *args_2[] = { &temp, &block_count, &out };

            jitc_submit_gpu(KernelType::Reduce, func, 1, thread_count,
                            shared_size, ts->stream, args_2, nullptr, size);

            jitc_free(temp);
        }
    } else {
        uint32_t block_size = size, blocks = 1;
        if (pool_size() > 1) {
            block_size = jitc_llvm_block_size;
            blocks     = (size + block_size - 1) / block_size;
        }

        void *target = out;
        if (blocks > 1)
            target = jitc_malloc(AllocType::HostAsync, blocks * tsize);

        Reduction reduction = jitc_reduce_create(type, op);
        jitc_submit_cpu(
            KernelType::Reduce,
            [block_size, size, tsize, ptr, reduction, target](uint32_t index) {
                reduction(ptr, index * block_size,
                          std::min((index + 1) * block_size, size),
                          (uint8_t *) target + index * tsize);
            },

            size,
            std::max(1u, blocks));

        if (blocks > 1) {
            jitc_reduce(backend, type, op, target, blocks, out);
            jitc_free(target);
        }
    }
}

/// 'All' reduction for boolean arrays
bool jitc_all(JitBackend backend, uint8_t *values, uint32_t size) {
    /* When \c size is not a multiple of 4, the implementation will initialize up
       to 3 bytes beyond the end of the supplied range so that an efficient 32 bit
       reduction algorithm can be used. This is fine for allocations made using
       \ref jit_malloc(), which allow for this. */

    uint32_t reduced_size = (size + 3) / 4,
             trailing     = reduced_size * 4 - size;

    jitc_log(Debug, "jit_all(" DRJIT_PTR ", size=%u)", (uintptr_t) values, size);

    if (trailing) {
        bool filler = true;
        jitc_memset_async(backend, values + size, trailing, sizeof(bool), &filler);
    }

    bool result;
    if (backend == JitBackend::CUDA) {
        uint8_t *out = (uint8_t *) jitc_malloc(AllocType::HostPinned, 4);
        jitc_reduce(backend, VarType::UInt32, ReduceOp::And, values, reduced_size, out);
        jitc_sync_thread();
        result = (out[0] & out[1] & out[2] & out[3]) != 0;
        jitc_free(out);
    } else {
        uint8_t out[4];
        jitc_reduce(backend, VarType::UInt32, ReduceOp::And, values, reduced_size, out);
        jitc_sync_thread();
        result = (out[0] & out[1] & out[2] & out[3]) != 0;
    }

    return result;
}

/// 'Any' reduction for boolean arrays
bool jitc_any(JitBackend backend, uint8_t *values, uint32_t size) {
    /* When \c size is not a multiple of 4, the implementation will initialize up
       to 3 bytes beyond the end of the supplied range so that an efficient 32 bit
       reduction algorithm can be used. This is fine for allocations made using
       \ref jit_malloc(), which allow for this. */

    uint32_t reduced_size = (size + 3) / 4,
             trailing     = reduced_size * 4 - size;

    jitc_log(Debug, "jit_any(" DRJIT_PTR ", size=%u)", (uintptr_t) values, size);

    if (trailing) {
        bool filler = false;
        jitc_memset_async(backend, values + size, trailing, sizeof(bool), &filler);
    }

    bool result;
    if (backend == JitBackend::CUDA) {
        uint8_t *out = (uint8_t *) jitc_malloc(AllocType::HostPinned, 4);
        jitc_reduce(backend, VarType::UInt32, ReduceOp::Or, values,
                    reduced_size, out);
        jitc_sync_thread();
        result = (out[0] | out[1] | out[2] | out[3]) != 0;
        jitc_free(out);
    } else {
        uint8_t out[4];
        jitc_reduce(backend, VarType::UInt32, ReduceOp::Or, values,
                    reduced_size, out);
        jitc_sync_thread();
        result = (out[0] | out[1] | out[2] | out[3]) != 0;
    }

    return result;
}

template <typename T>
void sum_reduce_1(uint32_t start, uint32_t end, const void *in_, uint32_t index,
                  void *scratch) {
    const T *in = (const T *) in_;
    T accum = T(0);
    for (uint32_t i = start; i != end; ++i)
        accum += in[i];
    ((T*) scratch)[index] = accum;
}

template <typename T>
void sum_reduce_2(uint32_t start, uint32_t end, const void *in_, void *out_,
                  uint32_t index, const void *scratch, bool exclusive) {
    const T *in = (const T *) in_;
    T *out = (T *) out_;

    T accum;
    if (scratch)
        accum = ((const T *) scratch)[index];
    else
        accum = T(0);

    if (exclusive) {
        for (uint32_t i = start; i != end; ++i) {
            T value = in[i];
            out[i] = accum;
            accum += value;
        }
    } else {
        for (uint32_t i = start; i != end; ++i) {
            T value = in[i];
            accum += value;
            out[i] = accum;
        }
    }
}

void sum_reduce_1(VarType vt, uint32_t start, uint32_t end, const void *in, uint32_t index, void *scratch) {
    switch (vt) {
        case VarType::UInt32:  sum_reduce_1<uint32_t>(start, end, in, index, scratch); break;
        case VarType::UInt64:  sum_reduce_1<uint64_t>(start, end, in, index, scratch); break;
        case VarType::Float32: sum_reduce_1<float>   (start, end, in, index, scratch); break;
        case VarType::Float64: sum_reduce_1<double>  (start, end, in, index, scratch); break;
        default:
            jitc_raise("jit_prefix_sum(): type %s is not supported!", type_name[(int) vt]);
    }
}

void sum_reduce_2(VarType vt, uint32_t start, uint32_t end, const void *in, void *out, uint32_t index, const void *scratch, bool exclusive) {
    switch (vt) {
        case VarType::UInt32:  sum_reduce_2<uint32_t>(start, end, in, out, index, scratch, exclusive); break;
        case VarType::UInt64:  sum_reduce_2<uint64_t>(start, end, in, out, index, scratch, exclusive); break;
        case VarType::Float32: sum_reduce_2<float>   (start, end, in, out, index, scratch, exclusive); break;
        case VarType::Float64: sum_reduce_2<double>  (start, end, in, out, index, scratch, exclusive); break;
        default:
            jitc_raise("jit_prefix_sum(): type %s is not supported!", type_name[(int) vt]);
    }
}

/// Exclusive prefix sum
void jitc_prefix_sum(JitBackend backend, VarType vt, bool exclusive,
               const void *in, uint32_t size, void *out) {
    if (size == 0)
        return;
    if (vt == VarType::Int32)
        vt = VarType::UInt32;

    const uint32_t isize = type_size[(int) vt];
    ThreadState *ts = thread_state(backend);

    if (backend == JitBackend::CUDA) {
        const Device &device = state.devices[ts->device];
        scoped_set_context guard(ts->context);

        if (size == 1) {
            if (exclusive) {
                cuda_check(cuMemsetD8Async((CUdeviceptr) out, 0, isize, ts->stream));
            } else {
                if (in != out)
                    cuda_check(cuMemcpyAsync((CUdeviceptr) out,
                                             (CUdeviceptr) in, isize,
                                             ts->stream));
            }
        } else if ((isize == 4 && size <= 4096) || (isize == 8 && size < 2048)) {
            // Kernel for small arrays
            uint32_t items_per_thread = isize == 8 ? 2 : 4,
                     thread_count     = round_pow2((size + items_per_thread - 1)
                                                    / items_per_thread),
                     shared_size      = thread_count * 2 * isize;

            jitc_log(Debug,
                     "jit_prefix_sum(" DRJIT_PTR " -> " DRJIT_PTR
                     ", type=%s, exclusive=%i, size=%u, type=small, threads=%u, shared=%u)",
                     (uintptr_t) in, (uintptr_t) out, type_name[(int) vt], exclusive, size,
                     thread_count, shared_size);

            CUfunction kernel =
                (exclusive ? jitc_cuda_prefix_sum_exc_small
                           : jitc_cuda_prefix_sum_inc_small)[(int) vt][device.id];

            if (!kernel)
                jitc_raise("jit_prefix_sum(): type %s is not supported!", type_name[(int) vt]);

            void *args[] = { &in, &out, &size };
            jitc_submit_gpu(
                KernelType::Other, kernel, 1,
                thread_count, shared_size, ts->stream, args, nullptr, size);
        } else {
            // Kernel for large arrays
            uint32_t items_per_thread = isize == 8 ? 8 : 16,
                     thread_count     = 128,
                     items_per_block  = items_per_thread * thread_count,
                     block_count      = (size + items_per_block - 1) / items_per_block,
                     shared_size      = items_per_block * isize,
                     scratch_items    = block_count + 32;

            jitc_log(Debug,
                     "jit_prefix_sum(" DRJIT_PTR " -> " DRJIT_PTR
                     ", type=%s, exclusive=%i, size=%u, type=large, blocks=%u, threads=%u, "
                     "shared=%u, scratch=%zu)",
                     (uintptr_t) in, (uintptr_t) out, type_name[(int) vt], exclusive, size,
                     block_count, thread_count, shared_size, scratch_items * sizeof(uint64_t));

            CUfunction kernel =
                (exclusive ? jitc_cuda_prefix_sum_exc_large
                           : jitc_cuda_prefix_sum_inc_large)[(int) vt][device.id];

            if (!kernel)
                jitc_raise("jit_prefix_sum(): type %s is not supported!", type_name[(int) vt]);

            uint64_t *scratch = (uint64_t *) jitc_malloc(
                AllocType::Device, scratch_items * sizeof(uint64_t));

            /// Initialize scratch space and padding
            uint32_t block_count_init, thread_count_init;
            device.get_launch_config(&block_count_init, &thread_count_init,
                                     scratch_items);

            void *args[] = { &scratch, &scratch_items };
            jitc_submit_gpu(KernelType::Other,
                            jitc_cuda_prefix_sum_large_init[device.id],
                            block_count_init, thread_count_init, 0, ts->stream,
                            args, nullptr, scratch_items);

            scratch += 32; // move beyond padding area
            void *args_2[] = { &in, &out, &size, &scratch };
            jitc_submit_gpu(KernelType::Other, kernel, block_count,
                            thread_count, shared_size, ts->stream, args_2,
                            nullptr, scratch_items);
            scratch -= 32;

            jitc_free(scratch);
        }
    } else {
        uint32_t block_size = size, blocks = 1;
        if (pool_size() > 1) {
            block_size = jitc_llvm_block_size;
            blocks     = (size + block_size - 1) / block_size;
        }

        jitc_log(Debug,
                "jit_prefix_sum(" DRJIT_PTR " -> " DRJIT_PTR
                ", size=%u, block_size=%u, blocks=%u)",
                (uintptr_t) in, (uintptr_t) out, size, block_size, blocks);

        void *scratch = nullptr;

        if (blocks > 1) {
            scratch = (void *) jitc_malloc(AllocType::HostAsync, blocks * isize);

            jitc_submit_cpu(
                KernelType::Other,
                [block_size, size, in, vt, scratch](uint32_t index) {
                    uint32_t start = index * block_size,
                             end = std::min(start + block_size, size);

                    sum_reduce_1(vt, start, end, in, index, scratch);
                },
                size, blocks);

            jitc_prefix_sum(backend, vt, true, scratch, blocks, scratch);
        }

        jitc_submit_cpu(
            KernelType::Other,
            [block_size, size, in, out, vt, scratch, exclusive](uint32_t index) {
                uint32_t start = index * block_size,
                         end = std::min(start + block_size, size);

                sum_reduce_2(vt, start, end, in, out, index, scratch, exclusive);
            },
            size, blocks
        );

        jitc_free(scratch);
    }
}

/// Mask compression
uint32_t jitc_compress(JitBackend backend, const uint8_t *in, uint32_t size, uint32_t *out) {
    if (size == 0)
        return 0;

    ThreadState *ts = thread_state(backend);

    if (backend == JitBackend::CUDA) {
        const Device &device = state.devices[ts->device];
        scoped_set_context guard(ts->context);

        uint32_t *count_out = (uint32_t *) jitc_malloc(
            AllocType::HostPinned, sizeof(uint32_t));

        if (size <= 4096) {
            // Kernel for small arrays
            uint32_t items_per_thread = 4,
                     thread_count     = round_pow2((size + items_per_thread - 1)
                                                    / items_per_thread),
                     shared_size      = thread_count * 2 * sizeof(uint32_t),
                     trailer          = thread_count * items_per_thread - size;

            jitc_log(Debug,
                    "jit_compress(" DRJIT_PTR " -> " DRJIT_PTR
                    ", size=%u, type=small, threads=%u, shared=%u)",
                    (uintptr_t) in, (uintptr_t) out, size, thread_count,
                    shared_size);

            if (trailer > 0)
                cuda_check(cuMemsetD8Async((CUdeviceptr) (in + size), 0, trailer,
                                           ts->stream));

            void *args[] = { &in, &out, &size, &count_out };
            jitc_submit_gpu(
                KernelType::Other, jitc_cuda_compress_small[device.id], 1,
                thread_count, shared_size, ts->stream, args, nullptr, size);
        } else {
            // Kernel for large arrays
            uint32_t items_per_thread = 16,
                     thread_count     = 128,
                     items_per_block  = items_per_thread * thread_count,
                     block_count      = (size + items_per_block - 1) / items_per_block,
                     shared_size      = items_per_block * sizeof(uint32_t),
                     scratch_items    = block_count + 32,
                     trailer          = items_per_block * block_count - size;

            jitc_log(Debug,
                    "jit_compress(" DRJIT_PTR " -> " DRJIT_PTR
                    ", size=%u, type=large, blocks=%u, threads=%u, shared=%u, "
                    "scratch=%u)",
                    (uintptr_t) in, (uintptr_t) out, size, block_count,
                    thread_count, shared_size, scratch_items * 4);

            uint64_t *scratch = (uint64_t *) jitc_malloc(
                AllocType::Device, scratch_items * sizeof(uint64_t));

            // Initialize scratch space and padding
            uint32_t block_count_init, thread_count_init;
            device.get_launch_config(&block_count_init, &thread_count_init,
                                     scratch_items);

            void *args[] = { &scratch, &scratch_items };
            jitc_submit_gpu(KernelType::Other,
                            jitc_cuda_prefix_sum_large_init[device.id],
                            block_count_init, thread_count_init, 0, ts->stream,
                            args, nullptr, scratch_items);

            if (trailer > 0)
                cuda_check(cuMemsetD8Async((CUdeviceptr) (in + size), 0, trailer,
                                           ts->stream));

            scratch += 32; // move beyond padding area
            void *args_2[] = { &in, &out, &scratch, &count_out };
            jitc_submit_gpu(KernelType::Other,
                            jitc_cuda_compress_large[device.id], block_count,
                            thread_count, shared_size, ts->stream, args_2,
                            nullptr, scratch_items);
            scratch -= 32;

            jitc_free(scratch);
        }
        jitc_sync_thread();
        uint32_t count_out_v = *count_out;
        jitc_free(count_out);
        return count_out_v;
    } else {
        uint32_t block_size = size, blocks = 1;
        if (pool_size() > 1) {
            block_size = jitc_llvm_block_size;
            blocks     = (size + block_size - 1) / block_size;
        }

        uint32_t count_out = 0;

        jitc_log(Debug,
                "jit_compress(" DRJIT_PTR " -> " DRJIT_PTR
                ", size=%u, block_size=%u, blocks=%u)",
                (uintptr_t) in, (uintptr_t) out, size, block_size, blocks);

        uint32_t *scratch = nullptr;

        if (blocks > 1) {
            scratch = (uint32_t *) jitc_malloc(AllocType::HostAsync,
                                               blocks * sizeof(uint32_t));

            jitc_submit_cpu(
                KernelType::Other,
                [block_size, size, in, scratch](uint32_t index) {
                    uint32_t start = index * block_size,
                             end = std::min(start + block_size, size);

                    uint32_t accum = 0;
                    for (uint32_t i = start; i != end; ++i)
                        accum += (uint32_t) in[i];

                    scratch[index] = accum;
                },

                size, blocks
            );

            jitc_prefix_sum(backend, VarType::UInt32, true, scratch, blocks, scratch);
        }

        jitc_submit_cpu(
            KernelType::Other,
            [block_size, size, scratch, in, out, &count_out](uint32_t index) {
                uint32_t start = index * block_size,
                         end = std::min(start + block_size, size);

                uint32_t accum = 0;
                if (scratch)
                    accum = scratch[index];

                for (uint32_t i = start; i != end; ++i) {
                    uint32_t value = (uint32_t) in[i];
                    if (value)
                        out[accum] = i;
                    accum += value;
                }

                if (end == size)
                    count_out = accum;
            },

            size, blocks
        );

        jitc_free(scratch);
        jitc_sync_thread();

        return count_out;
    }
}

static void cuda_transpose(ThreadState *ts, const uint32_t *in, uint32_t *out,
                           uint32_t rows, uint32_t cols) {
    const Device &device = state.devices[ts->device];

    uint16_t blocks_x = (uint16_t) ((cols + 15u) / 16u),
             blocks_y = (uint16_t) ((rows + 15u) / 16u);

    scoped_set_context guard(ts->context);
    jitc_log(Debug,
            "jit_transpose(" DRJIT_PTR " -> " DRJIT_PTR
            ", rows=%u, cols=%u, blocks=%ux%u)",
            (uintptr_t) in, (uintptr_t) out, rows, cols, blocks_x, blocks_y);

    void *args[] = { &in, &out, &rows, &cols };

    cuda_check(cuLaunchKernel(
        jitc_cuda_transpose[device.id], blocks_x, blocks_y, 1, 16, 16, 1,
        16 * 17 * sizeof(uint32_t), ts->stream, args, nullptr));
}

static ProfilerRegion profiler_region_mkperm("jit_mkperm");
static ProfilerRegion profiler_region_mkperm_phase_1("jit_mkperm_phase_1");
static ProfilerRegion profiler_region_mkperm_phase_2("jit_mkperm_phase_2");

/// Compute a permutation to reorder an integer array into a sorted configuration
uint32_t jitc_mkperm(JitBackend backend, const uint32_t *ptr, uint32_t size,
                     uint32_t bucket_count, uint32_t *perm, uint32_t *offsets) {
    if (size == 0)
        return 0;
    else if (unlikely(bucket_count == 0))
        jitc_fail("jit_mkperm(): bucket_count cannot be zero!");

    ProfilerPhase profiler(profiler_region_mkperm);
    ThreadState *ts = thread_state(backend);

    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        const Device &device = state.devices[ts->device];

        // Don't use more than 1 block/SM due to shared memory requirement
        const uint32_t warp_size = 32;
        uint32_t block_count, thread_count;
        device.get_launch_config(&block_count, &thread_count, size, 1024, 1);

        // Always launch full warps (the kernel impl. assumes that this is the case)
        uint32_t warp_count = (thread_count + warp_size - 1) / warp_size;
        thread_count = warp_count * warp_size;

        uint32_t bucket_size_1   = bucket_count * sizeof(uint32_t),
                 bucket_size_all = bucket_size_1 * block_count;

        /* If there is a sufficient amount of shared memory, atomically accumulate into a
           shared memory buffer. Otherwise, use global memory, which is much slower. */
        uint32_t shared_size = 0;
        const char *variant = nullptr;
        CUfunction phase_1 = nullptr, phase_4 = nullptr;
        bool initialize_buckets = false;

        if (bucket_size_1 * warp_count <= device.shared_memory_bytes) {
            /* "Tiny" variant, which uses shared memory atomics to produce a stable
               permutation. Handles up to 512 buckets with 64KiB of shared memory. */

            phase_1 = jitc_cuda_mkperm_phase_1_tiny[device.id];
            phase_4 = jitc_cuda_mkperm_phase_4_tiny[device.id];
            shared_size = bucket_size_1 * warp_count;
            bucket_size_all *= warp_count;
            variant = "tiny";
        } else if (bucket_size_1 <= device.shared_memory_bytes) {
            /* "Small" variant, which uses shared memory atomics and handles up to
               16K buckets with 64KiB of shared memory. The permutation can be
               somewhat unstable due to scheduling variations when performing atomic
               operations (although some effort is made to keep it stable within
               each group of 32 elements by performing an intra-warp reduction.) */

            phase_1 = jitc_cuda_mkperm_phase_1_small[device.id];
            phase_4 = jitc_cuda_mkperm_phase_4_small[device.id];
            shared_size = bucket_size_1;
            variant = "small";
        } else {
            /* "Large" variant, which uses global memory atomics and handles
               arbitrarily many elements (though this is somewhat slower than the
               previous two shared memory variants). The permutation can be somewhat
               unstable due to scheduling variations when performing atomic
               operations (although some effort is made to keep it stable within
               each group of 32 elements by performing an intra-warp reduction.)
               Buckets must be zero-initialized explicitly. */

            phase_1 = jitc_cuda_mkperm_phase_1_large[device.id];
            phase_4 = jitc_cuda_mkperm_phase_4_large[device.id];
            variant = "large";
            initialize_buckets = true;
        }

        bool needs_transpose = bucket_size_1 != bucket_size_all;
        uint32_t *buckets_1, *buckets_2, *counter = nullptr;
        buckets_1 = buckets_2 =
            (uint32_t *) jitc_malloc(AllocType::Device, bucket_size_all);

        // Scratch space for matrix transpose operation
        if (needs_transpose)
            buckets_2 = (uint32_t *) jitc_malloc(AllocType::Device, bucket_size_all);

        if (offsets) {
            counter = (uint32_t *) jitc_malloc(AllocType::Device, sizeof(uint32_t)),
            cuda_check(cuMemsetD8Async((CUdeviceptr) counter, 0, sizeof(uint32_t),
                                       ts->stream));
        }

        if (initialize_buckets)
            cuda_check(cuMemsetD8Async((CUdeviceptr) buckets_1, 0,
                                       bucket_size_all, ts->stream));

        /* Determine the amount of work to be done per block, and ensure that it is
           divisible by the warp size (the kernel implementation assumes this.) */
        uint32_t size_per_block = (size + block_count - 1) / block_count;
        size_per_block = (size_per_block + warp_size - 1) / warp_size * warp_size;

        jitc_log(Debug,
                "jit_mkperm(" DRJIT_PTR
                ", size=%u, bucket_count=%u, block_count=%u, thread_count=%u, "
                "size_per_block=%u, variant=%s, shared_size=%u)",
                (uintptr_t) ptr, size, bucket_count, block_count, thread_count,
                size_per_block, variant, shared_size);

        // Phase 1: Count the number of occurrences per block
        void *args_1[] = { &ptr, &buckets_1, &size, &size_per_block,
                           &bucket_count };

        jitc_submit_gpu(KernelType::CallReduce, phase_1, block_count,
                        thread_count, shared_size, ts->stream, args_1, nullptr,
                        size);

        // Phase 2: exclusive prefix sum over transposed buckets
        if (needs_transpose)
            cuda_transpose(ts, buckets_1, buckets_2,
                           bucket_size_all / bucket_size_1, bucket_count);

        jitc_prefix_sum(backend, VarType::UInt32, true, buckets_2,
                  bucket_size_all / sizeof(uint32_t), buckets_2);

        if (needs_transpose)
            cuda_transpose(ts, buckets_2, buckets_1, bucket_count,
                           bucket_size_all / bucket_size_1);

        // Phase 3: collect non-empty buckets (optional)
        if (likely(offsets)) {
            uint32_t block_count_3, thread_count_3;
            device.get_launch_config(&block_count_3, &thread_count_3,
                                     bucket_count * block_count);

            // Round up to a multiple of the thread count
            uint32_t bucket_count_rounded =
                (bucket_count + thread_count_3 - 1) / thread_count_3 * thread_count_3;

            void *args_3[] = { &buckets_1, &bucket_count, &bucket_count_rounded,
                               &size,      &counter,      &offsets };

            jitc_submit_gpu(KernelType::CallReduce,
                            jitc_cuda_mkperm_phase_3[device.id], block_count_3,
                            thread_count_3, sizeof(uint32_t) * thread_count_3,
                            ts->stream, args_3, nullptr, size);

            cuda_check(cuMemcpyAsync((CUdeviceptr) (offsets + 4 * size_t(bucket_count)),
                                     (CUdeviceptr) counter, sizeof(uint32_t),
                                     ts->stream));

            cuda_check(cuEventRecord(ts->event, ts->stream));
        }

        // Phase 4: write out permutation based on bucket counts
        void *args_4[] = { &ptr, &buckets_1, &perm, &size, &size_per_block,
                           &bucket_count };

        jitc_submit_gpu(KernelType::CallReduce, phase_4, block_count,
                        thread_count, shared_size, ts->stream, args_4, nullptr,
                        size);

        if (likely(offsets)) {
            unlock_guard guard_2(state.lock);
            cuda_check(cuEventSynchronize(ts->event));
        }

        jitc_free(buckets_1);
        if (needs_transpose)
            jitc_free(buckets_2);
        jitc_free(counter);

        return offsets ? offsets[4 * bucket_count] : 0u;
    } else { // if (!ts->cuda)
        uint32_t blocks = 1, block_size = size, pool_size = ::pool_size();

        if (pool_size > 1) {
            // Try to spread out uniformly over cores
            blocks = pool_size * 4;
            block_size = (size + blocks - 1) / blocks;

            // But don't make the blocks too small
            block_size = std::max(jitc_llvm_block_size, block_size);

            // Finally re-adjust block count given the selected block size
            blocks = (size + block_size - 1) / block_size;
        }

        jitc_log(Debug,
                "jit_mkperm(" DRJIT_PTR
                ", size=%u, bucket_count=%u, block_size=%u, blocks=%u)",
                (uintptr_t) ptr, size, bucket_count, block_size, blocks);

        uint32_t **buckets =
            (uint32_t **) jitc_malloc(AllocType::HostAsync, sizeof(uint32_t *) * blocks);

        uint32_t unique_count = 0;

        // Phase 1
        jitc_submit_cpu(
            KernelType::CallReduce,
            [block_size, size, buckets, bucket_count, ptr](uint32_t index) {
                ProfilerPhase profiler(profiler_region_mkperm_phase_1);
                uint32_t start = index * block_size,
                         end = std::min(start + block_size, size);

                size_t bsize = sizeof(uint32_t) * (size_t) bucket_count;
                uint32_t *buckets_local = (uint32_t *) malloc_check(bsize);
                memset(buckets_local, 0, bsize);

                 for (uint32_t i = start; i != end; ++i)
                     buckets_local[ptr[i]]++;

                 buckets[index] = buckets_local;
            },

            size, blocks
        );

        // Local accumulation step
        jitc_submit_cpu(
            KernelType::CallReduce,
            [bucket_count, blocks, buckets, offsets, &unique_count](uint32_t) {
                uint32_t sum = 0, unique_count_local = 0;
                for (uint32_t i = 0; i < bucket_count; ++i) {
                    uint32_t sum_local = 0;
                    for (uint32_t j = 0; j < blocks; ++j) {
                        uint32_t value = buckets[j][i];
                        buckets[j][i] = sum + sum_local;
                        sum_local += value;
                    }
                    if (sum_local > 0) {
                        if (offsets) {
                            offsets[unique_count_local*4] = i;
                            offsets[unique_count_local*4 + 1] = sum;
                            offsets[unique_count_local*4 + 2] = sum_local;
                            offsets[unique_count_local*4 + 3] = 0;
                        }
                        unique_count_local++;
                        sum += sum_local;
                    }
                }

                unique_count = unique_count_local;
            },

            size
        );

        Task *local_task = jitc_task;
        task_retain(local_task);

        // Phase 2
        jitc_submit_cpu(
            KernelType::CallReduce,
            [block_size, size, buckets, perm, ptr](uint32_t index) {
                ProfilerPhase profiler(profiler_region_mkperm_phase_2);

                uint32_t start = index * block_size,
                         end = std::min(start + block_size, size);

                uint32_t *buckets_local = buckets[index];

                for (uint32_t i = start; i != end; ++i) {
                    uint32_t idx = buckets_local[ptr[i]]++;
                    perm[idx] = i;
                }

                free(buckets_local);
            },

            size, blocks
        );

        // Free memory (happens asynchronously after the above stmt.)
        jitc_free(buckets);

        unlock_guard guard(state.lock);
        task_wait_and_release(local_task);
        return unique_count;
    }
}

using BlockOp = void (*) (const void *ptr, void *out, uint32_t start, uint32_t end, uint32_t block_size);

template <typename Value> static BlockOp jitc_block_copy_create() {
    return [](const void *in_, void *out_, uint32_t start, uint32_t end, uint32_t block_size) {
        const Value *in = (const Value *) in_ + start;
        Value *out = (Value *) out_ + start * block_size;
        for (uint32_t i = start; i != end; ++i) {
            Value value = *in++;
            for (uint32_t j = 0; j != block_size; ++j)
                *out++ = value;
        }
    };
}

template <typename Value> static BlockOp jitc_block_sum_create() {
    return [](const void *in_, void *out_, uint32_t start, uint32_t end, uint32_t block_size) {
        const Value *in = (const Value *) in_ + start * block_size;
        Value *out = (Value *) out_ + start;
        for (uint32_t i = start; i != end; ++i) {
            Value sum = 0;
            for (uint32_t j = 0; j != block_size; ++j)
                sum += *in++;
            *out++ = sum;
        }
    };
}

static BlockOp jitc_block_copy_create(VarType type) {
    switch (type) {
        case VarType::UInt8:   return jitc_block_copy_create<uint8_t >();
        case VarType::UInt16:  return jitc_block_copy_create<uint16_t>();
        case VarType::UInt32:  return jitc_block_copy_create<uint32_t>();
        case VarType::UInt64:  return jitc_block_copy_create<uint64_t>();
        case VarType::Float32: return jitc_block_copy_create<float   >();
        case VarType::Float64: return jitc_block_copy_create<double  >();
        default: jitc_raise("jit_block_copy_create(): unsupported data type!");
    }
}

static BlockOp jitc_block_sum_create(VarType type) {
    switch (type) {
        case VarType::UInt8:   return jitc_block_sum_create<uint8_t >();
        case VarType::UInt16:  return jitc_block_sum_create<uint16_t>();
        case VarType::UInt32:  return jitc_block_sum_create<uint32_t>();
        case VarType::UInt64:  return jitc_block_sum_create<uint64_t>();
        case VarType::Float32: return jitc_block_sum_create<float   >();
        case VarType::Float64: return jitc_block_sum_create<double  >();
        default: jitc_raise("jit_block_sum_create(): unsupported data type!");
    }
}

static VarType make_int_type_unsigned(VarType type) {
    switch (type) {
        case VarType::Int8:  return VarType::UInt8;
        case VarType::Int16: return VarType::UInt16;
        case VarType::Int32: return VarType::UInt32;
        case VarType::Int64: return VarType::UInt64;
        default: return type;
    }
}

/// Replicate individual input elements to larger blocks
void jitc_block_copy(JitBackend backend, enum VarType type, const void *in, void *out,
                    uint32_t size, uint32_t block_size) {
    if (block_size == 0)
        jitc_raise("jit_block_copy(): block_size cannot be zero!");

    jitc_log(Debug,
            "jit_block_copy(" DRJIT_PTR " -> " DRJIT_PTR
            ", type=%s, block_size=%u, size=%u)",
            (uintptr_t) in, (uintptr_t) out,
            type_name[(int) type], block_size, size);

    if (block_size == 1) {
        uint32_t tsize = type_size[(int) type];
        jitc_memcpy_async(backend, out, in, size * tsize);
        return;
    }

    type = make_int_type_unsigned(type);

    ThreadState *ts = thread_state(backend);
    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        const Device &device = state.devices[ts->device];
        size *= block_size;

        CUfunction func = jitc_cuda_block_copy[(int) type][device.id];
        if (!func)
            jitc_raise("jit_block_copy(): no existing kernel for type=%s!",
                      type_name[(int) type]);

        uint32_t thread_count = std::min(size, 1024u),
                 block_count  = (size + thread_count - 1) / thread_count;

        void *args[] = { &in, &out, &size, &block_size };
        jitc_submit_gpu(KernelType::Other, func, block_count, thread_count, 0,
                        ts->stream, args, nullptr, size);
    } else {
        uint32_t work_unit_size = size, work_units = 1;
        if (pool_size() > 1) {
            work_unit_size = jitc_llvm_block_size;
            work_units     = (size + work_unit_size - 1) / work_unit_size;
        }

        BlockOp op = jitc_block_copy_create(type);

        jitc_submit_cpu(
            KernelType::Other,
            [in, out, op, work_unit_size, size, block_size](uint32_t index) {
                uint32_t start = index * work_unit_size,
                         end = std::min(start + work_unit_size, size);

                op(in, out, start, end, block_size);
            },

            size, work_units
        );
    }
}

/// Sum over elements within blocks
void jitc_block_sum(JitBackend backend, enum VarType type, const void *in, void *out,
                    uint32_t size, uint32_t block_size) {
    if (block_size == 0)
        jitc_raise("jit_block_sum(): block_size cannot be zero!");

    jitc_log(Debug,
            "jit_block_sum(" DRJIT_PTR " -> " DRJIT_PTR
            ", type=%s, block_size=%u, size=%u)",
            (uintptr_t) in, (uintptr_t) out,
            type_name[(int) type], block_size, size);

    uint32_t tsize = type_size[(int) type];
    size_t out_size = size * tsize;

    if (block_size == 1) {
        jitc_memcpy_async(backend, out, in, out_size);
        return;
    }

    type = make_int_type_unsigned(type);

    ThreadState *ts = thread_state(backend);
    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        const Device &device = state.devices[ts->device];

        size *= block_size;

        CUfunction func = jitc_cuda_block_sum[(int) type][device.id];
        if (!func)
            jitc_raise("jit_block_sum(): no existing kernel for type=%s!",
                      type_name[(int) type]);

        uint32_t thread_count = std::min(size, 1024u),
                 block_count  = (size + thread_count - 1) / thread_count;

        void *args[] = { &in, &out, &size, &block_size };
        cuda_check(cuMemsetD8Async((CUdeviceptr) out, 0, out_size, ts->stream));
        jitc_submit_gpu(KernelType::Other, func, block_count, thread_count, 0,
                        ts->stream, args, nullptr, size);
    } else {
        uint32_t work_unit_size = size, work_units = 1;
        if (pool_size() > 1) {
            work_unit_size = jitc_llvm_block_size;
            work_units     = (size + work_unit_size - 1) / work_unit_size;
        }

        BlockOp op = jitc_block_sum_create(type);

        jitc_submit_cpu(
            KernelType::Other,
            [in, out, op, work_unit_size, size, block_size](uint32_t index) {
                uint32_t start = index * work_unit_size,
                         end = std::min(start + work_unit_size, size);

                op(in, out, start, end, block_size);
            },

            size, work_units
        );
    }
}

/// Asynchronously update a single element in memory
void jitc_poke(JitBackend backend, void *dst, const void *src, uint32_t size) {
    jitc_log(Debug, "jit_poke(" DRJIT_PTR ", size=%u)", (uintptr_t) dst, size);

    VarType type;
    switch (size) {
        case 1: type = VarType::UInt8; break;
        case 2: type = VarType::UInt16; break;
        case 4: type = VarType::UInt32; break;
        case 8: type = VarType::UInt64; break;
        default:
            jitc_raise("jit_poke(): only size=1, 2, 4 or 8 are supported!");
    }

    ThreadState *ts = thread_state(backend);
    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        const Device &device = state.devices[ts->device];
        CUfunction func = jitc_cuda_poke[(int) type][device.id];
        void *args[] = { &dst, (void *) src };
        jitc_submit_gpu(KernelType::Other, func, 1, 1, 0,
                        ts->stream, args, nullptr, 1);
    } else {
        uint8_t src8[8] { };
        memcpy(&src8, src, size);

        jitc_submit_cpu(
            KernelType::Other,
            [src8, size, dst](uint32_t) {
                memcpy(dst, &src8, size);
            },

            size
        );
    }
}

void jitc_aggregate(JitBackend backend, void *dst_, AggregationEntry *agg,
                    uint32_t size) {
    ThreadState *ts = thread_state(backend);

    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        const Device &device = state.devices[ts->device];
        CUfunction func = jitc_cuda_aggregate[device.id];
        void *args[] = { &dst_, &agg, &size };

        uint32_t block_count, thread_count;
        device.get_launch_config(&block_count, &thread_count, size);

        jitc_log(InfoSym,
                 "jit_aggregate(" DRJIT_PTR " -> " DRJIT_PTR
                 ", size=%u, blocks=%u, threads=%u)",
                 (uintptr_t) agg, (uintptr_t) dst_, size, block_count,
                 thread_count);

        jitc_submit_gpu(KernelType::Other, func, block_count, thread_count, 0,
                        ts->stream, args, nullptr, 1);

        jitc_free(agg);
    } else {
        uint32_t work_unit_size = size, work_units = 1;
        if (pool_size() > 1) {
            work_unit_size = jitc_llvm_block_size;
            work_units     = (size + work_unit_size - 1) / work_unit_size;
        }

        jitc_log(InfoSym,
                 "jit_aggregate(" DRJIT_PTR " -> " DRJIT_PTR
                 ", size=%u, work_units=%u)",
                 (uintptr_t) agg, (uintptr_t) dst_, size, work_units);

        jitc_submit_cpu(
            KernelType::Other,
            [dst_, agg, size, work_unit_size](uint32_t index) {
                uint32_t start = index * work_unit_size,
                         end = std::min(start + work_unit_size, size);

                for (uint32_t i = start; i != end; ++i) {
                    AggregationEntry e = agg[i];

                    const void *src = e.src;
                    void *dst = (uint8_t *) dst_ + e.offset;

                    switch (e.size) {
                        case  1: *(uint8_t *)  dst =  (uint8_t)  (uintptr_t) src; break;
                        case  2: *(uint16_t *) dst =  (uint16_t) (uintptr_t) src; break;
                        case  4: *(uint32_t *) dst =  (uint32_t) (uintptr_t) src; break;
                        case  8: *(uint64_t *) dst =  (uint64_t) (uintptr_t) src; break;
                        case -1: *(uint8_t *)  dst = *(uint8_t *)  src; break;
                        case -2: *(uint16_t *) dst = *(uint16_t *) src; break;
                        case -4: *(uint32_t *) dst = *(uint32_t *) src; break;
                        case -8: *(uint64_t *) dst = *(uint64_t *) src; break;
                    }
                }
            },
            size, work_units);

        jitc_submit_cpu(
            KernelType::Other, [agg](uint32_t) { free(agg); }, 1, 1);
    }
}

void jitc_enqueue_host_func(JitBackend backend, void (*callback)(void *),
                            void *payload) {
    ThreadState *ts = thread_state(backend);

    if (backend == JitBackend::CUDA) {
        scoped_set_context guard(ts->context);
        cuda_check(cuLaunchHostFunc(ts->stream, callback, payload));
    } else {
        if (!jitc_task) {
            unlock_guard guard(state.lock);
            callback(payload);
        } else {
            jitc_submit_cpu(
                KernelType::Other, [payload, callback](uint32_t) { callback(payload); }, 1, 1);
        }
    }
}

using ReduceExpanded = void (*) (void *ptr, uint32_t start, uint32_t end, uint32_t exp, uint32_t size);

template <typename Value, typename Op>
static void jitc_reduce_expanded_impl(void *ptr_, uint32_t start, uint32_t end,
                                 uint32_t exp, uint32_t size) {
    Value *ptr = (Value *) ptr_;
    Op op;

    const uint32_t block = 128;

    uint32_t i = start;
    for (; i + block <= end; i += block)
        for (uint32_t j = 1; j < exp; ++j)
            for (uint32_t k = 0; k < block; ++k)
                ptr[i + k] = op(ptr[i + k], ptr[i + k + j * size]);

    for (; i < end; i += 1)
        for (uint32_t j = 1; j < exp; ++j)
            ptr[i] = op(ptr[i], ptr[i + j * size]);
}

template <typename Value>
static ReduceExpanded jitc_reduce_expanded_create(ReduceOp op) {
    using UInt = uint_with_size_t<Value>;

    struct Add { Value operator()(Value a, Value b) JIT_NO_UBSAN { return a + b; }};
    struct Mul { Value operator()(Value a, Value b) JIT_NO_UBSAN { return a * b; }};
    struct Min { Value operator()(Value a, Value b) { return std::min(a, b); }};
    struct Max { Value operator()(Value a, Value b) { return std::max(a, b); }};
    struct And {
        Value operator()(Value a, Value b) {
            if constexpr (std::is_integral_v<Value>)
                return a & b;
            else
                return 0;
        }
    };
    struct Or {
        Value operator()(Value a, Value b) {
            if constexpr (std::is_integral_v<Value>)
                return a | b;
            else
                return 0;
        }
    };

    switch (op) {
        case ReduceOp::Add: return jitc_reduce_expanded_impl<Value, Add>;
        case ReduceOp::Mul: return jitc_reduce_expanded_impl<Value, Mul>;
        case ReduceOp::Max: return jitc_reduce_expanded_impl<Value, Max>;
        case ReduceOp::Min: return jitc_reduce_expanded_impl<Value, Min>;
        case ReduceOp::And: return jitc_reduce_expanded_impl<Value, And>;
        case ReduceOp::Or: return jitc_reduce_expanded_impl<Value, Or>;

        default: jitc_raise("jit_reduce_expanded_create(): unsupported reduction type!");
    }
}


static ReduceExpanded jitc_reduce_expanded_create(VarType type, ReduceOp op) {
    using half = drjit::half;
    switch (type) {
        case VarType::Int32:   return jitc_reduce_expanded_create<int32_t >(op);
        case VarType::UInt32:  return jitc_reduce_expanded_create<uint32_t>(op);
        case VarType::Int64:   return jitc_reduce_expanded_create<int64_t >(op);
        case VarType::UInt64:  return jitc_reduce_expanded_create<uint64_t>(op);
        case VarType::Float16: return jitc_reduce_expanded_create<half    >(op);
        case VarType::Float32: return jitc_reduce_expanded_create<float   >(op);
        case VarType::Float64: return jitc_reduce_expanded_create<double  >(op);
        default: jitc_raise("jit_reduce_create(): unsupported data type!");
    }
}

void jitc_reduce_expanded(VarType vt, ReduceOp op, void *ptr, uint32_t exp, uint32_t size) {
    jitc_log(Debug, "jit_reduce_expanded(" DRJIT_PTR ", type=%s, op=%s, expfactor=%u, size=%u)",
            (uintptr_t) ptr, type_name[(int) vt],
            reduction_name[(int) op], exp, size);

    ReduceExpanded kernel = jitc_reduce_expanded_create(vt, op);

    uint32_t block_size = size, blocks = 1;
    if (pool_size() > 1) {
        block_size = jitc_llvm_block_size;
        blocks     = (size + block_size - 1) / block_size;
    }

    jitc_submit_cpu(
        KernelType::Reduce,
        [ptr, block_size, exp, size, kernel](uint32_t index) {
            kernel(ptr, index * block_size,
                   std::min((index + 1) * block_size, size), exp, size);
        },

        size, std::max(1u, blocks));
}
