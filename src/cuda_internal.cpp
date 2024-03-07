#include "cuda_internal.h"
#include "var.h"
#include "log.h"

const static char *reduction_name[(int) ReduceOp::Count] = { "none", "sum", "mul",
                                                      "min", "max", "and", "or" };

static void jitc_submit_gpu(KernelType type, CUfunction kernel, uint32_t block_count,
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
void CUDAThreadState::jitc_memset_async(void *ptr, uint32_t size_,
                                        uint32_t isize, const void *src){
    
    if (isize != 1 && isize != 2 && isize != 4 && isize != 8)
        jitc_raise("CUDAThreadState::jit_memset_async(): invalid element size (must be 1, 2, 4, or 8)!");

    jitc_trace("CUDAThreadState::jit_memset_async(" DRJIT_PTR ", isize=%u, size=%u)",
              (uintptr_t) ptr, isize, size_);

    if (size_ == 0)
        return;

    size_t size = size_;

    // Try to convert into ordinary memset if possible
    uint64_t zero = 0;
    if (memcmp(src, &zero, isize) == 0) {
        size *= isize;
        isize = 1;
    }
    
    // CUDA Specific
    scoped_set_context guard(this->context);
    switch (isize) {
        case 1:
            cuda_check(cuMemsetD8Async((CUdeviceptr) ptr,
                                       ((uint8_t *) src)[0], size,
                                       this->stream));
            break;

        case 2:
            cuda_check(cuMemsetD16Async((CUdeviceptr) ptr,
                                        ((uint16_t *) src)[0], size,
                                        this->stream));
            break;

        case 4:
            cuda_check(cuMemsetD32Async((CUdeviceptr) ptr,
                                        ((uint32_t *) src)[0], size,
                                        this->stream));
            break;

        case 8: {
                const Device &device = state.devices[this->device];
                uint32_t block_count, thread_count;
                device.get_launch_config(&block_count, &thread_count, size_);
                void *args[] = { &ptr, &size_, (void *) src };
                CUfunction kernel = jitc_cuda_fill_64[device.id];
                jitc_submit_gpu(KernelType::Other, kernel, block_count,
                                thread_count, 0, this->stream, args, nullptr,
                                size_);
            }
            break;
    }
}

void CUDAThreadState::jitc_reduce(VarType type, ReduceOp op, const void *ptr,
                                  uint32_t size, void *out) {
    
    jitc_log(Debug, "jit_reduce(" DRJIT_PTR ", type=%s, op=%s, size=%u)",
            (uintptr_t) ptr, type_name[(int) type],
            reduction_name[(int) op], size);

    uint32_t tsize = type_size[(int) type];
    
    // CUDA specific
    scoped_set_context guard(this->context);
    const Device &device = state.devices[this->device];
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
                        shared_size, this->stream, args, nullptr, size);
    } else {
        void *temp = jitc_malloc(AllocType::Device, size_t(block_count) * tsize);

        // First reduction
        void *args_1[] = { &ptr, &size, &temp };

        jitc_submit_gpu(KernelType::Reduce, func, block_count, thread_count,
                        shared_size, this->stream, args_1, nullptr, size);

        // Second reduction
        void *args_2[] = { &temp, &block_count, &out };

        jitc_submit_gpu(KernelType::Reduce, func, 1, thread_count,
                        shared_size, this->stream, args_2, nullptr, size);

        jitc_free(temp);
    }
}

bool CUDAThreadState::jitc_all(uint8_t *values, uint32_t size) {
    /* When \c size is not a multiple of 4, the implementation will initialize up
       to 3 bytes beyond the end of the supplied range so that an efficient 32 bit
       reduction algorithm can be used. This is fine for allocations made using
       \ref jit_malloc(), which allow for this. */

    uint32_t reduced_size = (size + 3) / 4,
             trailing     = reduced_size * 4 - size;

    jitc_log(Debug, "jit_all(" DRJIT_PTR ", size=%u)", (uintptr_t) values, size);

    if (trailing) {
        bool filler = true;
        this->jitc_memset_async(values + size, trailing, sizeof(bool), &filler);
    }

    // CUDA specific
    bool result;
    
    uint8_t *out = (uint8_t *) jitc_malloc(AllocType::HostPinned, 4);
    this->jitc_reduce(VarType::UInt32, ReduceOp::And, values, reduced_size, out);
    jitc_sync_thread();
    result = (out[0] & out[1] & out[2] & out[3]) != 0;
    jitc_free(out);

    return result;
}

bool CUDAThreadState::jitc_any(uint8_t *values, uint32_t size) {
    /* When \c size is not a multiple of 4, the implementation will initialize up
       to 3 bytes beyond the end of the supplied range so that an efficient 32 bit
       reduction algorithm can be used. This is fine for allocations made using
       \ref jit_malloc(), which allow for this. */
    
    uint32_t reduced_size = (size + 3) / 4,
             trailing     = reduced_size * 4 - size;

    jitc_log(Debug, "jit_any(" DRJIT_PTR ", size=%u)", (uintptr_t) values, size);

    if (trailing) {
        bool filler = false;
        this->jitc_memset_async(values + size, trailing, sizeof(bool), &filler);
    }

    // CUDA specific
    bool result;
    
    uint8_t *out = (uint8_t *) jitc_malloc(AllocType::HostPinned, 4);
    this->jitc_reduce(VarType::UInt32, ReduceOp::Or, values,
                      reduced_size, out);
    jitc_sync_thread();
    result = (out[0] | out[1] | out[2] | out[3]) != 0;
    jitc_free(out);

    return result;
    
}

void CUDAThreadState::jitc_prefix_sum(VarType vt, bool exclusive,
                                      const void *in, uint32_t size,
                                      void *out) {
    if (size == 0)
        return;
    if (vt == VarType::Int32)
        vt = VarType::UInt32;

    const uint32_t isize = type_size[(int) vt];

    // CUDA specific
    const Device &device = state.devices[this->device];
    scoped_set_context guard(this->context);

    if (size == 1) {
        if (exclusive) {
            cuda_check(cuMemsetD8Async((CUdeviceptr) out, 0, isize, this->stream));
        } else {
            if (in != out)
                cuda_check(cuMemcpyAsync((CUdeviceptr) out,
                                         (CUdeviceptr) in, isize,
                                         this->stream));
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
            thread_count, shared_size, this->stream, args, nullptr, size);
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
                        block_count_init, thread_count_init, 0, this->stream,
                        args, nullptr, scratch_items);

        scratch += 32; // move beyond padding area
        void *args_2[] = { &in, &out, &size, &scratch };
        jitc_submit_gpu(KernelType::Other, kernel, block_count,
                        thread_count, shared_size, this->stream, args_2,
                        nullptr, scratch_items);
        scratch -= 32;

        jitc_free(scratch);
    }
}

uint32_t CUDAThreadState::jitc_compress(const uint8_t *in, uint32_t size,
                                        uint32_t *out) {
    if (size == 0)
        return 0;

    // CUDA specific
    const Device &device = state.devices[this->device];
    scoped_set_context guard(this->context);

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
                                       this->stream));

        void *args[] = { &in, &out, &size, &count_out };
        jitc_submit_gpu(
            KernelType::Other, jitc_cuda_compress_small[device.id], 1,
            thread_count, shared_size, this->stream, args, nullptr, size);
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
                        block_count_init, thread_count_init, 0, this->stream,
                        args, nullptr, scratch_items);

        if (trailer > 0)
            cuda_check(cuMemsetD8Async((CUdeviceptr) (in + size), 0, trailer,
                                       this->stream));

        scratch += 32; // move beyond padding area
        void *args_2[] = { &in, &out, &scratch, &count_out };
        jitc_submit_gpu(KernelType::Other,
                        jitc_cuda_compress_large[device.id], block_count,
                        thread_count, shared_size, this->stream, args_2,
                        nullptr, scratch_items);
        scratch -= 32;

        jitc_free(scratch);
    }
    jitc_sync_thread();
    uint32_t count_out_v = *count_out;
    jitc_free(count_out);
    return count_out_v;
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

uint32_t CUDAThreadState::jitc_mkperm(const uint32_t *ptr, uint32_t size,
                                      uint32_t bucket_count, uint32_t *perm,
                                      uint32_t *offsets) {
    if (size == 0)
        return 0;
    else if (unlikely(bucket_count == 0))
        jitc_fail("jit_mkperm(): bucket_count cannot be zero!");

    // CUDA specific
    scoped_set_context guard(this->context);
    const Device &device = state.devices[this->device];

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
                                   this->stream));
    }

    if (initialize_buckets)
        cuda_check(cuMemsetD8Async((CUdeviceptr) buckets_1, 0,
                                   bucket_size_all, this->stream));

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
                    thread_count, shared_size, this->stream, args_1, nullptr,
                    size);

    // Phase 2: exclusive prefix sum over transposed buckets
    if (needs_transpose)
        cuda_transpose(this, buckets_1, buckets_2,
                       bucket_size_all / bucket_size_1, bucket_count);

    this->jitc_prefix_sum(VarType::UInt32, true, buckets_2,
              bucket_size_all / sizeof(uint32_t), buckets_2);

    if (needs_transpose)
        cuda_transpose(this, buckets_2, buckets_1, bucket_count,
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
                        this->stream, args_3, nullptr, size);

        cuda_check(cuMemcpyAsync((CUdeviceptr) (offsets + 4 * size_t(bucket_count)),
                                 (CUdeviceptr) counter, sizeof(uint32_t),
                                 this->stream));

        cuda_check(cuEventRecord(this->event, this->stream));
    }

    // Phase 4: write out permutation based on bucket counts
    void *args_4[] = { &ptr, &buckets_1, &perm, &size, &size_per_block,
                       &bucket_count };

    jitc_submit_gpu(KernelType::CallReduce, phase_4, block_count,
                    thread_count, shared_size, this->stream, args_4, nullptr,
                    size);

    if (likely(offsets)) {
        unlock_guard guard_2(state.lock);
        cuda_check(cuEventSynchronize(this->event));
    }

    jitc_free(buckets_1);
    if (needs_transpose)
        jitc_free(buckets_2);
    jitc_free(counter);

    return offsets ? offsets[4 * bucket_count] : 0u;
}

void CUDAThreadState::jitc_memcpy(void *dst, const void *src, size_t size) {
    scoped_set_context guard_2(this->context);
    cuda_check(cuMemcpy((CUdeviceptr) dst, (CUdeviceptr) src, size));
}
