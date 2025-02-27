/*
    src/api.cpp -- C -> C++ API locking wrappers

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "var.h"
#include "eval.h"
#include "log.h"
#include "util.h"
#include "registry.h"
#include "llvm.h"
#include "cuda_tex.h"
#include "op.h"
#include "call.h"
#include "loop.h"
#include "cond.h"
#include "profile.h"
#include "array.h"
#include "record_ts.h"
#include <thread>
#include <condition_variable>
#include <drjit-core/half.h>
#include <drjit-core/texture.h>

#if defined(DRJIT_ENABLE_OPTIX)
#include <drjit-core/optix.h>
#include "optix.h"
#endif

#include <nanothread/nanothread.h>

void jit_init(uint32_t backends) {
    lock_guard guard(state.lock);
    jitc_init(backends);
}

void jit_init_async(uint32_t backends) {
    /// Probably overkill for a simple wait flag..
    struct Sync {
        bool flag = false;
        std::mutex lock;
        std::condition_variable cv;
    };

    std::shared_ptr<Sync> sync = std::make_shared<Sync>();
    std::unique_lock<std::mutex> guard(sync->lock);

    std::thread([backends, sync]() {
        lock_guard guard2(state.lock);
        {
            std::unique_lock<std::mutex> guard3(sync->lock);
            sync->flag = true;
            sync->cv.notify_one();
        }
        jitc_init(backends);
    }).detach();

    while (!sync->flag)
        sync->cv.wait(guard);
}

int jit_has_backend(JitBackend backend) {
    lock_guard guard(state.lock);

    bool result;
    switch (backend) {
        case JitBackend::LLVM:
            result = state.backends & (uint32_t) JitBackend::LLVM;
            break;

        case JitBackend::CUDA:
            result = (state.backends & (uint32_t) JitBackend::CUDA)
                && !state.devices.empty();
            break;

        default:
            jitc_raise("jit_has_backend(): invalid input!");
    }

    return (int) result;
}

void jit_shutdown(int light) {
    lock_guard guard(state.lock);
    jitc_shutdown(light);
}

uint32_t jit_scope(JitBackend backend) {
    lock_guard guard(state.lock);
    return thread_state(backend)->scope;
}

void jit_set_scope(JitBackend backend, uint32_t scope) {
    lock_guard guard(state.lock);
    jitc_trace("jit_set_scope(%u)", scope);
    thread_state(backend)->scope = scope;
}

uint32_t jit_new_scope(JitBackend backend) {
    lock_guard guard(state.lock);
    return jitc_new_scope(backend);
}

void jit_set_log_level_stderr(LogLevel level) {
    /// Allow changing this variable without acquiring a lock
    state.log_level_stderr = level;
}

LogLevel jit_log_level_stderr() {
    /// Allow reading this variable without acquiring a lock
    return state.log_level_stderr;
}

void jit_set_log_level_callback(LogLevel level, LogCallback callback) {
    lock_guard guard(state.lock);
    state.log_level_callback = callback ? level : Disable;
    state.log_callback = callback;
}

LogLevel jit_log_level_callback() {
    lock_guard guard(state.lock);
    return state.log_level_callback;
}

void jit_log(LogLevel level, const char* fmt, ...) {
    lock_guard guard(state.lock);
    va_list args;
    va_start(args, fmt);
    jitc_vlog(level, fmt, args);
    va_end(args);
}

void jit_raise(const char* fmt, ...) {
    lock_guard guard(state.lock);
    va_list args;
    va_start(args, fmt);
    jitc_vraise(fmt, args);
    // va_end(args); (dead code)
}

void jit_fail(const char* fmt, ...) noexcept {
    lock_guard guard(state.lock);
    va_list args;
    va_start(args, fmt);
    jitc_vfail(fmt, args);
    // va_end(args); (dead code)
}

void jit_set_flags(uint32_t flags) {
    jitc_set_flags(flags);
}

uint32_t jit_flags() {
    return jitc_flags();
}

void jit_set_flag(JitFlag flag, int enable) {
    jitc_set_flag(flag, enable);
}

int jit_flag(JitFlag flag) {
    return jitc_flag(flag);
}

uint32_t jit_record_checkpoint(JitBackend backend) {
    uint32_t result = (uint32_t) thread_state(backend)->side_effects_symbolic.size();
    if (jit_flag(JitFlag::SymbolicScope))
        result |= 0x80000000u;
    return result;
}

uint32_t jit_record_begin(JitBackend backend, const char *name) {
    ThreadState *ts = thread_state(backend);
    std::vector<std::string> &stack = ts->record_stack;

    // Potentially signal failure to limit recursion depth
    if (name && std::count(stack.begin(), stack.end(), name) > 1)
        return uint32_t(-1);
    stack.push_back(name ? name : std::string());

    if (name)
        jitc_log(Debug, "jit_record_begin(\"%s\")", name);
    else
        jitc_log(Debug, "jit_record_begin()");

    uint32_t result = (uint32_t) ts->side_effects_symbolic.size();
    if (jit_flag(JitFlag::SymbolicScope))
        result |= 0x80000000u;
    jit_set_flag(JitFlag::SymbolicScope, true);

    return result;
}

void jit_record_end(JitBackend backend, uint32_t value, int cleanup) {
    jitc_log(Debug, "jit_record_end()");

    ThreadState *ts = thread_state(backend);
    std::vector<std::string> &stack = ts->record_stack;

    if (unlikely(stack.empty()))
        jitc_fail("jit_record_end(): stack underflow!");

    stack.pop_back();

    // Set recording flag to previous value
    jit_set_flag(JitFlag::SymbolicScope, (value & 0x80000000u) != 0);
    value &= 0x7fffffff;

    if (cleanup) {
        lock_guard guard(state.lock);
        std::vector<uint32_t> &se = ts->side_effects_symbolic;
        if (value > se.size())
            jitc_raise("jit_record_end(): position lies beyond the end of the queue!");

        while (value < se.size()) {
            uint32_t index = se.back();
            se.pop_back();
            jitc_log(Debug, "jit_record_end(): deleting side effect r%u", index);
            jitc_var_dec_ref(index);
        }
    }
}

void* jit_cuda_stream() {
    lock_guard guard(state.lock);
    return jitc_cuda_stream();
}

void* jit_cuda_context() {
    lock_guard guard(state.lock);
    return jitc_cuda_context();
}

void jit_cuda_push_context(void* ctx) {
    lock_guard guard(state.lock);
    jitc_cuda_push_context(ctx);
}

void* jit_cuda_pop_context() {
    lock_guard guard(state.lock);
    return jitc_cuda_pop_context();
}

int jit_cuda_device_count() {
    lock_guard guard(state.lock);
    return (int) state.devices.size();
}

void jit_cuda_set_device(int device) {
    lock_guard guard(state.lock);
    jitc_cuda_set_device(device);
}

int jit_cuda_device() {
    lock_guard guard(state.lock);
    return thread_state(JitBackend::CUDA)->device;
}

int jit_cuda_device_raw() {
    lock_guard guard(state.lock);
    return state.devices[thread_state(JitBackend::CUDA)->device].id;
}

int jit_cuda_compute_capability() {
    lock_guard guard(state.lock);
    return state.devices[thread_state(JitBackend::CUDA)->device].compute_capability;
}

void jit_cuda_set_target(uint32_t ptx_version, uint32_t compute_capability) {
    lock_guard guard(state.lock);
    ThreadState *ts = thread_state(JitBackend::CUDA);
    ts->ptx_version = ptx_version;
    ts->compute_capability = compute_capability;
}

void *jit_cuda_lookup(const char *name) {
    lock_guard guard(state.lock);
    return jitc_cuda_lookup(name);
}

void jit_cuda_sync_stream(uintptr_t stream) {
    // Only using thread-local state so no shared JIT state synchronization needed
    return jitc_cuda_sync_stream(stream);
}

void jit_llvm_set_thread_count(uint32_t size) {
    pool_set_size(nullptr, size);
}

void jit_llvm_set_block_size(uint32_t size) {
    if ((size & (size - 1)) != 0 || size < jitc_llvm_vector_width)
        jit_raise("jit_llvm_set_block_size(): value must be a power of two and "
                  "bigger than the packet size (%u)!", jitc_llvm_vector_width);
    jitc_llvm_block_size = size;
}

uint32_t jit_llvm_block_size() {
    return jitc_llvm_block_size;
}

uint32_t jit_llvm_thread_count() {
    return pool_size(nullptr);
}

void jit_llvm_set_target(const char *target_cpu,
                         const char *target_features,
                         uint32_t vector_width) {
    lock_guard guard(state.lock);
    jitc_llvm_set_target(target_cpu, target_features, vector_width);
}

const char *jit_llvm_target_cpu() {
    lock_guard guard(state.lock);
    return jitc_llvm_target_cpu;
}

const char *jit_llvm_target_features() {
    lock_guard guard(state.lock);
    return jitc_llvm_target_features;
}

void jit_llvm_version(int *major, int *minor, int *patch) {
    lock_guard guard(state.lock);
    if (major)
        *major = jitc_llvm_version_major;
    if (minor)
        *minor = jitc_llvm_version_minor;
    if (patch)
        *patch = jitc_llvm_version_patch;
}

uint32_t jit_llvm_vector_width() {
    return jitc_llvm_vector_width;
}

void jit_sync_thread() {
    lock_guard guard(state.lock);
    jitc_sync_thread();
}

void jit_sync_device() {
    lock_guard guard(state.lock);
    jitc_sync_device();
}

void jit_sync_all_devices() {
    lock_guard guard(state.lock);
    jitc_sync_all_devices();
}

void jit_flush_kernel_cache() {
    lock_guard guard(state.lock);
    jitc_flush_kernel_cache();
}

void *jit_malloc(AllocType type, size_t size) {
    lock_guard guard(state.lock);
    return jitc_malloc(type, size);
}

void jit_free(void *ptr) {
    lock_guard guard(state.lock);
    jitc_free(ptr);
}

void jit_flush_malloc_cache() {
    lock_guard guard(state.lock);
    jitc_flush_malloc_cache(false);
}

void jit_malloc_clear_statistics() {
    lock_guard guard(state.lock);
    jitc_malloc_clear_statistics();
}

enum AllocType jit_malloc_type(void *ptr) {
    lock_guard guard(state.lock);
    return jitc_malloc_type(ptr);
}

int jit_malloc_device(void *ptr) {
    lock_guard guard(state.lock);
    return jitc_malloc_device(ptr);
}

void *jit_malloc_migrate(void *ptr, AllocType type, int move) {
    lock_guard guard(state.lock);
    return jitc_malloc_migrate(ptr, type, move);
}

enum AllocType jit_var_alloc_type(uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_alloc_type(index);
}

int jit_var_device(uint32_t index) {
    if (index == 0)
        return -1;
    lock_guard guard(state.lock);
    return jitc_var_device(index);
}

uint32_t jit_var_literal(JitBackend backend, VarType type, const void *value,
                             size_t size, int eval) {
    lock_guard guard(state.lock);
    return jitc_var_literal(backend, type, value, size, eval);
}

uint32_t jit_var_bool(JitBackend backend, bool value) {
    Variable v;
    memcpy(&v.literal, &value, sizeof(bool));
    v.kind = (uint32_t) VarKind::Literal;
    v.type = (uint32_t) VarType::Bool;
    v.size = 1;
    v.backend = (uint32_t) backend;
    lock_guard guard(state.lock);
    return jitc_var_new(v);
}

uint32_t jit_var_u32(JitBackend backend, uint32_t value) {
    Variable v;
    memcpy(&v.literal, &value, sizeof(uint32_t));
    v.kind = (uint32_t) VarKind::Literal;
    v.type = (uint32_t) VarType::UInt32;
    v.size = 1;
    v.backend = (uint32_t) backend;
    lock_guard guard(state.lock);
    return jitc_var_new(v);
}

uint32_t jit_var_i32(JitBackend backend, int32_t value) {
    Variable v;
    memcpy(&v.literal, &value, sizeof(int32_t));
    v.kind = (uint32_t) VarKind::Literal;
    v.type = (uint32_t) VarType::Int32;
    v.size = 1;
    v.backend = (uint32_t) backend;
    lock_guard guard(state.lock);
    return jitc_var_new(v);
}

uint32_t jit_var_u64(JitBackend backend, uint64_t value) {
    Variable v;
    memcpy(&v.literal, &value, sizeof(uint64_t));
    v.kind = (uint32_t) VarKind::Literal;
    v.type = (uint32_t) VarType::UInt64;
    v.size = 1;
    v.backend = (uint32_t) backend;
    lock_guard guard(state.lock);
    return jitc_var_new(v);
}

uint32_t jit_var_i64(JitBackend backend, int64_t value) {
    Variable v;
    memcpy(&v.literal, &value, sizeof(int64_t));
    v.kind = (uint32_t) VarKind::Literal;
    v.type = (uint32_t) VarType::Int64;
    v.size = 1;
    v.backend = (uint32_t) backend;
    lock_guard guard(state.lock);
    return jitc_var_new(v);
}

uint32_t jit_var_f16(JitBackend backend, drjit::half value) {
    Variable v;
    memcpy(&v.literal, &value, sizeof(drjit::half));
    v.kind = (uint32_t) VarKind::Literal;
    v.type = (uint32_t) VarType::Float16;
    v.size = 1;
    v.backend = (uint32_t) backend;
    lock_guard guard(state.lock);
    return jitc_var_new(v);
}

uint32_t jit_var_f32(JitBackend backend, float value) {
    Variable v;
    memcpy(&v.literal, &value, sizeof(float));
    v.kind = (uint32_t) VarKind::Literal;
    v.type = (uint32_t) VarType::Float32;
    v.size = 1;
    v.backend = (uint32_t) backend;
    lock_guard guard(state.lock);
    return jitc_var_new(v);
}

uint32_t jit_var_f64(JitBackend backend, double value) {
    Variable v;
    memcpy(&v.literal, &value, sizeof(double));
    v.kind = (uint32_t) VarKind::Literal;
    v.type = (uint32_t) VarType::Float64;
    v.size = 1;
    v.backend = (uint32_t) backend;
    lock_guard guard(state.lock);
    return jitc_var_new(v);
}

uint32_t jit_var_class(JitBackend backend, void *ptr) {
    uint32_t value = jit_registry_id(ptr);

    ThreadState *ts = thread_state(backend);
    if (value && ts->call_self_value == value) {
        jit_var_inc_ref(ts->call_self_index);
        return ts->call_self_index;
    }

    return jit_var_u32(backend, value);
}

uint32_t jit_var_undefined(JitBackend backend, VarType type, size_t size) {
    lock_guard guard(state.lock);
    return jitc_var_undefined(backend, type, size);
}

uint32_t jit_var_counter(JitBackend backend, size_t size) {
    lock_guard guard(state.lock);
    return jitc_var_counter(backend, size, true);
}

uint32_t jit_var_op(JitOp op, const uint32_t *dep) {
    lock_guard guard(state.lock);
    return jitc_var_op(op, dep);
}

uint32_t jit_var_gather(uint32_t source, uint32_t index, uint32_t mask) {
    lock_guard guard(state.lock);
    return jitc_var_gather(source, index, mask);
}

void jit_var_gather_packet(size_t n, uint32_t source, uint32_t index, uint32_t mask, uint32_t *out) {
    lock_guard guard(state.lock);
    jitc_var_gather_packet(n, source, index, mask, out);
}

uint32_t jit_var_scatter(uint32_t target, uint32_t value,
                         uint32_t index, uint32_t mask,
                         ReduceOp op, ReduceMode mode) {
    lock_guard guard(state.lock);
    return jitc_var_scatter(target, value, index, mask, op, mode);
}

uint32_t jit_var_scatter_packet(size_t n, uint32_t target,
                                const uint32_t *values, uint32_t index,
                                uint32_t mask, ReduceOp op, ReduceMode mode) {
    lock_guard guard(state.lock);
    return jitc_var_scatter_packet(n, target, values, index, mask, op, mode);
}

void jit_var_scatter_add_kahan(uint32_t *target_1, uint32_t *target_2,
                                  uint32_t value, uint32_t index, uint32_t mask) {
    lock_guard guard(state.lock);
    jitc_var_scatter_add_kahan(target_1, target_2, value, index, mask);
}

uint32_t jit_var_scatter_inc(uint32_t *target, uint32_t index, uint32_t mask) {
    lock_guard guard(state.lock);
    return jitc_var_scatter_inc(target, index, mask);
}

uint32_t jit_var_pointer(JitBackend backend, const void *value,
                             uint32_t dep, int write) {
    lock_guard guard(state.lock);
    return jitc_var_pointer(backend, value, dep, write);
}

uint32_t jit_var_call_input(uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_call_input(index);
}

uint32_t jit_var_inc_ref_impl(uint32_t index) noexcept {
    if (index == 0)
        return 0;

    lock_guard guard(state.lock);
    jitc_var_inc_ref(index);
    return index;
}

void jit_var_dec_ref_impl(uint32_t index) noexcept {
    if (index == 0)
        return;

    lock_guard guard(state.lock);
    jitc_var_dec_ref(index);
}

uint32_t jit_var_ref(uint32_t index) {
    if (index == 0)
        return 0;

    lock_guard guard(state.lock);
    return jitc_var(index)->ref_count;
}

uint32_t jit_var_data(uint32_t index, void **ptr_out) {
    lock_guard guard(state.lock);
    return jitc_var_data(index, true, ptr_out);
}

size_t jit_var_size(uint32_t index) {
    if (index == 0)
        return 0;

    lock_guard guard(state.lock);
    return (size_t) jitc_var(index)->size;
}

VarState jit_var_state(uint32_t index) {
    if (index == 0)
        return VarState::Invalid;

    lock_guard guard(state.lock);
    return jitc_var_state(index);
}

int jit_var_is_zero_literal(uint32_t index) {
    if (index == 0)
        return 0;

    lock_guard guard(state.lock);
    const Variable *v = jitc_var(index);
    return v->is_literal() && v->literal == 0;
}

int jit_var_is_finite_literal(uint32_t index) {
    if (index == 0)
        return 0;

    lock_guard guard(state.lock);
    const Variable *v = jitc_var(index);
    if (!v->is_literal())
        return 0;

    switch ((VarType) v->type) {
        case VarType::Float16: {
                drjit::half h;
                memcpy((void*)&h, &v->literal, sizeof(drjit::half));
                return (int) std::isnormal((float)h);
        }

        case VarType::Float32: {
                float f;
                memcpy(&f, &v->literal, sizeof(float));
                return (int) std::isnormal(f);
            }

        case VarType::Float64: {
                double d;
                memcpy(&d, &v->literal, sizeof(double));
                return (int) std::isnormal(d);
            }

        default:
            return 1;
    }
}

int jit_var_is_unaligned(uint32_t index) {
    if (index == 0)
        return 0;

    lock_guard guard(state.lock);
    Variable *var = jitc_var(index);

    return (VarKind) var->kind == VarKind::Evaluated && var->unaligned;
}

uint32_t jit_var_resize(uint32_t index, size_t size) {
    lock_guard guard(state.lock);
    return jitc_var_resize(index, size);
}

VarType jit_var_type(uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_type(index);
}

const char *jit_var_kind_name(uint32_t index) {
    lock_guard guard(state.lock);
    return var_kind_name[jitc_var(index)->kind];
}

int jit_var_is_dirty(uint32_t index) {
    if (index == 0)
        return 0;

    lock_guard guard(state.lock);
    return jitc_var(index)->is_dirty();
}

const char *jit_var_label(uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_label(index);
}

uint32_t jit_var_set_label(uint32_t index, size_t argc, ...) {
    if (unlikely(index == 0))
        return 0;

    const char *label = nullptr;

    // First, turn the variable-length argument list into a usable label
    va_list ap;
    va_start(ap, argc);

    StringBuffer buf;
    if (argc == 1) {
        label = va_arg(ap, const char *);
    } else if (argc > 1) {
        for (size_t i = 0; i < argc; ++i) {
            const char *s = va_arg(ap, const char *);
            bool isnum = s[0] >= '0' || s[1] <= '9';

            if (isnum) {
                buffer.put('[');
                buffer.put(s, strlen(s));
                buffer.put(']');
            } else {
                if (i > 0)
                    buffer.put('.');
                buffer.put(s, strlen(s));
            }
        }
        label = buffer.get();
    }
    va_end(ap);

    lock_guard guard(state.lock);

    Variable *v = jitc_var(index);

    if (v->ref_count == 1) {
        jitc_var_set_label(index, label);
        jitc_var_inc_ref(index, v);
        return index;
    } else {
        uint32_t result = jitc_var_copy(index);
        jitc_var_set_label(result, label);
        return result;
    }
}

void jit_var_set_callback(uint32_t index,
                          void (*callback)(uint32_t, int, void *),
                          void *data) {
    lock_guard guard(state.lock);
    jitc_var_set_callback(index, callback, data, false);
}

uint32_t jit_var_mem_map(JitBackend backend, VarType type, void *ptr, size_t size, int free) {
    lock_guard guard(state.lock);
    return jitc_var_mem_map(backend, type, ptr, size, free);
}

uint32_t jit_var_mem_copy(JitBackend backend, AllocType atype, VarType vtype,
                          const void *value, size_t size) {
    lock_guard guard(state.lock);
    return jitc_var_mem_copy(backend, atype, vtype, value, size);
}

uint32_t jit_var_copy(uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_copy(index);
}

uint32_t jit_var_migrate(uint32_t index, AllocType type) {
    lock_guard guard(state.lock);
    return jitc_var_migrate(index, type);
}

void jit_var_mark_side_effect(uint32_t index) {
    lock_guard guard(state.lock);
    jitc_var_mark_side_effect(index);
}

uint32_t jit_var_mask_peek(JitBackend backend) {
    lock_guard guard(state.lock);
    return jitc_var_mask_peek(backend);
}

uint32_t jit_var_mask_apply(uint32_t index, uint32_t size) {
    lock_guard guard(state.lock);
    return jitc_var_mask_apply(index, size);
}

void jit_var_mask_push(JitBackend backend, uint32_t index) {
    lock_guard guard(state.lock);
    jitc_var_mask_push(backend, index);
}

void jit_var_mask_pop(JitBackend backend) {
    lock_guard guard(state.lock);
    jitc_var_mask_pop(backend);
}

uint32_t jit_var_mask_default(JitBackend backend, size_t size) {
    lock_guard guard(state.lock);
    return jitc_var_mask_default(backend, size);
}

int jit_var_any(uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_any(index);
}

int jit_var_all(uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_all(index);
}

uint32_t jit_var_any_async(JitBackend backend, uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_any_async(backend, index);
}

uint32_t jit_var_all_async(JitBackend backend, uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_all_async(backend, index);
}

uint32_t jit_var_reduce(JitBackend backend, VarType vt, ReduceOp op,
                        uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_reduce(backend, vt, op, index);
}

uint32_t jit_var_reduce_dot(uint32_t index_1,
                            uint32_t index_2) {
    lock_guard guard(state.lock);
    return jitc_var_reduce_dot(index_1, index_2);
}

uint32_t jit_var_block_prefix_reduce(ReduceOp op, uint32_t index,
                                     uint32_t block_size, int exclusive,
                                     int reverse) {
    lock_guard guard(state.lock);
    return jitc_var_block_prefix_reduce(op, index, block_size, exclusive != 0,
                                        reverse != 0);
}

const char *jit_var_whos() {
    lock_guard guard(state.lock);
    return jitc_var_whos();
}

const char *jit_var_graphviz() {
    lock_guard guard(state.lock);
    return jitc_var_graphviz();
}

const char *jit_var_str(uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_str(index);
}

void jit_var_read(uint32_t index, size_t offset, void *dst) {
    lock_guard guard(state.lock);
    jitc_var_read(index, offset, dst);
}

uint32_t jit_var_write(uint32_t index, size_t offset, const void *src) {
    lock_guard guard(state.lock);
    return jitc_var_write(index, offset, src);
}

void jit_eval() {
    lock_guard guard(state.lock);
    jitc_eval(thread_state_cuda);
    jitc_eval(thread_state_llvm);
}

int jit_var_eval(uint32_t index) {
    if (index == 0)
        return 0;
    lock_guard guard(state.lock);
    return jitc_var_eval(index);
}

int jit_var_schedule(uint32_t index) {
    if (index == 0)
        return 0;
    lock_guard guard(state.lock);
    return jitc_var_schedule(index);
}

uint32_t jit_var_schedule_force(uint32_t index, int *rv) {
    lock_guard guard(state.lock);
    return jitc_var_schedule_force(index, rv);
}

void jit_prefix_push(JitBackend backend, const char *value) {
    lock_guard guard(state.lock);
    jitc_prefix_push(backend, value);
}

void jit_prefix_pop(JitBackend backend) {
    lock_guard guard(state.lock);
    jitc_prefix_pop(backend);
}

const char *jit_prefix(JitBackend backend) {
    return thread_state(backend)->prefix;
}

void jit_memset_async(JitBackend backend, void *ptr, uint32_t size, uint32_t isize,
                      const void *src) {
    lock_guard guard(state.lock);
    jitc_memset_async(backend, ptr, size, isize, src);
}

void jit_memcpy(JitBackend backend, void *dst, const void *src, size_t size) {
    lock_guard guard(state.lock);
    jitc_memcpy(backend, dst, src, size);
}

void jit_memcpy_async(JitBackend backend, void *dst, const void *src, size_t size) {
    lock_guard guard(state.lock);
    jitc_memcpy_async(backend, dst, src, size);
}

void jit_reduce(JitBackend backend, VarType type, ReduceOp op,
                uint32_t size, const void *in, void *out) {
    lock_guard guard(state.lock);
    jitc_reduce(backend, type, op, size, in, out);
}

void jit_block_reduce(JitBackend backend, VarType type, ReduceOp op,
                      uint32_t size, uint32_t block_size, const void *in,
                      void *out) {
    lock_guard guard(state.lock);
    jitc_block_reduce(backend, type, op, size, block_size, in, out);
}

void jit_block_prefix_reduce(JitBackend backend, VarType type, ReduceOp op,
                             uint32_t block_size, uint32_t size, int exclusive,
                             int reverse, const void *in, void *out) {
    lock_guard guard(state.lock);
    jitc_block_prefix_reduce(backend, type, op, block_size, size,
                             exclusive != 0, reverse != 0, in, out);
}

uint32_t jit_compress(JitBackend backend, const uint8_t *in, uint32_t size, uint32_t *out) {
    lock_guard guard(state.lock);
    return jitc_compress(backend, in, size, out);
}

uint32_t jit_mkperm(JitBackend backend, const uint32_t *values, uint32_t size,
                    uint32_t bucket_count, uint32_t *perm, uint32_t *offsets) {
    lock_guard guard(state.lock);
    return jitc_mkperm(backend, values, size, bucket_count, perm, offsets);
}

uint32_t jit_registry_put(const char *variant, const char *domain, void *ptr) {
    lock_guard guard(state.lock);
    return jitc_registry_put(variant, domain, ptr);
}

void jit_registry_remove(const void *ptr) {
    lock_guard guard(state.lock);
    jitc_registry_remove(ptr);
}

uint32_t jit_registry_id(const void *ptr) {
    lock_guard guard(state.lock);
    return jitc_registry_id(ptr);
}

uint32_t jit_registry_id_bound(const char *variant, const char *domain) {
    lock_guard guard(state.lock);
    return jitc_registry_id_bound(variant, domain);
}

void jit_registry_get_pointers(const char *variant, const char *domain_name,
                               void **dest) {
    lock_guard guard(state.lock);
    return jitc_registry_get_pointers(variant, domain_name, dest);
}

void *jit_registry_ptr(const char *variant, const char *domain, uint32_t id) {
    lock_guard guard(state.lock);
    return jitc_registry_ptr(variant, domain, id);
}

void *jit_registry_peek(const char *variant, const char *domain) {
    lock_guard guard(state.lock);
    return jitc_registry_peek(variant, domain);
}

void jit_registry_clear() {
    lock_guard guard(state.lock);
    jitc_registry_clear();
}

void jit_var_set_self(JitBackend backend, uint32_t value, uint32_t index) {
    lock_guard guard(state.lock);
    jitc_var_set_self(backend, value, index);
}

void jit_var_self(JitBackend backend, uint32_t *value, uint32_t *index) {
    lock_guard guard(state.lock);
    jitc_var_self(backend, value, index);
}

void jit_var_call(const char *name, int symbolic, uint32_t self, uint32_t mask,
                  uint32_t n_inst, uint32_t max_inst_id,
                  const uint32_t *inst_id, uint32_t n_in,
                  const uint32_t *in, uint32_t n_out_nested,
                  const uint32_t *out_nested, const uint32_t *se_offset,
                  uint32_t *out) {
    lock_guard guard(state.lock);
    jitc_var_call(name, (bool) symbolic, self, mask, n_inst, max_inst_id,
                  inst_id, n_in, in, n_out_nested, out_nested, se_offset, out);
}

void jit_aggregate(JitBackend backend, void *dst, AggregationEntry *agg,
                   uint32_t size) {
    lock_guard guard(state.lock);
    return jitc_aggregate(backend, dst, agg, size);
}

struct CallBucket *jit_var_call_reduce(JitBackend backend, const char *variant,
                                       const char *domain, uint32_t index,
                                       uint32_t *bucket_count_inout) {
    lock_guard guard(state.lock);
    return jitc_var_call_reduce(backend, variant, domain, index,
                                bucket_count_inout);
}

void jit_kernel_history_clear() {
    lock_guard guard(state.lock);
    state.kernel_history.clear();
}

struct KernelHistoryEntry *jit_kernel_history() {
    lock_guard guard(state.lock);
    jitc_sync_thread();
    return state.kernel_history.get();
}

#if defined(DRJIT_ENABLE_OPTIX)
OptixDeviceContext jit_optix_context() {
    lock_guard guard(state.lock);
    return jitc_optix_context();
}

void *jit_optix_lookup(const char *name) {
    lock_guard guard(state.lock);
    return jitc_optix_lookup(name);
}

uint32_t jit_optix_configure_pipeline(const OptixPipelineCompileOptions *pco,
                                      OptixModule module,
                                      const OptixProgramGroup *pg,
                                      uint32_t pg_count) {
    lock_guard guard(state.lock);
    return jitc_optix_configure_pipeline(pco, module, pg, pg_count);
}

uint32_t jit_optix_configure_sbt(const OptixShaderBindingTable *sbt, uint32_t pipeline) {
    lock_guard guard(state.lock);
    return jitc_optix_configure_sbt(sbt, pipeline);
}

void jit_optix_update_sbt(uint32_t index, const OptixShaderBindingTable *sbt) {
    lock_guard guard(state.lock);
    jitc_optix_update_sbt(index, sbt);
}

void jit_optix_ray_trace(uint32_t nargs, uint32_t *args, uint32_t mask,
                         uint32_t pipeline, uint32_t sbt) {
    lock_guard guard(state.lock);
    jitc_optix_ray_trace(nargs, args, mask, pipeline, sbt);
}

#endif

void jit_llvm_ray_trace(uint32_t func, uint32_t scene, int shadow_ray,
                        const uint32_t *in, uint32_t *out) {
    lock_guard guard(state.lock);
    jitc_llvm_ray_trace(func, scene, shadow_ray, in, out);
}

void *jit_cuda_tex_create(size_t ndim, const size_t *shape, size_t n_channels,
                          int format, int filter_mode, int wrap_mode) {
    lock_guard guard(state.lock);
    return jitc_cuda_tex_create(ndim, shape, n_channels, format, filter_mode, wrap_mode);
}

void jit_cuda_tex_get_shape(size_t ndim, const void *texture_handle,
                            size_t *shape) {
    lock_guard guard(state.lock);
    jitc_cuda_tex_get_shape(ndim, texture_handle, shape);
}

void jit_cuda_tex_get_indices(const void *texture_handle, uint32_t *indices) {
    lock_guard guard(state.lock);
    jitc_cuda_tex_get_indices(texture_handle, indices);
}

void jit_cuda_tex_memcpy_d2t(size_t ndim, const size_t *shape,
                             const void *src_ptr, void *dst_texture) {
    lock_guard guard(state.lock);
    jitc_cuda_tex_memcpy_d2t(ndim, shape, src_ptr, dst_texture);
}

void jit_cuda_tex_memcpy_t2d(size_t ndim, const size_t *shape,
                             const void *src_texture, void *dst_ptr) {
    lock_guard guard(state.lock);
    jitc_cuda_tex_memcpy_t2d(ndim, shape, src_texture, dst_ptr);
}

void jit_cuda_tex_lookup(size_t ndim, const void *texture_handle,
                         const uint32_t *pos, uint32_t active, uint32_t *out) {
    lock_guard guard(state.lock);
    jitc_cuda_tex_lookup(ndim, texture_handle, pos, active, out);
}

void jit_cuda_tex_bilerp_fetch(size_t ndim, const void *texture_handle,
                               const uint32_t *pos, uint32_t active,
                               uint32_t *out) {
    lock_guard guard(state.lock);
    jitc_cuda_tex_bilerp_fetch(ndim, texture_handle, pos, active, out);
}

void jit_cuda_tex_destroy(void *texture) {
    lock_guard guard(state.lock);
    jitc_cuda_tex_destroy(texture);
}

uint32_t jit_var_neg(uint32_t a0) {
    lock_guard guard(state.lock);
    return jitc_var_neg(a0);
}

uint32_t jit_var_not(uint32_t a0) {
    lock_guard guard(state.lock);
    return jitc_var_not(a0);
}

uint32_t jit_var_sqrt(uint32_t a0) {
    lock_guard guard(state.lock);
    return jitc_var_sqrt(a0);
}

uint32_t jit_var_abs(uint32_t a0) {
    lock_guard guard(state.lock);
    return jitc_var_abs(a0);
}

uint32_t jit_var_add(uint32_t a0, uint32_t a1) {
    lock_guard guard(state.lock);
    return jitc_var_add(a0, a1);
}

uint32_t jit_var_sub(uint32_t a0, uint32_t a1) {
    lock_guard guard(state.lock);
    return jitc_var_sub(a0, a1);
}

uint32_t jit_var_mul(uint32_t a0, uint32_t a1) {
    lock_guard guard(state.lock);
    return jitc_var_mul(a0, a1);
}

uint32_t jit_var_div(uint32_t a0, uint32_t a1) {
    lock_guard guard(state.lock);
    return jitc_var_div(a0, a1);
}

uint32_t jit_var_mod(uint32_t a0, uint32_t a1) {
    lock_guard guard(state.lock);
    return jitc_var_mod(a0, a1);
}

uint32_t jit_var_mulhi(uint32_t a0, uint32_t a1) {
    lock_guard guard(state.lock);
    return jitc_var_mulhi(a0, a1);
}

uint32_t jit_var_fma(uint32_t a0, uint32_t a1, uint32_t a2) {
    lock_guard guard(state.lock);
    return jitc_var_fma(a0, a1, a2);
}

uint32_t jit_var_min(uint32_t a0, uint32_t a1) {
    lock_guard guard(state.lock);
    return jitc_var_min(a0, a1);
}

uint32_t jit_var_max(uint32_t a0, uint32_t a1) {
    lock_guard guard(state.lock);
    return jitc_var_max(a0, a1);
}

uint32_t jit_var_ceil(uint32_t a0) {
    lock_guard guard(state.lock);
    return jitc_var_ceil(a0);
}

uint32_t jit_var_floor(uint32_t a0) {
    lock_guard guard(state.lock);
    return jitc_var_floor(a0);
}

uint32_t jit_var_round(uint32_t a0) {
    lock_guard guard(state.lock);
    return jitc_var_round(a0);
}

uint32_t jit_var_trunc(uint32_t a0) {
    lock_guard guard(state.lock);
    return jitc_var_trunc(a0);
}

uint32_t jit_var_eq(uint32_t a0, uint32_t a1) {
    lock_guard guard(state.lock);
    return jitc_var_eq(a0, a1);
}

uint32_t jit_var_neq(uint32_t a0, uint32_t a1) {
    lock_guard guard(state.lock);
    return jitc_var_neq(a0, a1);
}

uint32_t jit_var_lt(uint32_t a0, uint32_t a1) {
    lock_guard guard(state.lock);
    return jitc_var_lt(a0, a1);
}

uint32_t jit_var_le(uint32_t a0, uint32_t a1) {
    lock_guard guard(state.lock);
    return jitc_var_le(a0, a1);
}

uint32_t jit_var_gt(uint32_t a0, uint32_t a1) {
    lock_guard guard(state.lock);
    return jitc_var_gt(a0, a1);
}

uint32_t jit_var_ge(uint32_t a0, uint32_t a1) {
    lock_guard guard(state.lock);
    return jitc_var_ge(a0, a1);
}

uint32_t jit_var_select(uint32_t a0, uint32_t a1, uint32_t a2) {
    lock_guard guard(state.lock);
    return jitc_var_select(a0, a1, a2);
}

uint32_t jit_var_popc(uint32_t a0) {
    lock_guard guard(state.lock);
    return jitc_var_popc(a0);
}

uint32_t jit_var_clz(uint32_t a0) {
    lock_guard guard(state.lock);
    return jitc_var_clz(a0);
}

uint32_t jit_var_ctz(uint32_t a0) {
    lock_guard guard(state.lock);
    return jitc_var_ctz(a0);
}

uint32_t jit_var_brev(uint32_t a0) {
    lock_guard guard(state.lock);
    return jitc_var_brev(a0);
}

uint32_t jit_var_and(uint32_t a0, uint32_t a1) {
    lock_guard guard(state.lock);
    return jitc_var_and(a0, a1);
}

uint32_t jit_var_or(uint32_t a0, uint32_t a1) {
    lock_guard guard(state.lock);
    return jitc_var_or(a0, a1);
}

uint32_t jit_var_xor(uint32_t a0, uint32_t a1) {
    lock_guard guard(state.lock);
    return jitc_var_xor(a0, a1);
}

uint32_t jit_var_shl(uint32_t a0, uint32_t a1) {
    lock_guard guard(state.lock);
    return jitc_var_shl(a0, a1);
}

uint32_t jit_var_shr(uint32_t a0, uint32_t a1) {
    lock_guard guard(state.lock);
    return jitc_var_shr(a0, a1);
}

uint32_t jit_var_rcp(uint32_t a0) {
    lock_guard guard(state.lock);
    return jitc_var_rcp(a0);
}

uint32_t jit_var_rsqrt(uint32_t a0) {
    lock_guard guard(state.lock);
    return jitc_var_rsqrt(a0);
}

uint32_t jit_var_sin_intrinsic(uint32_t a0) {
    lock_guard guard(state.lock);
    return jitc_var_sin_intrinsic(a0);
}

uint32_t jit_var_cos_intrinsic(uint32_t a0) {
    lock_guard guard(state.lock);
    return jitc_var_cos_intrinsic(a0);
}

uint32_t jit_var_exp2_intrinsic(uint32_t a0) {
    lock_guard guard(state.lock);
    return jitc_var_exp2_intrinsic(a0);
}

uint32_t jit_var_log2_intrinsic(uint32_t a0) {
    lock_guard guard(state.lock);
    return jitc_var_log2_intrinsic(a0);
}

uint32_t jit_var_cast(uint32_t index, VarType target_type,
                      int reinterpret) {
    lock_guard guard(state.lock);
    return jitc_var_cast(index, target_type, reinterpret);
}

uint32_t jit_var_call_mask(JitBackend backend) {
    lock_guard guard(state.lock);
    return jitc_var_call_mask(backend);
}

size_t jit_type_size(VarType type) noexcept {
    return type_size[(int) type];
}

const char *jit_type_name(VarType type) noexcept {
    return type_name[(int) type];
}

VarInfo jit_set_backend(uint32_t index) noexcept {
    VarInfo info;

    lock_guard guard(state.lock);
    Variable *var = jitc_var(index);
    default_backend = (JitBackend) var->backend;

    info.backend = (JitBackend)var->backend;
    info.type = (VarType)var->type;
    info.state = jitc_var_state(index);
    info.size = var->size;
    info.is_array = var->is_array();
    info.unaligned = var->unaligned;
    if(info.state == VarState::Literal)
        info.literal = var->literal;
    else if (info.state == VarState::Evaluated)
        info.data = var->data;

    return info;
}

uint32_t jit_var_loop_start(const char *name, bool symbolic, size_t n_indices, uint32_t *indices) {
    lock_guard guard(state.lock);
    return jitc_var_loop_start(name, symbolic, n_indices, indices);
}

void jit_var_loop_update_inner_in(uint32_t loop, uint32_t *indices) {
    lock_guard guard(state.lock);
    return jitc_var_loop_update_inner_in(loop, indices);
}

uint32_t jit_var_loop_cond(uint32_t loop, uint32_t active) {
    lock_guard guard(state.lock);
    return jitc_var_loop_cond(loop, active);
}

int jit_var_loop_end(uint32_t loop, uint32_t cond, uint32_t *indices, uint32_t checkpoint) {
    lock_guard guard(state.lock);
    return jitc_var_loop_end(loop, cond, indices, checkpoint);
}

uint32_t jit_var_cond_start(const char *name, bool symbolic, uint32_t cond_t, uint32_t cond_f) {
    lock_guard guard(state.lock);
    return jitc_var_cond_start(name, symbolic, cond_t, cond_f);
}

uint32_t jit_var_cond_append(uint32_t index, const uint32_t *rv, size_t count) {
    lock_guard guard(state.lock);
    return jitc_var_cond_append(index, rv, count);
}

void jit_var_cond_end(uint32_t index, uint32_t *rv_out) {
    lock_guard guard(state.lock);
    jitc_var_cond_end(index, rv_out);
}

void jit_set_source_location(const char *fname, size_t lineno) noexcept {
    jitc_set_source_location(fname, lineno);
}

uint64_t jit_var_stash_ref(uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_stash_ref(index);
}

void jit_var_unstash_ref(uint64_t handle) {
    lock_guard guard(state.lock);
    return jitc_var_unstash_ref(handle);
}

void jit_enqueue_host_func(JitBackend backend, void (*callback)(void *),
                           void *payload) {
    lock_guard guard(state.lock);
    jitc_enqueue_host_func(backend, callback, payload);
}

/// Compress a sparse boolean array into an index array of the active indices
uint32_t jit_var_compress(uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_compress(index);
}

// Shrink a variable after it has been created
uint32_t jit_var_shrink(uint32_t index, size_t size) {
    lock_guard guard(state.lock);
    return jitc_var_shrink(index, size);
}

void jit_profile_mark(const char *message) {
    jitc_profile_mark(message);
}

void jit_profile_range_push(const char *message) {
    jitc_profile_range_push(message);
}

void jit_profile_range_pop() {
    jitc_profile_range_pop();
}

size_t llvm_expand_threshold = 1024 * 1024; // 1M entries

void jit_llvm_set_expand_threshold(size_t size) {
    llvm_expand_threshold = size;
}

size_t jit_llvm_expand_threshold() noexcept {
    return llvm_expand_threshold;
}

uint64_t jit_reduce_identity(VarType vt, ReduceOp op) {
    lock_guard guard(state.lock);
    return jitc_reduce_identity(vt, op);
}

uint32_t jit_var_block_reduce(ReduceOp op, uint32_t index, uint32_t block_size, int symbolic) {
    lock_guard guard(state.lock);
    return jitc_var_block_reduce(op, index, block_size, symbolic);
}

int jit_can_scatter_reduce(JitBackend backend, VarType vt, ReduceOp op) {
    lock_guard guard(state.lock);
    return jitc_can_scatter_reduce(backend, vt, op);
}

uint32_t jit_var_tile(uint32_t index, uint32_t count) {
    lock_guard guard(state.lock);
    return jitc_var_tile(index, count);
}

uint32_t jit_var_repeat(uint32_t index, uint32_t count) {
    lock_guard guard(state.lock);
    return jitc_var_repeat(index, count);
}

uint32_t jit_array_create(JitBackend backend, VarType vt, size_t size,
                          size_t length) {
    lock_guard guard(state.lock);
    return jitc_array_create(backend, vt, size, length);
}

uint32_t jit_array_init(uint32_t target, uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_array_init(target, index);
}

uint32_t jit_array_read(uint32_t source, uint32_t offset, uint32_t mask) {
    lock_guard guard(state.lock);
    return jitc_array_read(source, offset, mask);
}

uint32_t jit_array_write(uint32_t target, uint32_t offset, uint32_t value, uint32_t mask) {
    lock_guard guard(state.lock);
    return jitc_array_write(target, offset, value, mask);
}

size_t jit_array_length(uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_array_length(index);
}

uint32_t jit_var_reverse(uint32_t index) {
    lock_guard guard(state.lock);
    return jitc_var_reverse(index);
}

int jit_leak_warnings() {
    return (int) state.leak_warnings;
}

void jit_set_leak_warnings(int value) {
    state.leak_warnings = (bool) value;
}

void jit_freeze_start(JitBackend backend, const uint32_t *inputs,
                      uint32_t n_inputs) {
    lock_guard guard(state.lock);
    return jitc_freeze_start(backend, inputs, n_inputs);
}

Recording *jit_freeze_stop(JitBackend backend, const uint32_t *outputs,
                           uint32_t n_outputs) {
    lock_guard guard(state.lock);
    return jitc_freeze_stop(backend, outputs, n_outputs);
}

int jit_freeze_pause(JitBackend backend) {
    lock_guard guard(state.lock);
    return jitc_freeze_pause(backend);
}

int jit_freeze_resume(JitBackend backend) {
    lock_guard guard(state.lock);
    return jitc_freeze_resume(backend);
}

void jit_freeze_abort(JitBackend backend) {
    lock_guard guard(state.lock);
    return jitc_freeze_abort(backend);
}

void jit_freeze_replay(Recording *recording, const uint32_t *inputs,
                       uint32_t *outputs) {
    lock_guard guard(state.lock);
    jitc_freeze_replay(recording, inputs, outputs);
}

int jit_freeze_dry_run(Recording *recording, const uint32_t *inputs) {
    lock_guard guard(state.lock);
    return jitc_freeze_dry_run(recording, inputs);
}

void jit_freeze_destroy(Recording *recording) {
    lock_guard guard(state.lock);
    jitc_freeze_destroy(recording);
}
