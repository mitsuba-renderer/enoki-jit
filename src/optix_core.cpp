#include <drjit-core/optix.h>
#include <tsl/robin_map.h>
#include "optix.h"
#include "optix_api.h"
#include "internal.h"
#include "log.h"
#include "eval.h"
#include "var.h"
#include "op.h"
#include "util.h"

#if !defined(NDEBUG) || defined(DRJIT_ENABLE_OPTIX_DEBUG_VALIDATION)
#define DRJIT_ENABLE_OPTIX_DEBUG_VALIDATION_ON
#endif

#define jitc_optix_check(err) jitc_optix_check_impl((err), __FILE__, __LINE__)
extern void jitc_optix_check_impl(OptixResult errval, const char *file, const int line);

static bool jitc_optix_cache_hit = false;
static bool jitc_optix_cache_global_disable = false;

void jitc_optix_log(unsigned int level, const char *tag, const char *message, void *) {
    size_t len = strlen(message);
    if (level <= (uint32_t) state.log_level_stderr)
        fprintf(stderr, "jit_optix_log(): [%s] %s%s", tag, message,
                (len > 0 && message[len - 1] == '\n') ? "" : "\n");

    if (strcmp(tag, "DISKCACHE") == 0 &&
        strncmp(message, "Cache miss for key", 18) == 0)
        jitc_optix_cache_hit = false;

    if (strcmp(tag, "DISK CACHE") == 0 &&
        strncmp(message, "OPTIX_CACHE_MAXSIZE is set to 0", 31) == 0)
        jitc_optix_cache_global_disable = true;
}

static OptixPipelineCompileOptions jitc_optix_default_compile_options() {
    OptixPipelineCompileOptions pco { };
    pco.numAttributeValues = 2;
    pco.pipelineLaunchParamsVariableName = "params";

#ifndef DRJIT_ENABLE_OPTIX_DEBUG_VALIDATION_ON
    pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#else
    pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG |
                         OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                         OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#endif

    return pco;
}

OptixDeviceContext jitc_optix_context() {
    ThreadState *ts = thread_state(JitBackend::CUDA);
    OptixDeviceContext &ctx = state.devices[ts->device].optix_context;

    if (!ctx) {
        if (!jitc_optix_api_init())
            jitc_raise("Could not initialize OptiX!");

        OptixDeviceContextOptions ctx_opts {
            jitc_optix_log, nullptr, 4,
#ifndef DRJIT_ENABLE_OPTIX_DEBUG_VALIDATION_ON
            OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF
#else
            OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
#endif
        };

        jitc_optix_check(optixDeviceContextCreate(ts->context, &ctx_opts, &ctx));

#if !defined(_WIN32)
        jitc_optix_check(optixDeviceContextSetCacheLocation(ctx, jitc_temp_path));
#else
        size_t len = wcstombs(nullptr, jitc_temp_path, 0) + 1;
        std::unique_ptr<char[]> temp(new char[len]);
        wcstombs(temp.get(), jitc_temp_path, len);
        jitc_optix_check(optixDeviceContextSetCacheLocation(ctx, temp.get()));
#endif
        jitc_optix_check(optixDeviceContextSetCacheEnabled(ctx, 1));
    }

    // =====================================================
    // Create default OptiX pipeline for testcases, etc.
    // =====================================================

    if (!state.optix_default_sbt_index) {
        OptixPipelineCompileOptions pco = jitc_optix_default_compile_options();
        OptixModuleCompileOptions mco { };
#ifndef DRJIT_ENABLE_OPTIX_DEBUG_VALIDATION_ON
        mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#else
        mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
#endif

        const char *minimal = ".version 6.0 .target sm_50 .address_size 64 "
                              ".entry __miss__dr() { ret; }";

        char log[128];
        size_t log_size = sizeof(log);

        OptixModule mod;
        jitc_optix_check(optixModuleCreate(
            ctx, &mco, &pco, minimal, strlen(minimal), log, &log_size, &mod));

        OptixProgramGroupDesc pgd { };
        pgd.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgd.miss.module = mod;
        pgd.miss.entryFunctionName = "__miss__dr";

        OptixProgramGroupOptions pgo { };
        OptixProgramGroup pg;
        log_size = sizeof(log);
        jitc_optix_check(optixProgramGroupCreate(ctx, &pgd, 1, &pgo, log, &log_size, &pg));

        OptixShaderBindingTable sbt {};
        sbt.missRecordBase = jitc_malloc(AllocType::HostPinned, OPTIX_SBT_RECORD_HEADER_SIZE);
        jitc_optix_check(optixSbtRecordPackHeader(pg, sbt.missRecordBase));
        sbt.missRecordBase = jitc_malloc_migrate(sbt.missRecordBase, AllocType::Device, 1);
        sbt.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
        sbt.missRecordCount = 1;

        uint32_t pipeline_index = jitc_optix_configure_pipeline(&pco, mod, &pg, 1);
        auto it2 = state.extra.find(pipeline_index);
        if (it2 == state.extra.end())
            jitc_fail("jitc_optix_context(): 'extra' entry not found!");
        state.optix_default_pipeline = (OptixPipelineData*) it2->second.callback_data;

        uint32_t sbt_index = jitc_optix_configure_sbt(&sbt, pipeline_index);
        auto it = state.extra.find(sbt_index);
        if (it == state.extra.end())
            jitc_fail("jitc_optix_context(): 'extra' entry not found!");
        state.optix_default_sbt
            = (OptixShaderBindingTable*) it->second.callback_data;

        state.optix_default_sbt_index = sbt_index;
        jitc_var_dec_ref(pipeline_index);
    }

    return ctx;
}

void jitc_optix_context_destroy(Device &d) {
    if (d.optix_context) {
        jitc_optix_check(optixDeviceContextDestroy(d.optix_context));
        d.optix_context = nullptr;
    }
}

uint32_t jitc_optix_configure_pipeline(const OptixPipelineCompileOptions *pco,
                                       OptixModule module,
                                       const OptixProgramGroup *pg,
                                       uint32_t pg_count) {
    jitc_log(InfoSym, "jitc_optix_configure_pipeline(pg_count=%u)", pg_count);

    if (!pco || !module || !pg || pg_count == 0)
        jitc_raise("jitc_optix_configure_pipeline(): invalid input arguments!");

    OptixPipelineData *p = new OptixPipelineData();
    p->module = module;
    p->program_groups = std::vector<OptixProgramGroup>();
    memcpy(&p->compile_options, pco, sizeof(OptixPipelineCompileOptions));
    for (uint32_t i = 0; i < pg_count; ++i)
        p->program_groups.push_back(pg[i]);

    uint32_t index =
        jitc_var_new_node_0(JitBackend::CUDA, VarKind::Nop,
                            VarType::Void, 1, 0, (uintptr_t) p);

    jitc_var(index)->extra = 1;

    // Free pipeline resources when this variable is destroyed
    Extra &extra = state.extra[index];
    extra.callback_data = p;
    extra.callback = [](uint32_t /*index*/, int free, void *ptr) {
        if (!free)
            return;
        jitc_log(InfoSym, "jit_optix_configure_pipeline(): free optix pipeline");
        OptixPipelineData *p = (OptixPipelineData*) ptr;
        for (size_t i = 0; i < p->program_groups.size(); i++)
            jitc_optix_check(optixProgramGroupDestroy(p->program_groups[i]));
        jitc_optix_check(optixModuleDestroy(p->module));
        delete p;
    };

    return index;
}

uint32_t jitc_optix_configure_sbt(const OptixShaderBindingTable *sbt,
                                  uint32_t pipeline) {
    jitc_log(InfoSym, "jitc_optix_configure_sbt()");

    if (!sbt || !pipeline)
        jitc_raise("jitc_optix_configure_sbt(): invalid input arguments!");

    if (jitc_var_type(pipeline) != VarType::Void)
        jitc_raise("jitc_optix_configure_sbt(): type mismatch for pipeline argument!");

    OptixShaderBindingTable *p = new OptixShaderBindingTable();
    memcpy(p, sbt, sizeof(OptixShaderBindingTable));

    uint32_t index = jitc_var_new_node_1(
        JitBackend::CUDA, VarKind::Nop, VarType::Void, 1, 0, pipeline,
        jitc_var(pipeline), (uintptr_t) p);

    jitc_var(index)->extra = 1;

    // Free SBT resources when this variable is destroyed
    Extra &extra = state.extra[index];
    extra.callback_data = p;
    extra.callback = [](uint32_t /*index*/, int free, void *ptr) {
        if (!free)
            return;
        jitc_log(InfoSym, "jit_optix_configure_sbt(): free optix shader binding table");
        OptixShaderBindingTable *sbt = (OptixShaderBindingTable*) ptr;
        jitc_free(sbt->hitgroupRecordBase);
        jitc_free(sbt->missRecordBase);
        delete sbt;
    };

    return index;
}

void jitc_optix_update_sbt(uint32_t index, const OptixShaderBindingTable *sbt) {
    Extra &extra = state.extra[index];
    memcpy(extra.callback_data, sbt, sizeof(OptixShaderBindingTable));
}

bool jitc_optix_compile(ThreadState *ts, const char *buf, size_t buf_size,
                        const char *kern_name, Kernel &kernel) {
    char error_log[16384];

    if (!optixModuleCreateWithTasks)
        jitc_fail("jit_optix_compile(): OptiX not initialized, make sure "
                  "evaluation happens before Optix shutdown!");

    // =====================================================
    // 2. Compile an OptiX module
    // =====================================================

    OptixModuleCompileOptions mco { };
#ifndef DRJIT_ENABLE_OPTIX_DEBUG_VALIDATION_ON
    mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#else
    mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
#endif

    jitc_optix_cache_hit = !jitc_optix_cache_global_disable;
    size_t log_size = sizeof(error_log);
    OptixDeviceContext &optix_context = state.devices[ts->device].optix_context;
    OptixPipelineData &pipeline = *ts->optix_pipeline;

    OptixTask task;
    int rv = optixModuleCreateWithTasks(
        optix_context, &mco, &pipeline.compile_options, buf, buf_size,
        error_log, &log_size, &kernel.optix.mod, &task);

    if (rv) {
        jitc_log(Error, "jit_optix_compile(): "
                 "optixModuleCreateWithTasks() failed. Please see the PTX "
                 "assembly listing and error message below:\n\n%s\n\n%s",
                 buf, error_log);
        jitc_optix_check(rv);
    }

    std::function<void(OptixTask)> execute_task = [&](OptixTask task) {
        unsigned int max_new_tasks = std::max(pool_size(), 1u);

        std::unique_ptr<OptixTask[]> new_tasks =
            std::make_unique<OptixTask[]>(max_new_tasks);
        unsigned int new_task_count = 0;
        optixTaskExecute(task, new_tasks.get(), max_new_tasks, &new_task_count);

        parallel_for(
            drjit::blocked_range<size_t>(0, new_task_count, 1),
            [&](const drjit::blocked_range<size_t> &range) {
                for (auto i = range.begin(); i != range.end(); ++i) {
                    OptixTask new_task = new_tasks[i];
                    execute_task(new_task);
                }
            }
        );
    };
    execute_task(task);

    int compilation_state = 0;
    jitc_optix_check(
        optixModuleGetCompilationState(kernel.optix.mod, &compilation_state));
    if (compilation_state != OPTIX_MODULE_COMPILE_STATE_COMPLETED)
        jitc_fail("jit_optix_compile(): optixModuleGetCompilationState() "
                  "indicates that the compilation did not complete "
                  "succesfully. The module's compilation state is: %#06x\n"
                  "Please see the PTX assembly listing and error message "
                  "below:\n\n%s\n\n%s", compilation_state, buf, error_log);

    // =====================================================
    // 3. Create an OptiX program group
    // =====================================================

    size_t n_programs = 1 + callable_count_unique;

    OptixProgramGroupOptions pgo { };
    std::unique_ptr<OptixProgramGroupDesc[]> pgd(
        new OptixProgramGroupDesc[n_programs]);
    memset(pgd.get(), 0, n_programs * sizeof(OptixProgramGroupDesc));

    pgd[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgd[0].raygen.module = kernel.optix.mod;
    pgd[0].raygen.entryFunctionName = strdup(kern_name);

    for (auto const &it : globals_map) {
        if (!it.first.callable)
            continue;

        char *name = (char *) malloc_check(52);
        snprintf(name, 52, "__direct_callable__%016llx%016llx",
                 (unsigned long long) it.first.hash.high64,
                 (unsigned long long) it.first.hash.low64);

        uint32_t index = 1 + it.second.callable_index;
        pgd[index].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        pgd[index].callables.moduleDC = kernel.optix.mod;
        pgd[index].callables.entryFunctionNameDC = name;
    }

    kernel.optix.pg = new OptixProgramGroup[n_programs];
    kernel.optix.pg_count = (uint32_t) n_programs;

    log_size = sizeof(error_log);
    rv = optixProgramGroupCreate(optix_context, pgd.get(),
                                 (unsigned int) n_programs, &pgo, error_log,
                                 &log_size, kernel.optix.pg);
    if (rv) {
        jitc_log(Error, "jit_optix_compile(): optixProgramGroupCreate() "
                 "failed. Please see the PTX assembly listing and error "
                 "message below:\n\n%s\n\n%s", buf, error_log);
        jitc_optix_check(rv);
    }

    const size_t stride = OPTIX_SBT_RECORD_HEADER_SIZE;
    uint8_t *sbt_record = (uint8_t *)
        jitc_malloc(AllocType::HostPinned, n_programs * stride);

    for (size_t i = 0; i < n_programs; ++i)
        jitc_optix_check(optixSbtRecordPackHeader(
            kernel.optix.pg[i], sbt_record + stride * i));

    kernel.optix.sbt_record = (uint8_t *)
        jitc_malloc_migrate(sbt_record, AllocType::Device, 1);

    // =====================================================
    // 4. Create an OptiX pipeline
    // =====================================================

    OptixPipelineLinkOptions link_options {};
    link_options.maxTraceDepth = 1;

    size_t size_before = pipeline.program_groups.size();
#if 0
    OptixStackSizes stack_sizes;
    unsigned int cssRG = 0, dssDC = 0;
#endif

    for (uint32_t i = 0; i < n_programs; ++i) {
        if (i == 0)
            free((char *) pgd[0].raygen.entryFunctionName);
        else
            free((char *) pgd[i].callables.entryFunctionNameDC);
        pipeline.program_groups.push_back(kernel.optix.pg[i]);
#if 0
        jitc_optix_check(optixProgramGroupGetStackSize(
            pipeline.program_groups[i], &stack_sizes, pipeline));
        if (i == 0)
            cssRG = stack_sizes.cssRG;
        else
            dssDC = std::max(dssDC, stack_sizes.dssDC);
#endif
    }

    log_size = sizeof(error_log);
    rv = optixPipelineCreate(optix_context, &pipeline.compile_options,
                             &link_options, pipeline.program_groups.data(),
                             (unsigned int) pipeline.program_groups.size(),
                             error_log, &log_size, &kernel.optix.pipeline);
    if (rv) {
        jitc_log(Error, "jit_optix_compile(): optixPipelineCreate() failed. "
                 "Please see the PTX assembly listing and error message "
                 "below:\n\n%s\n\n%s", buf, error_log);
        jitc_optix_check(rv);
    }

    // Setup the direct stack and continuation stack size.
    // See OptiX documentation for more detail:
    // https://raytracing-docs.nvidia.com/optix7/guide/index.html#program_pipeline_creation#7235

    OptixStackSizes ssp = {};
    for (size_t i = 0; i < pipeline.program_groups.size(); ++i) {
        OptixStackSizes ss;
        rv = optixProgramGroupGetStackSize(pipeline.program_groups[i], &ss,
                                           kernel.optix.pipeline);
        if (rv) {
            jitc_log(Error, "jit_optix_compile(): optixProgramGroupGetStackSize() "
                            "failed:\n\n%s", error_log);
            jitc_optix_check(rv);
        }
        ssp.cssRG = std::max(ssp.cssRG, ss.cssRG);
        ssp.cssMS = std::max(ssp.cssMS, ss.cssMS);
        ssp.cssCH = std::max(ssp.cssCH, ss.cssCH);
        ssp.cssAH = std::max(ssp.cssAH, ss.cssAH);
        ssp.cssIS = std::max(ssp.cssIS, ss.cssIS);
        ssp.cssCC = std::max(ssp.cssCC, ss.cssCC);
        ssp.dssDC = std::max(ssp.dssDC, ss.dssDC);
    }

    if (ssp.cssCC > 0)
        jitc_log(Error, "jit_optix_compile(): an OptiX program is using "
                        "continuous callables which is not supported by Dr.Jit!");

    unsigned int max_dc_depth = 2; // Support nested VCalls
    unsigned int dc_stack_size_from_traversal = 0; // DC is not invoked from IS or AH.
    unsigned int dc_stack_size_from_state = max_dc_depth * ssp.dssDC; // DC is invoked from RG, MS, or CH.
    unsigned int continuation_stack_size = ssp.cssRG + std::max(std::max(ssp.cssCH, ssp.cssMS), ssp.cssAH + ssp.cssIS);

    unsigned int max_traversable_graph_depth = 2; // Support instancing
    if (pipeline.compile_options.traversableGraphFlags == OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS)
        max_traversable_graph_depth = 1;

    rv = optixPipelineSetStackSize(kernel.optix.pipeline,
                                   dc_stack_size_from_traversal,
                                   dc_stack_size_from_state,
                                   continuation_stack_size,
                                   max_traversable_graph_depth);
    if (rv) {
        jitc_log(Error, "jit_optix_compile(): optixPipelineSetStackSize() "
                        "failed:\n\n%s", error_log);
        jitc_optix_check(rv);
    }

    kernel.size = 0;
    kernel.data = nullptr;
    pipeline.program_groups.resize(size_before);
    return jitc_optix_cache_hit;
}

void jitc_optix_free(const Kernel &kernel) {
    jitc_optix_check(optixPipelineDestroy(kernel.optix.pipeline));
    for (uint32_t i = 0; i < kernel.optix.pg_count; ++i)
        jitc_optix_check(optixProgramGroupDestroy(kernel.optix.pg[i]));
    delete[] kernel.optix.pg;
    jitc_optix_check(optixModuleDestroy(kernel.optix.mod));
    jitc_free(kernel.optix.sbt_record);
}

void jitc_optix_launch(ThreadState *ts, const Kernel &kernel,
                       uint32_t launch_size, const void *args,
                       uint32_t n_args) {
    OptixShaderBindingTable &sbt = *ts->optix_sbt;
    sbt.raygenRecord = kernel.optix.sbt_record;

    if (kernel.optix.pg_count > 1) {
        sbt.callablesRecordBase = kernel.optix.sbt_record + OPTIX_SBT_RECORD_HEADER_SIZE;
        sbt.callablesRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
        sbt.callablesRecordCount = kernel.optix.pg_count - 1;
    }

    uint32_t offset = 0;

    /* We accept kernel launches up to the maximum value of 2^30 threads by
       default. When using an older non-preemptable WDDM driver setup on
       Windows, a long-running kernel may freeze the OS graphics and eventually
       cause a device reset. That's not good, so we submit smaller batches that
       correspond roughly to 1 sample/pixel for a full HD resolution frame. */
    uint32_t limit = state.devices[ts->device].preemptable ? (1 << 30) : (1 << 21);

    while (launch_size > 0) {
        uint32_t sub_launch_size = launch_size < limit ? launch_size : limit;

        // Bytes 4..8 used to store optional offset parameter
        if (offset != 0)
            cuMemsetD32Async(
                (CUdeviceptr) ((uint8_t *) args + sizeof(uint32_t)),
                offset, 1, ts->stream);

        jitc_optix_check(
            optixLaunch(kernel.optix.pipeline, ts->stream, (CUdeviceptr) args,
                        n_args * sizeof(void *), &sbt,
                        sub_launch_size, 1, 1));

        launch_size -= sub_launch_size;
        offset += sub_launch_size;
    }
}

void jitc_optix_ray_trace(uint32_t n_args, uint32_t *args,
                          int shadow_ray, uint32_t mask,
                          uint32_t pipeline, uint32_t sbt) {
    VarType types[]{ VarType::Bool,    VarType::UInt64,  VarType::Float32,
                     VarType::Float32, VarType::Float32, VarType::Float32,
                     VarType::Float32, VarType::Float32, VarType::Float32,
                     VarType::Float32, VarType::Float32, VarType::UInt32,
                     VarType::UInt32,  VarType::UInt32,  VarType::UInt32,
                     VarType::UInt32,  VarType::UInt32,  VarType::UInt32,
                     VarType::UInt32,  VarType::UInt32,  VarType::UInt32,
                     VarType::UInt32,  VarType::UInt32,  VarType::UInt32 };

    if (n_args < 16)
        jitc_raise("jit_optix_ray_trace(): too few arguments (got %u < 16)", n_args);

    uint32_t np = n_args - 16, size = 0;
    if (np > 32)
        jitc_raise("jit_optix_ray_trace(): too many payloads (got %u > 32)", np);

    if (jitc_var_type(pipeline) != VarType::Void ||
        jitc_var_type(sbt) != VarType::Void)
        jitc_raise("jit_optix_ray_trace(): type mismatch for pipeline argument!");

    // Validate input types, determine size of the operation
    bool placeholder = false, dirty = false;
    for (uint32_t i = 0; i <= n_args; ++i) {
        uint32_t index = i < n_args ? args[i] : mask;
        VarType ref = i < n_args ? types[i] : VarType::Bool;
        const Variable *v = jitc_var(index);
        if ((VarType) v->type != ref)
            jitc_raise("jit_optix_ray_trace(): type mismatch for arg. %u (got %s, "
                       "expected %s)",
                       i, type_name[v->type], type_name[(int) ref]);
        size = std::max(size, v->size);
        placeholder |= (bool) v->placeholder;
        dirty |= v->is_dirty();
    }

    for (uint32_t i = 0; i <= n_args; ++i) {
        uint32_t index = (i < n_args) ? args[i] : mask;
        const Variable *v = jitc_var(index);
        if (v->size != 1 && v->size != size)
            jitc_raise("jit_optix_ray_trace(): arithmetic involving arrays of "
                       "incompatible size!");
    }

    if (dirty) {
        jitc_eval(thread_state(JitBackend::CUDA));
        dirty = false;

        for (uint32_t i = 0; i <= n_args; ++i) {
            uint32_t index = (i < n_args) ? args[i] : mask;
            dirty |= jitc_var(index)->is_dirty();
        }

        if (dirty)
            jitc_raise("jit_optix_ray_trace(): inputs remain dirty after evaluation!");
    }

    // Potentially apply any masks on the mask stack
    Ref valid = steal(jitc_var_mask_apply(mask, size));

    jitc_log(InfoSym, "jit_optix_ray_trace(): tracing %u ray%s, %u payload value%s%s.",
             size, size != 1 ? "s" : "", np, np == 1 ? "" : "s",
             placeholder ? " (part of a recorded computation)" : "");

    Ref index = steal(jitc_var_new_node_3(
        JitBackend::CUDA, VarKind::TraceRay, VarType::Void, size, placeholder,
        valid, jitc_var(valid), pipeline, jitc_var(pipeline), sbt,
        jitc_var(sbt), shadow_ray));

    Variable *v = jitc_var(index);
    v->extra = v->optix = 1;

    Extra &extra = state.extra[index];
    extra.n_dep = n_args;
    extra.dep = (uint32_t *) malloc_check(sizeof(uint32_t) * extra.n_dep);
    for (uint32_t i = 0; i < n_args; ++i) {
        uint32_t id = args[i];
        extra.dep[i] = id;
        jitc_var_inc_ref(id);
    }

    for (uint32_t i = 0; i < np; ++i)
        args[16 + i] = jitc_var_new_node_1(
            JitBackend::CUDA, VarKind::Extract, VarType::UInt32,
            size, placeholder, index, jitc_var(index), (uint64_t) i);
}

void jitc_optix_check_impl(OptixResult errval, const char *file,
                           const int line) {
    if (unlikely(errval != 0)) {
        const char *name = optixGetErrorName(errval),
                   *msg  = optixGetErrorString(errval);
        jitc_fail("jit_optix_check(): API error %04i (%s): \"%s\" in "
                  "%s:%i.", (int) errval, name, msg, file, line);
    }
}

void jit_optix_check_impl(int errval, const char *file, const int line) {
    if (errval) {
        lock_guard guard(state.lock);
        jitc_optix_check_impl(errval, file, line);
    }
}

