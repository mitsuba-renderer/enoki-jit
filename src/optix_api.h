/*
    src/optix_api.h -- Low-level interface to OptiX API

    Copyright (c) 2022 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

// ---------------------------------------------------------------------

/// Try to resolve the OptiX API functions
extern bool jitc_optix_api_init();

/// Free any resources allocated by jitc_optix_api_init()
extern void jitc_optix_api_shutdown();

// ---------------------------------------------------------------------

#if !defined(DR_OPTIX_SYM)
#  define DR_OPTIX_SYM(x) extern x;
#endif

using OptixResult = int;
using OptixProgramGroupKind = int;
using OptixCompileDebugLevel = int;
using OptixDeviceContext = void*;
using OptixLogCallback = void (*)(unsigned int, const char *, const char *, void *);
using OptixTask = void*;
using OptixModule = void*;
using OptixProgramGroup = void*;
using OptixPipeline = void*;

#define OPTIX_EXCEPTION_FLAG_NONE                     0
#define OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW           1
#define OPTIX_EXCEPTION_FLAG_TRACE_DEPTH              2
#define OPTIX_EXCEPTION_FLAG_DEBUG                    8
#define OPTIX_ERROR_VALIDATION_FAILURE                7053
#define OPTIX_COMPILE_DEBUG_LEVEL_NONE                0x2350
#define OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL             0x2351
#define OPTIX_COMPILE_DEBUG_LEVEL_MODERATE            0x2353
#define OPTIX_COMPILE_DEBUG_LEVEL_FULL                0x2352
#define OPTIX_COMPILE_OPTIMIZATION_LEVEL_0            0x2340
#define OPTIX_COMPILE_OPTIMIZATION_LEVEL_1            0x2341
#define OPTIX_COMPILE_OPTIMIZATION_LEVEL_2            0x2342
#define OPTIX_COMPILE_OPTIMIZATION_LEVEL_3            0x2343
#define OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF      0
#define OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL      ((int) 0xFFFFFFFF)
#define OPTIX_MODULE_COMPILE_STATE_COMPLETED          0x2364
#define OPTIX_PROGRAM_GROUP_KIND_RAYGEN               0x2421
#define OPTIX_PROGRAM_GROUP_KIND_CALLABLES            0x2425
#define OPTIX_PROGRAM_GROUP_KIND_MISS                 0x2422
#define OPTIX_SBT_RECORD_HEADER_SIZE                  32
#define OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS 1

struct OptixDeviceContextOptions {
    OptixLogCallback logCallbackFunction;
    void *logCallbackData;
    int logCallbackLevel;
    int validationMode;
};

struct OptixPayloadType {
    unsigned int numPayloadValues;
    const unsigned int *payloadSemantics;
};

struct OptixModuleCompileOptions {
    int maxRegisterCount;
    int optLevel;
    int debugLevel;
    const void *boundValues;
    unsigned int numBoundValues;
    unsigned int numPayloadTypes;
    const OptixPayloadType *payloadTypes;
};

struct OptixPipelineLinkOptions {
    unsigned int maxTraceDepth;
};

struct OptixProgramGroupSingleModule {
    OptixModule module;
    const char* entryFunctionName;
};

struct OptixProgramGroupHitgroup {
    OptixModule moduleCH;
    const char* entryFunctionNameCH;
    OptixModule moduleAH;
    const char* entryFunctionNameAH;
    OptixModule moduleIS;
    const char* entryFunctionNameIS;
};

struct OptixProgramGroupCallables {
    OptixModule moduleDC;
    const char* entryFunctionNameDC;
    OptixModule moduleCC;
    const char* entryFunctionNameCC;
};

struct OptixProgramGroupDesc {
    OptixProgramGroupKind kind;
    unsigned int flags;

    union {
        OptixProgramGroupSingleModule raygen;
        OptixProgramGroupSingleModule miss;
        OptixProgramGroupSingleModule exception;
        OptixProgramGroupCallables callables;
        OptixProgramGroupHitgroup hitgroup;
    };
};

struct OptixStackSizes {
    unsigned int cssRG;
    unsigned int cssMS;
    unsigned int cssCH;
    unsigned int cssAH;
    unsigned int cssIS;
    unsigned int cssCC;
    unsigned int dssDC;
};

struct OptixProgramGroupOptions {
    OptixPayloadType *payloadType;
};

DR_OPTIX_SYM(OptixResult (*optixQueryFunctionTable)(int, unsigned int, void *,
                                                    const void **, void *,
                                                    size_t));
DR_OPTIX_SYM(const char *(*optixGetErrorName)(OptixResult));
DR_OPTIX_SYM(const char *(*optixGetErrorString)(OptixResult));
DR_OPTIX_SYM(OptixResult (*optixDeviceContextCreate)(
    CUcontext, const OptixDeviceContextOptions *, OptixDeviceContext *));
DR_OPTIX_SYM(OptixResult (*optixDeviceContextDestroy)(OptixDeviceContext));
DR_OPTIX_SYM(
    OptixResult (*optixDeviceContextSetCacheEnabled)(OptixDeviceContext, int));
DR_OPTIX_SYM(OptixResult (*optixDeviceContextSetCacheLocation)(
    OptixDeviceContext, const char *));
DR_OPTIX_SYM(OptixResult (*optixModuleCreate)(
    OptixDeviceContext, const OptixModuleCompileOptions *,
    const OptixPipelineCompileOptions *, const char *, size_t, char *, size_t *,
    OptixModule *));
DR_OPTIX_SYM(OptixResult (*optixModuleCreateWithTasks)(
    OptixDeviceContext, const OptixModuleCompileOptions *,
    const OptixPipelineCompileOptions *, const char *, size_t, char *, size_t *,
    OptixModule *, OptixTask *));
DR_OPTIX_SYM(OptixResult (*optixModuleGetCompilationState)(OptixModule, int *));
DR_OPTIX_SYM(OptixResult (*optixModuleDestroy)(OptixModule));
DR_OPTIX_SYM(OptixResult (*optixTaskExecute)(OptixTask, OptixTask *,
                                             unsigned int, unsigned int *));
DR_OPTIX_SYM(OptixResult (*optixProgramGroupCreate)(
    OptixDeviceContext, const OptixProgramGroupDesc *, unsigned int,
    const OptixProgramGroupOptions *, char *, size_t *, OptixProgramGroup *));
DR_OPTIX_SYM(OptixResult (*optixProgramGroupDestroy)(OptixProgramGroup));
DR_OPTIX_SYM(OptixResult (*optixPipelineCreate)(
    OptixDeviceContext, const OptixPipelineCompileOptions *,
    const OptixPipelineLinkOptions *, const OptixProgramGroup *, unsigned int,
    char *, size_t *, OptixPipeline *));
DR_OPTIX_SYM(OptixResult (*optixPipelineDestroy)(OptixPipeline));
DR_OPTIX_SYM(OptixResult (*optixLaunch)(OptixPipeline, CUstream, CUdeviceptr,
                                        size_t, const OptixShaderBindingTable *,
                                        unsigned int, unsigned int,
                                        unsigned int));
DR_OPTIX_SYM(OptixResult (*optixSbtRecordPackHeader)(OptixProgramGroup,
                                                     void *));
DR_OPTIX_SYM(OptixResult (*optixPipelineSetStackSize)(
    OptixPipeline, unsigned int, unsigned int, unsigned int, unsigned int));
DR_OPTIX_SYM(OptixResult (*optixProgramGroupGetStackSize)(OptixProgramGroup,
                                                          OptixStackSizes *,
                                                          OptixPipeline));
