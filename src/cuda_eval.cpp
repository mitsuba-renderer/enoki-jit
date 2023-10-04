/**
 * This file contains the logic that assembles a CUDA PTX representation from a
 * recorded Dr.Jit computation graph. It implements a small template engine
 * involving plentiful use of the 'fmt' formatting routine.
 *
 * Its format interface supports the following format string characters. Note
 * that it uses the '$' (dollar) escape character, since '%' is used for CUDA
 * register prefixes (otherwise, lots of escaping would be needed).
 *
 *  Format  Input          Example result    Description
 * --------------------------------------------------------------------------
 *  $u      uint32_t      `1234`             Decimal number (32 bit)
 *  $U      uint64_t      `1234`             Decimal number (64 bit)
 * --------------------------------------------------------------------------
 *  $x      uint32_t      `4d2`              Hexadecimal number (32 bit)
 *  $X      uint64_t      `4d2`              Hexadecimal number (64 bit)
 *  $Q      uint64_t      `00000000000004d2` Hex. number, 0-filled (64 bit)
 * --------------------------------------------------------------------------
 *  $s      const char *  `foo`              Zero-terminated string
 *  $c      char          `f`                A single ASCII character
 * --------------------------------------------------------------------------
 *  $t      Variable      `f32`              Variable type
 * --------------------------------------------------------------------------
 *  $b      Variable      `b32`              Variable type, binary format
 * --------------------------------------------------------------------------
 *  $v      Variable      `%f1234`           Variable name
 * --------------------------------------------------------------------------
 *  $o      Variable      `5`                Variable offset in param. array
 * --------------------------------------------------------------------------
 *  $l      Variable      `1`                Literal value of variable (hex)
 * --------------------------------------------------------------------------
 */

#include "eval.h"
#include "internal.h"
#include "var.h"
#include "log.h"
#include "vcall.h"
#include "optix.h"

#define fmt(fmt, ...) buffer.fmt_cuda(count_args(__VA_ARGS__), fmt, ##__VA_ARGS__)
#define put(...)      buffer.put(__VA_ARGS__)
#define fmt_intrinsic(fmt, ...)                                                \
    do {                                                                       \
        size_t tmpoff = buffer.size();                                         \
        buffer.fmt_cuda(count_args(__VA_ARGS__), fmt, ##__VA_ARGS__);          \
        jitc_register_global(buffer.get() + tmpoff);                           \
        buffer.rewind_to(tmpoff);                                              \
    } while (0);

// Forward declarations
static void jitc_cuda_render_stmt(uint32_t index, const Variable *v);
static void jitc_cuda_render_var(uint32_t index, Variable *v);
static void jitc_cuda_render_scatter(const Variable *v, const Variable *ptr,
                                     const Variable *value, const Variable *index,
                                     const Variable *mask);
static void jitc_cuda_render_scatter_inc(Variable *v, const Variable *ptr,
                                         const Variable *index, const Variable *mask);
static void jitc_cuda_render_scatter_kahan(const Variable *v, uint32_t index);
static void jitc_cuda_render_printf(uint32_t index, const Variable *v,
                                    const Variable *mask);

#if defined(DRJIT_ENABLE_OPTIX)
static void jitc_cuda_render_trace(uint32_t index, const Variable *v,
                                   const Variable *valid,
                                   const Variable *pipeline,
                                   const Variable *sbt);
#endif

void jitc_cuda_assemble(ThreadState *ts, ScheduledGroup group,
                        uint32_t n_regs, uint32_t n_params) {
    bool params_global = !uses_optix && n_params > DRJIT_CUDA_ARG_LIMIT;
    bool print_labels  = std::max(state.log_level_stderr,
                                 state.log_level_callback) >= LogLevel::Trace ||
                        (jitc_flags() & (uint32_t) JitFlag::PrintIR);

#if defined(DRJIT_ENABLE_OPTIX)
    /* If use optix and the kernel contains no ray tracing operations,
       fall back to the default OptiX pipeline and shader binding table. */
    if (uses_optix) {
        /// Ensure OptiX is initialized
        (void) jitc_optix_context();
        ts->optix_pipeline = state.optix_default_pipeline;
        ts->optix_sbt = state.optix_default_sbt;
    }
#endif

    /* Special registers:

         %r0   :  Index
         %r1   :  Step
         %r2   :  Size
         %p0   :  Stopping predicate
         %rd0  :  Temporary for parameter pointers
         %rd1  :  Pointer to parameter table in global memory if too big

         %b3, %w3, %r3, %rd3, %f3, %d3, %p3: reserved for use in compound
         statements that must write a temporary result to a register.
    */

    fmt(".version $u.$u\n"
        ".target sm_$u\n"
        ".address_size 64\n\n",
        ts->ptx_version / 10, ts->ptx_version % 10,
        ts->compute_capability);

    if (!uses_optix) {
        fmt(".entry drjit_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^("
            ".param .align 8 .b8 params[$u]) { \n",
            params_global ? 8u : (n_params * (uint32_t) sizeof(void *)));
    } else {
        fmt(".const .align 8 .b8 params[$u];\n\n"
            ".entry __raygen__^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^() {\n",
            n_params * (uint32_t) sizeof(void *));
    }

    fmt("    .reg.b8   %b <$u>; .reg.b16 %w<$u>; .reg.b32 %r<$u>;\n"
        "    .reg.b64  %rd<$u>; .reg.f32 %f<$u>; .reg.f64 %d<$u>;\n"
        "    .reg.pred %p <$u>;\n\n",
        n_regs, n_regs, n_regs, n_regs, n_regs, n_regs, n_regs);

    if (!uses_optix) {
        put("    mov.u32 %r0, %ctaid.x;\n"
            "    mov.u32 %r1, %ntid.x;\n"
            "    mov.u32 %r2, %tid.x;\n"
            "    mad.lo.u32 %r0, %r0, %r1, %r2;\n");

        if (likely(!params_global)) {
           put("    ld.param.u32 %r2, [params];\n");
        } else {
           put("    ld.param.u64 %rd1, [params];\n"
               "    ldu.global.u32 %r2, [%rd1];\n");
        }

        put("    setp.ge.u32 %p0, %r0, %r2;\n"
            "    @%p0 bra done;\n"
            "\n"
            "    mov.u32 %r3, %nctaid.x;\n"
            "    mul.lo.u32 %r1, %r3, %r1;\n"
            "\n");

        fmt("body: // sm_$u\n", state.devices[ts->device].compute_capability);
    } else {
        put("    call (%r0), _optix_get_launch_index_x, ();\n"
            "    ld.const.u32 %r1, [params + 4];\n"
            "    add.u32 %r0, %r0, %r1;\n\n"
            "body:\n");
    }

    const char *params_base = "params",
               *params_type = "param";

    if (uses_optix) {
        params_type = "const";
    } else if (params_global) {
        params_base = "%rd1";
        params_type = "global";
    }

    for (uint32_t gi = group.start; gi != group.end; ++gi) {
        uint32_t index = schedule[gi].index;
        Variable *v = jitc_var(index);
        const uint32_t vti = v->type,
                       size = v->size;
        const VarType vt = (VarType) vti;
        bool assemble = false;

        if (unlikely(v->extra)) {
            auto it = state.extra.find(index);
            if (it == state.extra.end())
                jitc_fail("jit_assemble_cuda(): internal error: 'extra' entry not found!");

            const Extra &extra = it->second;
            if (print_labels && vt != VarType::Void) {
                const char *label =  jitc_var_label(index);
                if (label && label[0])
                    fmt("    // $s\n", label);
            }

            if (extra.assemble) {
                extra.assemble(v, extra);
                assemble = true;
                v = jitc_var(index); // The address of 'v' can change
            }
        }

        if (likely(v->param_type == ParamType::Input)) {
            if (v->is_literal()) {
                fmt("    ld.$s.u64 $v, [$s+$o];\n", params_type, v, params_base, v);
                continue;
            } else {
                fmt("    ld.$s.u64 %rd0, [$s+$o];\n", params_type, params_base, v);
            }

            if (size > 1)
                fmt("    mad.wide.u32 %rd0, %r0, $a, %rd0;\n", v);

            if (vt != VarType::Bool) {
                fmt("    $s$t $v, [%rd0];\n",
                    size > 1 ? "ld.global.cs." : "ldu.global.", v, v);
            } else {
                fmt("    $s %w0, [%rd0];\n"
                    "    setp.ne.u16 $v, %w0, 0;\n",
                    size > 1 ? "ld.global.cs.u8" : "ldu.global.u8", v);
            }
            continue;
        } else if (!v->is_stmt()) {
            jitc_cuda_render_var(index, v);
        } else if (likely(!assemble)) {
            jitc_cuda_render_stmt(index, v);
        }

        if (v->param_type == ParamType::Output) {
            fmt("    ld.$s.u64 %rd0, [$s+$o];\n"
                "    mad.wide.u32 %rd0, %r0, $a, %rd0;\n",
                params_type, params_base, v, v);

            if (vt != VarType::Bool) {
                fmt("    st.global.cs.$t [%rd0], $v;\n", v, v);
            } else {
                fmt("    selp.u16 %w0, 1, 0, $v;\n"
                    "    st.global.cs.u8 [%rd0], %w0;\n", v);
            }
        }
    }

    if (!uses_optix) {
        put("\n"
            "    add.u32 %r0, %r0, %r1;\n"
            "    setp.ge.u32 %p0, %r0, %r2;\n"
            "    @!%p0 bra body;\n"
            "\n"
            "done:\n");
    }

    put("    ret;\n"
        "}\n");

    uint32_t ctr = 0;
    for (auto &it : globals_map) {
        put('\n');
        put(globals.get() + it.second.start, it.second.length);
        put('\n');
        if (!it.first.callable)
            continue;
        it.second.callable_index = ctr++;
    }

    if (callable_count > 0 && !uses_optix) {
        size_t suffix_start = buffer.size(),
               suffix_target =
                   (char *) strstr(buffer.get(), ".address_size 64\n\n") -
                   buffer.get() + 18;

        fmt(".extern .global .u64 callables[$u];\n\n", callable_count_unique);
        buffer.move_suffix(suffix_start, suffix_target);

        fmt("\n.visible .global .align 8 .u64 callables[$u] = {\n",
            callable_count_unique);
        for (auto const &it : globals_map) {
            if (!it.first.callable)
                continue;

            fmt("    func_$Q$Q$s\n",
                it.first.hash.high64, it.first.hash.low64,
                it.second.callable_index + 1 < callable_count_unique ? "," : "");
        }

        put("};\n\n");
    }

    jitc_vcall_upload(ts);
}

void jitc_cuda_assemble_func(const char *name, uint32_t inst_id,
                             uint32_t n_regs, uint32_t in_size,
                             uint32_t in_align, uint32_t out_size,
                             uint32_t out_align, uint32_t data_offset,
                             const tsl::robin_map<uint64_t, uint32_t, UInt64Hasher> &data_map,
                             uint32_t n_out, const uint32_t *out_nested,
                             bool use_self) {
    bool print_labels = std::max(state.log_level_stderr,
                                 state.log_level_callback) >= LogLevel::Trace ||
                        (jitc_flags() & (uint32_t) JitFlag::PrintIR);

    put(".visible .func");
    if (out_size)
        fmt(" (.param .align $u .b8 result[$u])", out_align, out_size);
    fmt(" $s^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^(",
        uses_optix ? "__direct_callable__" : "func_");

    if (use_self) {
        put(".reg .u32 self");
        if (!data_map.empty() || in_size)
            put(", ");
    }

    if (!data_map.empty()) {
        put(".reg .u64 data");
        if (in_size)
            put(", ");
    }

    if (in_size)
        fmt(".param .align $u .b8 params[$u]", in_align, in_size);

    fmt(
        ") {\n"
        "    // VCall: $s\n"
        "    .reg.b8   %b <$u>; .reg.b16 %w<$u>; .reg.b32 %r<$u>;\n"
        "    .reg.b64  %rd<$u>; .reg.f32 %f<$u>; .reg.f64 %d<$u>;\n"
        "    .reg.pred %p <$u>;\n\n",
        name, n_regs, n_regs, n_regs, n_regs, n_regs, n_regs, n_regs);

    for (ScheduledVariable &sv : schedule) {
        Variable *v = jitc_var(sv.index);
        const uint32_t vti = v->type;
        const VarType vt = (VarType) vti;

        if (unlikely(v->extra)) {
            auto it = state.extra.find(sv.index);
            if (it == state.extra.end())
                jitc_fail("jit_assemble_cuda(): internal error: 'extra' entry "
                          "not found!");

            const Extra &extra = it->second;
            if (print_labels && vt != VarType::Void) {
                const char *label =  jitc_var_label(sv.index);
                if (label && label[0])
                    fmt("    // $s\n", label);
            }

            if (extra.assemble) {
                extra.assemble(v, extra);
                continue;
            }
        }

        if (v->vcall_iface) {
            if (vt != VarType::Bool) {
                fmt("    ld.param.$t $v, [params+$o];\n", v, v, v);
            } else {
                fmt("    ld.param.u8 %w0, [params+$o];\n"
                    "    setp.ne.u16 $v, %w0, 0;\n", v, v);
            }
        } else if (v->is_data() || vt == VarType::Pointer) {
            uint64_t key = (uint64_t) sv.index + (((uint64_t) inst_id) << 32);
            auto it = data_map.find(key);

            if (unlikely(it == data_map.end())) {
                jitc_fail("jitc_cuda_assemble_func(): could not find entry for "
                          "variable r%u in 'data_map'", sv.index);
                continue;
            }

            if (it->second == (uint32_t) -1)
                jitc_fail(
                    "jitc_cuda_assemble_func(): variable r%u is referenced by "
                    "a recorded function call. However, it was evaluated "
                    "between the recording step and code generation (which "
                    "is happening now). This is not allowed.", sv.index);

            if (vt != VarType::Bool)
                fmt("    ld.global.$t $v, [data+$u];\n",
                    v, v, it->second - data_offset);
            else
                fmt("    ld.global.u8 %w0, [data+$u];\n"
                    "    setp.ne.u16 $v, %w0, 0;\n",
                    it->second - data_offset, v);
        } else if (v->is_literal()) {
            fmt("    mov.$b $v, $l;\n", v, v, v);
        } else if (v->is_node()) {
            jitc_cuda_render_var(sv.index, v);
        } else {
            jitc_cuda_render_stmt(sv.index, v);
        }
    }

    uint32_t offset = 0;
    for (uint32_t i = 0; i < n_out; ++i) {
        uint32_t index = out_nested[i];
        if (!index)
            continue;
        const Variable *v = jitc_var(index);
        uint32_t vti = v->type;

        if ((VarType) vti != VarType::Bool) {
            fmt("    st.param.$t [result+$u], $v;\n", v, offset, v);
        } else {
            fmt("    selp.u16 %w0, 1, 0, $v;\n"
                "    st.param.u8 [result+$u], %w0;\n",
                v, offset);
        }

        offset += type_size[vti];
    }

    put("    ret;\n"
        "}");
}

static const char *reduce_op_name[(int) ReduceOp::Count] = {
    "", "add", "mul", "min", "max", "and", "or"
};

static void jitc_cuda_render_var(uint32_t index, Variable *v) {
    const char *stmt = nullptr;
    Variable *a0 = v->dep[0] ? jitc_var(v->dep[0]) : nullptr,
             *a1 = v->dep[1] ? jitc_var(v->dep[1]) : nullptr,
             *a2 = v->dep[2] ? jitc_var(v->dep[2]) : nullptr,
             *a3 = v->dep[3] ? jitc_var(v->dep[3]) : nullptr;

    switch (v->kind) {
        case VarKind::Literal:
            fmt("    mov.$b $v, $l;\n", v, v, v);
            break;

        case VarKind::Nop:
            break;

        case VarKind::Neg:
            if (jitc_is_uint(v))
                fmt("    neg.s$u $v, $v;\n", type_size[v->type]*8, v, a0);
            else
                fmt(jitc_is_single(v) ? "    neg.ftz.$t $v, $v;\n"
                                      : "    neg.$t $v, $v;\n",
                    v, v, a0);
            break;

        case VarKind::Not:
            fmt("    not.$b $v, $v;\n", v, v, a0);
            break;

        case VarKind::Sqrt:
            fmt(jitc_is_single(v) ? "    sqrt.approx.ftz.$t $v, $v;\n"
                                  : "    sqrt.rn.$t $v, $v;\n", v, v, a0);
            break;

        case VarKind::Abs:
            fmt("    abs.$t $v, $v;\n", v, v, a0);
            break;

        case VarKind::Add:
            fmt(jitc_is_single(v) ? "    add.ftz.$t $v, $v, $v;\n"
                                  : "    add.$t $v, $v, $v;\n",
                v, v, a0, a1);
            break;

        case VarKind::Sub:
            fmt(jitc_is_single(v) ? "    sub.ftz.$t $v, $v, $v;\n"
                                  : "    sub.$t $v, $v, $v;\n",
                v, v, a0, a1);
            break;

        case VarKind::Mul:
            if (jitc_is_single(v))
                stmt = "    mul.ftz.$t $v, $v, $v;\n";
            else if (jitc_is_double(v))
                stmt = "    mul.$t $v, $v, $v;\n";
            else
                stmt = "    mul.lo.$t $v, $v, $v;\n";
            fmt(stmt, v, v, a0, a1);
            break;

        case VarKind::Div:
            if (jitc_is_single(v))
                stmt = "    div.approx.ftz.$t $v, $v, $v;\n";
            else if (jitc_is_double(v))
                stmt = "    div.rn.$t $v, $v, $v;\n";
            else
                stmt = "    div.$t $v, $v, $v;\n";
            fmt(stmt, v, v, a0, a1);
            break;

        case VarKind::Mod:
            fmt("    rem.$t $v, $v, $v;\n", v, v, a0, a1);
            break;

        case VarKind::Mulhi:
            fmt("    mul.hi.$t $v, $v, $v;\n", v, v, a0, a1);
            break;

        case VarKind::Fma:
            if (jitc_is_single(v))
                stmt = "    fma.rn.ftz.$t $v, $v, $v, $v;\n";
            else if (jitc_is_double(v))
                stmt = "    fma.rn.$t $v, $v, $v, $v;\n";
            else
                stmt = "    mad.lo.$t $v, $v, $v, $v;\n";
            fmt(stmt, v, v, a0, a1, a2);
            break;

        case VarKind::Min:
            fmt(jitc_is_single(v) ? "    min.ftz.$t $v, $v, $v;\n"
                                  : "    min.$t $v, $v, $v;\n",
                                    v, v, a0, a1);
            break;

        case VarKind::Max:
            fmt(jitc_is_single(v) ? "    max.ftz.$t $v, $v, $v;\n"
                                  : "    max.$t $v, $v, $v;\n",
                                    v, v, a0, a1);
            break;

        case VarKind::Ceil:
            fmt("    cvt.rpi.$t.$t $v, $v;\n", v, v, v, a0);
            break;

        case VarKind::Floor:
            fmt("    cvt.rmi.$t.$t $v, $v;\n", v, v, v, a0);
            break;

        case VarKind::Round:
            fmt("    cvt.rni.$t.$t $v, $v;\n", v, v, v, a0);
            break;

        case VarKind::Trunc:
            fmt("    cvt.rzi.$t.$t $v, $v;\n", v, v, v, a0);
            break;

        case VarKind::Eq:
            if (jitc_is_bool(a0))
                fmt("    xor.$t $v, $v, $v;\n"
                    "    not.$t $v, $v;", v, v, a0, a1, v, v, v);
            else
                fmt("    setp.eq.$t $v, $v, $v;\n", a0, v, a0, a1);
            break;

        case VarKind::Neq:
            if (jitc_is_bool(a0))
                fmt("    xor.$t $v, $v, $v;\n", v, v, a0, a1);
            else
                fmt("    setp.ne.$t $v, $v, $v;\n", a0, v, a0, a1);
            break;

        case VarKind::Lt:
            fmt(jitc_is_uint(a0) ? "    setp.lo.$t $v, $v, $v;\n"
                                 : "    setp.lt.$t $v, $v, $v;\n",
                a0, v, a0, a1);
            break;

        case VarKind::Le:
            fmt(jitc_is_uint(a0) ? "    setp.ls.$t $v, $v, $v;\n"
                                 : "    setp.le.$t $v, $v, $v;\n",
                a0, v, a0, a1);
            break;

        case VarKind::Gt:
            fmt(jitc_is_uint(a0) ? "    setp.hi.$t $v, $v, $v;\n"
                                 : "    setp.gt.$t $v, $v, $v;\n",
                a0, v, a0, a1);
            break;

        case VarKind::Ge:
            fmt(jitc_is_uint(a0) ? "    setp.hs.$t $v, $v, $v;\n"
                                 : "    setp.ge.$t $v, $v, $v;\n",
                a0, v, a0, a1);
            break;

        case VarKind::Select:
            if (!jitc_is_bool(a1)) {
                fmt("    selp.$t $v, $v, $v, $v;\n", v, v, a1, a2, a0);
            } else {
                fmt("    and.pred %p3, $v, $v;\n"
                    "    and.pred %p2, !$v, $v;\n"
                    "    or.pred $v, %p2, %p3;\n",
                    a0, a1, a0, a2, v);
            }
            break;

        case VarKind::Popc:
            if (type_size[v->type] == 4)
                fmt("    popc.$b $v, $v;\n", v, v, a0);
            else
                fmt("    popc.$b %r3, $v;\n"
                    "    cvt.$t.u32 $v, %r3;\n", v, a0, v, v);
            break;

        case VarKind::Clz:
            if (type_size[v->type] == 4)
                fmt("    clz.$b $v, $v;\n", v, v, a0);
            else
                fmt("    clz.$b %r3, $v;\n"
                    "    cvt.$t.u32 $v, %r3;\n", v, a0, v, v);
            break;

        case VarKind::Ctz:
            if (type_size[v->type] == 4)
                fmt("    brev.$b $v, $v;\n"
                    "    clz.$b $v, $v;\n", v, v, a0, v, v, v);
            else
                fmt("    brev.$b $v, $v;\n"
                    "    clz.$b %r3, $v;\n"
                    "    cvt.$t.u32 $v, %r3;\n", v, v, a0, v, v, v, v);
            break;

        case VarKind::And:
            if (a0->type == a1->type)
                fmt("    and.$b $v, $v, $v;\n", v, v, a0, a1);
            else
                fmt("    selp.$b $v, $v, 0, $v;\n", v, v, a0, a1);
            break;

        case VarKind::Or:
            if (a0->type == a1->type)
                fmt("    or.$b $v, $v, $v;\n", v, v, a0, a1);
            else
                fmt("    selp.$b $v, -1, $v, $v;\n", v, v, a0, a1);
            break;

        case VarKind::Xor:
            fmt("    xor.$b $v, $v, $v;\n", v, v, a0, a1);
            break;

        case VarKind::Shl:
            if (type_size[v->type] == 4)
                fmt("    shl.$b $v, $v, $v;\n", v, v, a0, a1);
            else
                fmt("    cvt.u32.$t %r3, $v;\n"
                    "    shl.$b $v, $v, %r3;\n", a1, a1, v, v, a0);
            break;

        case VarKind::Shr:
            if (type_size[v->type] == 4)
                fmt("    shr.$t $v, $v, $v;\n", v, v, a0, a1);
            else
                fmt("    cvt.u32.$t %r3, $v;\n"
                    "    shr.$t $v, $v, %r3;\n", a1, a1, v, v, a0);
            break;

        case VarKind::Rcp:
            fmt(jitc_is_single(v) ? "    rcp.approx.ftz.$t $v, $v;\n"
                                  : "    rcp.rn.$t $v, $v;\n", v, v, a0);
            break;

        case VarKind::Rsqrt:
            if (jitc_is_single(v))
                fmt("    rsqrt.approx.ftz.$t $v, $v;\n", v, v, a0);
            else
                fmt("    rcp.rn.$t $v, $v;\n"
                    "    sqrt.rn.$t $v, $v;\n", v, v, a0, v, v, v);
            break;

        case VarKind::Sin:
            fmt("    sin.approx.ftz.$t $v, $v;\n", v, v, a0);
            break;

        case VarKind::Cos:
            fmt("    cos.approx.ftz.$t $v, $v;\n", v, v, a0);
            break;

        case VarKind::Exp2:
            fmt("    ex2.approx.ftz.$t $v, $v;\n", v, v, a0);
            break;

        case VarKind::Log2:
            fmt("    lg2.approx.ftz.$t $v, $v;\n", v, v, a0);
            break;


        case VarKind::Cast:
            if (jitc_is_bool(v)) {
                fmt(jitc_is_float(a0) ? "    setp.ne.$t $v, $v, 0.0;\n"
                                      : "    setp.ne.$t $v, $v, 0;\n",
                    a0, v, a0);
            } else if (jitc_is_bool(a0)) {
                fmt(jitc_is_float(v) ? "    selp.$t $v, 1.0, 0.0, $v;\n"
                                     : "    selp.$t $v, 1, 0, $v;\n",
                    v, v, a0);
            } else if (jitc_is_float(v) && !jitc_is_float(a0)) {
                fmt("    cvt.rn.$t.$t $v, $v;\n", v, a0, v, a0);
            } else if (!jitc_is_float(v) && jitc_is_float(a0)) {
                fmt("    cvt.rzi.$t.$t $v, $v;\n", v, a0, v, a0);
            } else {
                fmt(jitc_is_float(v) && jitc_is_float(a0) &&
                            type_size[v->type] < type_size[a0->type]
                        ? "    cvt.rn.$t.$t $v, $v;\n"
                        : "    cvt.$t.$t $v, $v;\n",
                    v, a0, v, a0);
            }
            break;

        case VarKind::Bitcast:
            fmt("    mov.$b $v, $v;\n", v, v, a0);
            break;

        case VarKind::Gather: {
                bool index_zero = a1->is_literal() && a1->literal == 0;
                bool unmasked = a2->is_literal() && a2->literal == 1;
                bool is_bool = v->type == (uint32_t) VarType::Bool;

                if (!unmasked)
                    fmt("    @!$v bra l_$u_masked;\n", a2, v->reg_index);

                if (index_zero) {
                    fmt("    mov.u64 %rd3, $v;\n", a0);
                } else if (type_size[v->type] == 1) {
                    fmt("    cvt.u64.$t %rd3, $v;\n"
                        "    add.u64 %rd3, %rd3, $v;\n", a1, a1, a0);
                } else {
                    fmt("    mad.wide.$t %rd3, $v, $a, $v;\n",
                        a1, a1, v, a0);
                }

                if (is_bool) {
                    fmt("    ld.global.nc.u8 %w0, [%rd3];\n"
                        "    setp.ne.u16 $v, %w0, 0;\n", v);
                } else {
                    fmt("    ld.global.nc.$t $v, [%rd3];\n", v, v);
                }

                if (!unmasked)
                    fmt("    bra.uni l_$u_done;\n\n"
                        "l_$u_masked:\n"
                        "    mov.$b $v, 0;\n\n"
                        "l_$u_done:\n", v->reg_index,
                        v->reg_index, v, v, v->reg_index);
            }
            break;

        case VarKind::Scatter:
            jitc_cuda_render_scatter(v, a0, a1, a2, a3);
            break;

        case VarKind::ScatterInc:
            jitc_cuda_render_scatter_inc(v, a0, a1, a2);
            break;

        case VarKind::ScatterKahan:
            jitc_cuda_render_scatter_kahan(v, index);
            break;


        case VarKind::VCallSelf:
            fmt("    mov.u32 $v, self;\n", v);
            break;

        case VarKind::Counter:
            fmt("    mov.$t $v, %r0;\n", v, v);
            break;

        case VarKind::Printf:
            jitc_cuda_render_printf(index, v, a0);
            break;

        case VarKind::Dispatch:
            jitc_var_vcall_assemble((VCall *) state.extra[index].callback_data,
                                    a0->reg_index, a1->reg_index, a2->reg_index,
                                    a3 ? a3->reg_index : 0);
            break;

        case VarKind::TexLookup:
            fmt("    .reg.f32 $v_out_<4>;\n", v);
            if (a3)
                fmt("    tex.3d.v4.f32.f32 {$v_out_0, $v_out_1, $v_out_2, $v_out_3}, [$v, {$v, $v, $v, $v}];\n",
                    v, v, v, v, a0, a1, a2, a3, a3);
            else if (a2)
                fmt("    tex.2d.v4.f32.f32 {$v_out_0, $v_out_1, $v_out_2, $v_out_3}, [$v, {$v, $v}];\n",
                    v, v, v, v, a0, a1, a2);
            else
                fmt("    tex.1d.v4.f32.f32 {$v_out_0, $v_out_1, $v_out_2, $v_out_3}, [$v, {$v}];\n",
                    v, v, v, v, a0, a1);
            break;

        case VarKind::TexFetchBilerp:
            fmt("    .reg.f32 $v_out_<4>;\n"
                "    tld4.$c.2d.v4.f32.f32 {$v_out_0, $v_out_1, $v_out_2, $v_out_3}, [$v, {$v, $v}];\n",
                v, "rgba"[v->literal], v, v, v, v, a0, a1, a2);
            break;

#if defined(DRJIT_ENABLE_OPTIX)
        case VarKind::TraceRay:
            jitc_cuda_render_trace(index, v, a0, a1, a2);
            break;

#endif
        case VarKind::Extract:
            fmt("    mov.$b $v, $v_out_$u;\n", v, v, a0, (uint32_t) v->literal);
            break;

        default:
            jitc_fail("jitc_cuda_render_var(): unhandled variable kind \"%s\"!",
                      var_kind_name[(uint32_t) v->kind]);
    }
}

static void jitc_cuda_render_scatter(const Variable *v,
                                     const Variable *ptr,
                                     const Variable *value,
                                     const Variable *index,
                                     const Variable *mask) {
    bool index_zero = index->is_literal() && index->literal == 0;
    bool unmasked = mask->is_literal() && mask->literal == 1;
    bool is_bool = value->type == (uint32_t) VarType::Bool;

    if (!unmasked)
        fmt("    @!$v bra l_$u_done;\n", mask, v->reg_index);

    if (index_zero) {
        fmt("    mov.u64 %rd3, $v;\n", ptr);
    } else if (type_size[v->type] == 1) {
        fmt("    cvt.u64.$t %rd3, $v;\n"
            "    add.u64 %rd3, %rd3, $v;\n", index, index, ptr);
    } else {
        fmt("    mad.wide.$t %rd3, $v, $a, $v;\n",
            index, index, value, ptr);
    }
    const char *op = reduce_op_name[v->literal];
    const ThreadState *ts = thread_state_cuda;


    if (v->literal && callable_depth == 0 && type_size[value->type] == 4 &&
        ts->ptx_version >= 62 && ts->compute_capability >= 70 &&
        (jitc_flags() & (uint32_t) JitFlag::AtomicReduceLocal)) {
        fmt("    {\n"
            "        .func reduce_$s_$t(.param .u64 ptr, .param .$t value);\n"
            "        call.uni reduce_$s_$t, (%rd3, $v);\n"
            "    }\n",
            op, value, value, op, value, value);

        // Intrinsic to perform an intra-warp reduction before writing to global memory
        fmt_intrinsic(
            ".func reduce_$s_$t(.param .u64 ptr,\n"
            "                   .param .$t value) {\n"
            "    .reg .pred %p<14>;\n"
            "    .reg .$t %q<19>;\n"
            "    .reg .b32 %r<41>;\n"
            "    .reg .b64 %rd<2>;\n"
            "\n"
            "    ld.param.u64 %rd0, [ptr];\n"
            "    ld.param.$t %q3, [value];\n"
            "    activemask.b32 %r1;\n"
            "    match.any.sync.b64 %r2, %rd0, %r1;\n"
            "    setp.eq.s32 %p1, %r2, -1;\n"
            "    @%p1 bra.uni fast_path;\n"
            "\n"
            "    brev.b32 %r10, %r2;\n"
            "    bfind.shiftamt.u32 %r40, %r10;\n"
            "    shf.l.wrap.b32 %r12, -2, -2, %r40;\n"
            "    and.b32 %r39, %r2, %r12;\n"
            "    setp.ne.s32 %p2, %r39, 0;\n"
            "    vote.sync.any.pred %p3, %p2, %r1;\n"
            "    @!%p3 bra maybe_scatter;\n"
            "    mov.b32 %r5, %q3;\n"
            "\n"
            "slow_path_repeat:\n"
            "    brev.b32 %r14, %r39;\n"
            "    bfind.shiftamt.u32 %r15, %r14;\n"
            "    shfl.sync.idx.b32 %r17, %r5, %r15, 31, %r1;\n"
            "    mov.b32 %q6, %r17;\n"
            "    @%p2 $s.$t %q3, %q3, %q6;\n"
            "    shf.l.wrap.b32 %r19, -2, -2, %r15;\n"
            "    and.b32 %r39, %r39, %r19;\n"
            "    setp.ne.s32 %p2, %r39, 0;\n"
            "    vote.sync.any.pred %p3, %p2, %r1;\n"
            "    @!%p3 bra maybe_scatter;\n"
            "    bra.uni slow_path_repeat;\n"
            "\n"
            "fast_path:\n"
            "    mov.b32 %r22, %q3;\n"
            "    shfl.sync.down.b32 %r26, %r22, 16, 31, %r1;\n"
            "    mov.b32 %q7, %r26;\n"
            "    $s.$t %q8, %q7, %q3;\n"
            "    mov.b32 %r27, %q8;\n"
            "    shfl.sync.down.b32 %r29, %r27, 8, 31, %r1;\n"
            "    mov.b32 %q9, %r29;\n"
            "    $s.$t %q10, %q8, %q9;\n"
            "    mov.b32 %r30, %q10;\n"
            "    shfl.sync.down.b32 %r32, %r30, 4, 31, %r1;\n"
            "    mov.b32 %q11, %r32;\n"
            "    $s.$t %q12, %q10, %q11;\n"
            "    mov.b32 %r33, %q12;\n"
            "    shfl.sync.down.b32 %r34, %r33, 2, 31, %r1;\n"
            "    mov.b32 %q13, %r34;\n"
            "    $s.$t %q14, %q12, %q13;\n"
            "    mov.b32 %r35, %q14;\n"
            "    shfl.sync.down.b32 %r37, %r35, 1, 31, %r1;\n"
            "    mov.b32 %q15, %r37;\n"
            "    $s.$t %q3, %q14, %q15;\n"
            "    mov.u32 %r40, 0;\n"
            "\n"
            "maybe_scatter:\n"
            "    mov.u32 %r38, %laneid;\n"
            "    setp.ne.s32 %p13, %r40, %r38;\n"
            "    @%p13 bra done;\n"
            "    red.$s.$t [%rd0], %q3;\n"
            "\n"
            "done:\n"
            "    ret;\n"
            "}",
            op, value, value, value, value, op, value, op, value, op, value, op, value, op,
            value, op, value, op, value
        );
    } else {
        const char *op_type = v->literal ? "red" : "st";

        if (is_bool)
            fmt("    selp.u16 %w0, 1, 0, $v;\n"
                "    $s.global$s$s.u8 [%rd3], %w0;\n",
                value, op_type, v->literal ? "." : "", op);
        else
            fmt("    $s.global$s$s.$t [%rd3], $v;\n", op_type,
                v->literal ? "." : "", op, value, value);
    }

    if (!unmasked)
        fmt("\nl_$u_done:\n", v->reg_index);
}

static void jitc_cuda_render_scatter_inc(Variable *v,
                                         const Variable *ptr,
                                         const Variable *index,
                                         const Variable *mask) {
    bool index_zero = index->is_literal() && index->literal == 0;
    bool unmasked = mask->is_literal() && mask->literal == 1;

    fmt_intrinsic(
        ".func (.param .u32 rv) reduce_inc_u32 (.param .u64 ptr) {\n"
        "    .reg .pred %p<2>;\n"
        "    .reg .b32 %r<11>;\n"
        "    .reg .b64 %rd<2>;\n"
        "\n"
        "    ld.param.u64 %rd1, [ptr];\n"
        "    activemask.b32 %r2;\n"
        "    mov.u32 %r3, %lanemask_lt;\n"
        "    and.b32 %r3, %r3, %r2;\n"
        "    setp.ne.u32 %p1, %r3, 0;\n"
        "    @%p1 bra L2;\n"
        "\n"
        "L1:\n"
        "    popc.b32 %r4, %r2;\n"
        "    atom.global.add.u32 %r5, [%rd1], %r4;\n"
        "\n"
        "L2:\n"
        "    popc.b32 %r6, %r3;\n"
        "    brev.b32 %r7, %r2;\n"
        "    bfind.shiftamt.u32 %r8, %r7;\n"
        "    shfl.sync.idx.b32 %r9, %r5, %r8, 31, %r2;\n"
        "    add.u32 %r10, %r6, %r9;\n"
        "    st.param.u32 [rv], %r10;\n"
        "    ret;\n"
        "}\n");

    if (!unmasked)
        fmt("    @!$v bra l_$u_done;\n", mask, v->reg_index);

    if (index_zero) {
        fmt("    mov.u64 %rd3, $v;\n", ptr);
    } else {
        fmt("    mad.wide.$t %rd3, $v, 4, $v;\n",
            index, index, ptr);
    }

    fmt("    {\n"
        "        .func (.param .u32 rv) reduce_inc_u32 (.param .u64 ptr);\n"
        "        call.uni ($v), reduce_inc_u32, (%rd3);\n"
        "    }\n", v);

    if (!unmasked)
        fmt("\nl_$u_done:\n", v->reg_index);

    v->consumed = 1;
}

static void jitc_cuda_render_scatter_kahan(const Variable *v, uint32_t v_index) {
    const Extra &extra = state.extra[v_index];

    const Variable *ptr_1 = jitc_var(extra.dep[0]),
                   *ptr_2 = jitc_var(extra.dep[1]),
                   *index = jitc_var(extra.dep[2]),
                   *mask = jitc_var(extra.dep[3]),
                   *value = jitc_var(extra.dep[4]);

    bool unmasked = mask->is_literal() && mask->literal == 1;

    if (!unmasked)
        fmt("    @!$v bra l_$u_done;\n", mask, v->reg_index);

    fmt("    mad.wide.$t %rd2, $v, $a, $v;\n"
        "    mad.wide.$t %rd3, $v, $a, $v;\n",
        index, index, value, ptr_1,
        index, index, value, ptr_2);

    const char* op_suffix = jitc_is_single(value) ? ".ftz" : "";

    fmt("    {\n"
        "        .reg.$t %before, %after, %value, %case_1, %case_2;\n"
        "        .reg.$t %abs_before, %abs_value, %result;\n"
        "        .reg.pred %cond;\n"
        "\n"
        "        mov.$t %value, $v;\n"
        "        atom.global.add.$t %before, [%rd2], %value;\n"
        "        add$s.$t %after, %before, %value;\n"
        "        sub$s.$t %case_1, %before, %after;\n"
        "        add$s.$t %case_1, %case_1, %value;\n"
        "        sub$s.$t %case_2, %value, %after;\n"
        "        add$s.$t %case_2, %case_2, %before;\n"
        "        abs$s.$t %abs_before, %before;\n"
        "        abs$s.$t %abs_value, %value;\n"
        "        setp.ge.$t %cond, %abs_before, %abs_value;\n"
        "        selp.$t %result, %case_1, %case_2, %cond;\n"
        "        red.global.add.$t [%rd3], %result;\n"
        "    }\n",
        value,
        value,
        value, value,
        value,
        op_suffix, value,
        op_suffix, value,
        op_suffix, value,
        op_suffix, value,
        op_suffix, value,
        op_suffix, value,
        op_suffix, value,
        value,
        value,
        value);

    if (!unmasked)
        fmt("\nl_$u_done:\n", v->reg_index);
}

static void jitc_cuda_render_printf(uint32_t index, const Variable *v,
                                    const Variable *mask) {
    const Extra &extra = state.extra[index];
    const char *format_str = (const char *) v->literal;

    uint32_t offset = 0, align = 0;
    for (uint32_t i = 0; i < extra.n_dep; ++i) {
        Variable *v2 = jitc_var(extra.dep[i]);
        uint32_t vti = v2->type;
        if ((VarType) vti == VarType::Void)
            continue;
        uint32_t tsize = type_size[vti];
        if ((VarType) vti == VarType::Float32)
            tsize = 8;

        offset = (offset + tsize - 1) / tsize * tsize;
        offset += tsize;
        align = std::max(align, tsize);
    }

    if (align == 0)
        align = 1;
    if (offset == 0)
        offset = 1;

    put("    {\n");
    fmt("        .local .align $u .b8 buf[$u];\n", align, offset);
    put("        .extern .func (.param .b32 rv) vprintf (.param .b64 fmt, .param .b64 buf);\n");
    put("        .global .align 1 .b8 print_data[] = { ");

    for (uint32_t i = 0; ; ++i) {
        buffer.put_u32((uint32_t) format_str[i]);
        if (!format_str[i])
            break;
        put(", ");
    }
    put(" };\n");

    offset = 0;
    for (uint32_t i = 0; i < extra.n_dep; ++i) {
        Variable *v2 = jitc_var(extra.dep[i]);
        const VarType vt = (VarType) v2->type;
        uint32_t tsize = vt == VarType::Float32 ? 8 : type_size[(int) vt];

        if (vt == VarType::Void)
            continue;

        offset = (offset + tsize - 1) / tsize * tsize;

        if (vt == VarType::Float32) {
            fmt("        cvt.f64.f32 %d3, $v;\n"
                "        st.local.f64 [buf+$u], %d3;\n", v2, offset);
        } else {
            fmt("        st.local.$t [buf+$u], $v;\n", v2, offset, v2);
        }

        offset += tsize;
    }
    put("\n"
        "        .reg.b64 %fmt_generic, %buf_generic;\n"
        "        cvta.global.u64 %fmt_generic, print_data;\n"
        "        cvta.local.u64 %buf_generic, buf;\n"
        "        {\n"
        "            .param .b64 fmt_p;\n"
        "            .param .b64 buf_p;\n"
        "            .param .b32 rv_p;\n"
        "            st.param.b64 [fmt_p], %fmt_generic;\n"
        "            st.param.b64 [buf_p], %buf_generic;\n"
        "            ");
    if (mask && !(mask->is_literal() && mask->literal == 1))
        fmt("@$v ", mask);
    put("call (rv_p), vprintf, (fmt_p, buf_p);\n"
        "        }\n"
        "    }\n");
}

#if defined(DRJIT_ENABLE_OPTIX)
static void jitc_cuda_render_trace(uint32_t index, const Variable *v,
                                   const Variable *valid,
                                   const Variable *pipeline,
                                   const Variable *sbt) {
    ThreadState *ts = thread_state(JitBackend::CUDA);
    OptixPipelineData *pipeline_p = (OptixPipelineData *) pipeline->literal;
    OptixShaderBindingTable *sbt_p = (OptixShaderBindingTable*) sbt->literal;
    bool problem = false;
    bool shadow_ray = v->literal == 1;

    if (ts->optix_pipeline == state.optix_default_pipeline) {
        ts->optix_pipeline = pipeline_p;
    } else if (ts->optix_pipeline != pipeline_p) {
        jitc_log(
            Warn,
            "jit_eval(): more than one OptiX pipeline was used within a single "
            "kernel, which is not supported. Please split your kernel into "
            "smaller parts (e.g. using `dr::eval()`). Disabling the ray "
            "tracing operation to avoid potential undefined behavior.");
        problem = true;
    }

    if (ts->optix_sbt == state.optix_default_sbt) {
        ts->optix_sbt = sbt_p;
    } else if (ts->optix_sbt != sbt_p) {
        jitc_log(
            Warn,
            "jit_eval(): more than one OptiX shader binding table was used "
            "within a single kernel, which is not supported. Please split your "
            "kernel into smaller parts (e.g. using `dr::eval()`). Disabling "
            "the ray tracing operation to avoid potential undefined behavior.");
        problem = true;
    }

    const Extra &extra = state.extra[index];
    uint32_t payload_count = extra.n_dep - 16;

    fmt("    .reg.u32 $v_out_<32>;\n", v);

    if (problem) {
        for (uint32_t i = 0; i < 32; ++i)
            fmt("    mov.b32 $v_out_$u, 0;\n", v, i);
        return;
    }

    bool masked = !valid->is_literal() || valid->literal != 1;
    if (masked)
        fmt("    @!$v bra l_masked_$u;\n", valid, v->reg_index);

    fmt("    .reg.u32 $v_payload_type, $v_payload_count;\n"
        "    mov.u32 $v_payload_type, 0;\n"
        "    mov.u32 $v_payload_count, $u;\n",
        v, v, v, v, payload_count);

    // =====================================================
    // 1. Traverse
    // =====================================================
    put("    call (");
    for (uint32_t i = 0; i < 32; ++i)
        fmt("$v_out_$u$s", v, i, i + 1 < 32 ? ", " : "");
    put("), _optix_hitobject_traverse, (");
    fmt("$v_payload_type, ", v);
    for (uint32_t i = 1; i < 16; ++i)
        fmt("$v, ", jitc_var(extra.dep[i]));
    fmt("$v_payload_count, ", v);
    for (uint32_t i = 16; i < extra.n_dep; ++i)
        fmt("$v$s", jitc_var(extra.dep[i]), (i - 16 < 32) ? ", " : "");
    for (uint32_t i = payload_count; i < 32; ++i)
        fmt("$v_out_$u$s", v, i, (i + 1 < 32) ? ", " : "");
    put(");\n");

    // =====================================================
    // 2. Reorder
    // =====================================================
    const Variable *coherent = jitc_var(extra.dep[0]);
    bool ser_enabled = jitc_flags() & (uint32_t) JitFlag::ShaderExecutionReordering;
    bool coherent_enabled =
        (!coherent->is_literal() || coherent->literal == 0) || // False if coherent
        !(jitc_flags() & (uint32_t) JitFlag::OnlyReorderIfNonCoherent);

    bool shadow_enabled =
        !shadow_ray || // false if shadow_ray
        !(jitc_flags() & (uint32_t) JitFlag::DoNotReorderShadowRays);

    if (ser_enabled && coherent_enabled && shadow_enabled) {
        fmt("    .reg.u32 $v_reorder_hint, $v_reorder_hint_bits;\n"
            "    mov.u32 $v_reorder_hint, 0;\n"
            "    mov.u32 $v_reorder_hint_bits, 0;\n",
            v, v, v, v);
        put("    call (), _optix_hitobject_reorder, (");
        fmt("$v_reorder_hint, $v_reorder_hint_bits);\n ", v, v);
    }

    // =====================================================
    // 3. Invoke hit/miss program
    // =====================================================
    if (!shadow_ray) {
        put("    call (");
        for (uint32_t i = 0; i < 32; ++i)
            fmt("$v_out_$u$s", v, i, i + 1 < 32 ? ", " : "");
        put("), _optix_hitobject_invoke, (");
        fmt("$v_payload_type, ", v);
        fmt("$v_payload_count, ", v);
        for (uint32_t i = 0; i < 32; ++i)
            fmt("$v_out_$u$s", v, i, i + 1 < 32 ? ", " : "");
        put(");\n");
    } else {
        fmt("    call ($v_out_0), _optix_hitobject_is_hit, ();", v);
    }

    if (masked)
        fmt("\nl_masked_$u:\n", v->reg_index);
}
#endif

/// Convert an IR template with '$' expressions into valid IR
static void jitc_cuda_render_stmt(uint32_t index, const Variable *v) {
    const char *s = v->stmt;
    if (unlikely(*s == '\0'))
        return;
    put("    ");
    char c;
    do {
        const char *start = s;
        while (c = *s, c != '\0' && c != '$')
            s++;
        put(start, s - start);

        if (c == '$') {
            s++;
            const char **prefix_table = nullptr, type = *s++;
            switch (type) {
                case 'n': put(";\n    "); continue;
                case 't': prefix_table = type_name_ptx; break;
                case 'b': prefix_table = type_name_ptx_bin; break;
                case 's': prefix_table = type_size_str; break;
                case 'r': prefix_table = type_prefix; break;
                default:
                    jitc_fail("jit_cuda_render_stmt(): encountered invalid \"$\" "
                              "expression (unknown type \"%c\") in \"%s\"!", type, v->stmt);
            }

            uint32_t arg_id = *s++ - '0';
            if (unlikely(arg_id > 4))
                jitc_fail("jit_cuda_render_stmt(%s): encountered invalid \"$\" "
                          "expression (argument out of bounds)!", v->stmt);

            uint32_t dep_id = arg_id == 0 ? index : v->dep[arg_id - 1];
            if (unlikely(dep_id == 0))
                jitc_fail("jit_cuda_render_stmt(%s): encountered invalid \"$\" "
                          "expression (referenced variable %u is missing)!", v->stmt, arg_id);

            const Variable *dep = jitc_var(dep_id);
            const char *prefix = prefix_table[(int) dep->type];
            put(prefix, strlen(prefix));

            if (type == 'r') {
                buffer.put_u32(dep->reg_index);
                if (unlikely(dep->reg_index == 0))
                    jitc_fail("jitc_cuda_render_stmt(): variable has no register index!");
            }
        }
    } while (c != '\0');

    put(";\n");
}

/// Virtual function call code generation -- CUDA/PTX-specific bits
void jitc_var_vcall_assemble_cuda(VCall *vcall, uint32_t vcall_reg,
                                  uint32_t self_reg, uint32_t mask_reg,
                                  uint32_t offset_reg, uint32_t data_reg,
                                  uint32_t n_out, uint32_t in_size,
                                  uint32_t in_align, uint32_t out_size,
                                  uint32_t out_align) {

    // =====================================================
    // 1. Conditional branch
    // =====================================================

    fmt("\n    @!%p$u bra l_masked_$u;\n\n"
        "    { // VCall: $s\n", mask_reg, vcall_reg, vcall->name);

    // =====================================================
    // 2. Determine unique callable ID
    // =====================================================

    // %r3: callable ID
    // %rd3: (high 32 bit): data offset
    fmt("\n"
        "        mad.wide.u32 %rd3, %r$u, 8, %rd$u;\n"
        "        ld.global.u64 %rd3, [%rd3];\n"
        "        cvt.u32.u64 %r3, %rd3;\n",
        self_reg, offset_reg);

    // =====================================================
    // 3. Turn callable ID into a function pointer
    // =====================================================

    if (!uses_optix)
        put("        ld.global.u64 %rd2, callables[%r3];\n");
    else
        put("        call (%rd2), _optix_call_direct_callable, (%r3);\n");

    // =====================================================
    // 4. Obtain pointer to supplemental call data
    // =====================================================

    if (data_reg)
        fmt("        shr.u64 %rd3, %rd3, 32;\n"
            "        add.u64 %rd3, %rd3, %rd$u;\n",
            data_reg);

    // %rd2: function pointer (if applicable)
    // %rd3: call data pointer with offset

    // =====================================================
    // 5. Generate the actual function call
    // =====================================================

    put("\n");

    // Special handling for predicates
    for (uint32_t in : vcall->in) {
        auto it = state.variables.find(in);
        if (it == state.variables.end())
            continue;
        const Variable *v2 = &it->second;

        if ((VarType) v2->type != VarType::Bool)
            continue;

        fmt("        selp.u16 %w$u, 1, 0, %p$u;\n",
            v2->reg_index, v2->reg_index);
    }

    put("        {\n");

    // Call prototype
    put("            proto: .callprototype");
    if (out_size)
        fmt(" (.param .align $u .b8 result[$u])", out_align, out_size);
    put(" _(");
    if (vcall->use_self) {
        put(".reg .u32 self");
        if (data_reg || in_size)
            put(", ");
    }
    if (data_reg) {
        put(".reg .u64 data");
        if (in_size)
            put(", ");
    }
    if (in_size)
        fmt(".param .align $u .b8 params[$u]", in_align, in_size);
    put(");\n");

    // Input/output parameter arrays
    if (out_size)
        fmt("            .param .align $u .b8 out[$u];\n", out_align, out_size);
    if (in_size)
        fmt("            .param .align $u .b8 in[$u];\n", in_align, in_size);

    // =====================================================
    // 5.1. Pass the input arguments
    // =====================================================

    uint32_t offset = 0;
    for (uint32_t in : vcall->in) {
        auto it = state.variables.find(in);
        if (it == state.variables.end())
            continue;
        const Variable *v2 = &it->second;
        uint32_t size = type_size[v2->type];

        const char *tname = type_name_ptx[v2->type],
                   *prefix = type_prefix[v2->type];

        // Special handling for predicates (pass via u8)
        if ((VarType) v2->type == VarType::Bool) {
            tname = "u8";
            prefix = "%w";
        }

        fmt("            st.param.$s [in+$u], $s$u;\n", tname, offset, prefix,
            v2->reg_index);

        offset += size;
    }

    // =====================================================
    // 5.2. Call the target function
    // =====================================================

    bool multiple_enabled =
        (vcall->callables_set.size() > 1) || // False if only one target
        !(jitc_flags() & (uint32_t) JitFlag::ReorderVCallOnlyIfMultipleTargets);

    if (uses_optix &&
        (jitc_flags() & (uint32_t) JitFlag::ShaderExecutionReordering) &&
        (jitc_flags() & (uint32_t) JitFlag::ReorderBeforeVCalls) &&
        multiple_enabled
    ) {
        if (jitc_flags() & (uint32_t) JitFlag::UseInstanceIdInVCallReorder) {
            put("            call (), _optix_hitobject_make_nop, ();\n"
                "            .reg.u32 reorder_hint, reorder_hint_bits, reorder_hint_high;\n");
            fmt("            mov.u32 reorder_hint, %r$u;\n", self_reg);
            put("            shl.b32 reorder_hint_high, %r3, 0x10U;\n"
                "            or.b32 reorder_hint, reorder_hint_high, reorder_hint_high;\n");
        } else {
            put("            call (), _optix_hitobject_make_nop, ();\n"
                "            .reg.u32 reorder_hint, reorder_hint_bits;\n"
                "            mov.u32 reorder_hint, %r3;\n");
        }

        put("            mov.u32 reorder_hint_bits, 0x20U;\n"
            "            call (), _optix_hitobject_reorder, (reorder_hint, "
            "reorder_hint_bits);\n ");
    }

    if (vcall->use_self) {
        fmt("            call $s%rd2, (%r$u$s$s), proto;\n",
            out_size ? "(out), " : "", self_reg,
            data_reg ? ", %rd3" : "",
            in_size ? ", in" : "");
    } else {
        fmt("            call $s%rd2, ($s$s$s), proto;\n",
            out_size ? "(out), " : "", data_reg ? "%rd3" : "",
            data_reg && in_size ? ", " : "", in_size ? "in" : "");
    }

    // =====================================================
    // 5.3. Read back the output arguments
    // =====================================================

    offset = 0;
    for (uint32_t i = 0; i < n_out; ++i) {
        uint32_t index = vcall->out_nested[i],
                 index_2 = vcall->out[i];
        auto it = state.variables.find(index);
        if (it == state.variables.end())
            continue;
        uint32_t size = type_size[it->second.type],
                 load_offset = offset;
        offset += size;

        // Skip if expired
        auto it2 = state.variables.find(index_2);
        if (it2 == state.variables.end())
            continue;

        const Variable *v2 = &it2.value();
        if (v2->reg_index == 0 || v2->param_type == ParamType::Input)
            continue;

        const char *tname = type_name_ptx[v2->type],
                   *prefix = type_prefix[v2->type];

        // Special handling for predicates (pass via u8)
        if ((VarType) v2->type == VarType::Bool) {
            tname = "u8";
            prefix = "%w";
        }

        fmt("            ld.param.$s $s$u, [out+$u];\n",
            tname, prefix, v2->reg_index, load_offset);
    }

    put("        }\n\n");

    // =====================================================
    // 6. Special handling for predicates return value(s)
    // =====================================================

    for (uint32_t out : vcall->out) {
        auto it = state.variables.find(out);
        if (it == state.variables.end())
            continue;
        const Variable *v2 = &it->second;
        if ((VarType) v2->type != VarType::Bool)
            continue;
        if (v2->reg_index == 0 || v2->param_type == ParamType::Input)
            continue;

        // Special handling for predicates
        fmt("        setp.ne.u16 %p$u, %w$u, 0;\n",
            v2->reg_index, v2->reg_index);
    }


    fmt("        bra.uni l_done_$u;\n"
        "    }\n", vcall_reg);

    // =====================================================
    // 7. Prepare output registers for masked lanes
    // =====================================================

    fmt("\nl_masked_$u:\n", vcall_reg);
    for (uint32_t out : vcall->out) {
        auto it = state.variables.find(out);
        if (it == state.variables.end())
            continue;
        const Variable *v2 = &it->second;
        if (v2->reg_index == 0 || v2->param_type == ParamType::Input)
            continue;

        fmt("    mov.$b $v, 0;\n", v2, v2);
    }

    fmt("\nl_done_$u:\n", vcall_reg);
}

