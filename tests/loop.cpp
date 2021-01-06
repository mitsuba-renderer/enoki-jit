#include "test.h"
#include <enoki-jit/containers.h>
#include <utility>

template <typename Mask> struct Loop {
    static constexpr JitBackend Backend = Mask::Backend;

    template <typename... Args>
    Loop(const char *name, Args &... args)
        : m_name(name), m_state(0), m_se_offset((uint32_t) -1),
          m_size(0), m_record(jit_flag(JitFlag::LoopRecord)) {
        if constexpr (sizeof...(Args) > 0) {
            (put(args), ...);
            init();
        }
    }

    ~Loop() {
        if (m_record && m_se_offset != (uint32_t) -1) {
            // An error occurred while recording a loop
            jit_side_effects_rollback(Backend, m_se_offset);
            jit_set_flag(JitFlag::PostponeSideEffects, m_se_flag);
        }

        if (!m_record && m_index_out.size() > 0) {
            // An error occurred while evaluating a loop wavefront-style
            for (uint32_t i = 0; i < m_index_out.size(); ++i)
                jit_var_dec_ref_ext(m_index_out[i]);
            jit_var_mask_pop(Backend);
        }

        if (m_state != 0 && m_state != 3)
            jit_log(Warn, "Loop(): de-allocated in an inconsistent state. "
                          "(Loop.cond() must run exactly twice!)");
    }

    /// Register a loop variable // TODO: nested arrays, structs, etc.
    template <typename Value> void put(Value &value) {
        /// XXX complain when variables are attached
        m_index_p.push_back(value.index_ptr());
        m_index_in.push_back(value.index());
        size_t size = value.size();
        if (m_size != 0 && size != 1 && size != m_size)
            jit_raise("Loop.put(): loop variables have inconsistent sizes!");
        if (size > m_size)
            m_size = size;
    }

    /// Configure the loop variables for recording
    void init() {
        if (m_state)
            jit_raise("Loop(): was already initialized!");

        if (m_record) {
            step();
            m_se_offset = jit_side_effects_scheduled(Backend);
            m_se_flag = jit_flag(JitFlag::PostponeSideEffects);
            jit_set_flag(JitFlag::PostponeSideEffects, 1);
            m_state = 1;
        }
    }

    bool cond(const Mask &cond) {
        if (m_record)
            return cond_record(cond);
        else
            return cond_wavefront(cond);
    }

protected:
    bool cond_wavefront(const Mask &cond) {
        // Need to mask loop variables for disabled lanes
        if (m_cond.index()) {
            for (uint32_t i = 0; i < m_index_p.size(); ++i) {
                uint32_t i1 = *m_index_p[i], i2 = m_index_out[i];
                *m_index_p[i] = jit_var_new_op_3(JitOp::Select, m_cond.index(), i1, i2);
                jit_var_dec_ref_ext(i1);
                jit_var_dec_ref_ext(i2);
            }
            jit_var_mask_pop(Backend);
            m_index_out.clear();
            m_cond = Mask();
        }

        // Ensure all loop state is evaluated
        jit_var_schedule(cond.index());
        for (uint32_t i = 0; i < m_index_p.size(); ++i)
            jit_var_schedule(*m_index_p[i]);
        jit_eval();

        // Do we run another iteration?
        if (jit_var_any(cond.index())) {
            // Mask scatters/gathers/vcalls in the next iteration
            m_cond = cond;
            jit_var_mask_push(Backend, cond.index());

            for (uint32_t i = 0; i < m_index_p.size(); ++i) {
                uint32_t index = *m_index_p[i];
                jit_var_inc_ref_ext(index);
                m_index_out.push_back(index);
            }

            return true;
        } else {
            return false;
        }
    }

    bool cond_record(const Mask &cond) {
        switch (m_state++) {
            case 0:
                jit_raise("Loop(): must be initialized first!");

            case 1:
                m_cond = cond; // detach
                step();
                for (uint32_t i = 0; i < m_index_p.size(); ++i)
                    m_index_body.push_back(*m_index_p[i]);
                return true;

            case 2:
                for (uint32_t i = 0; i < m_index_p.size(); ++i)
                    m_index_out.push_back(*m_index_p[i]);
                jit_var_loop(m_name, m_cond.index(),
                             (uint32_t) m_index_p.size(), m_index_body.data(),
                             m_index_out.data(), m_se_offset,
                             m_index_out.data());
                for (uint32_t i = 0; i < m_index_p.size(); ++i) {
                    uint32_t &index = *m_index_p[i];
                    jit_var_dec_ref_ext(index);
                    index = m_index_out[i];
                }
                jit_set_flag(JitFlag::PostponeSideEffects, m_se_flag);
                return false;

            default:
                jit_raise("Loop(): invalid state!");
        }

        return false;
    }

    // Insert an indirection via placeholder variables
    void step() {
        for (size_t i = 0; i < m_index_p.size(); ++i) {
            uint32_t &index = *m_index_p[i],
                     next = jit_var_new_placeholder(index, 0);
            jit_var_dec_ref_ext(index);
            index = next;
        }
    }

private:
    const char *m_name;
    ek_vector<uint32_t> m_index_in;
    ek_vector<uint32_t> m_index_body;
    ek_vector<uint32_t> m_index_out;
    ek_vector<uint32_t *> m_index_p;
    Mask m_cond; // XXX detached_t<Mask>
    uint32_t m_state;
    uint32_t m_se_offset;
    int m_se_flag;
    size_t m_size;
    bool m_record;
};

TEST_CUDA(01_record_loop) {
    // Tests a simple loop evaluated at once, or in parts
    for (uint32_t i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::LoopRecord, i != 0);
        jit_set_flag(JitFlag::LoopOptimize, i == 2);

        for (uint32_t j = 0; j < 2; ++j) {
            UInt32 x = arange<UInt32>(10);
            Float y = zero<Float>(1);
            Float z = 1;

            Loop<Mask> loop("MyLoop", x, y, z);
            while (loop.cond(x < 5)) {
                y += Float(x);
                x += 1;
                z += 1;
            }

            if (j == 0) {
                jit_var_schedule(x.index());
                jit_var_schedule(y.index());
                jit_var_schedule(z.index());
            }

            jit_assert(strcmp(z.str(), "[6, 5, 4, 3, 2, 1, 1, 1, 1, 1]") == 0);
            jit_assert(strcmp(y.str(), "[10, 10, 9, 7, 4, 0, 0, 0, 0, 0]") == 0);
            jit_assert(strcmp(x.str(), "[5, 5, 5, 5, 5, 5, 6, 7, 8, 9]") == 0);
        }
    }
}

TEST_CUDA(02_side_effect) {
    // Tests that side effects only happen once
    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::LoopRecord, i != 0);
        jit_set_flag(JitFlag::LoopOptimize, i == 2);

        for (uint32_t j = 0; j < 3; ++j) {
            UInt32 x = arange<UInt32>(10);
            Float y = zero<Float>(1);
            UInt32 target = zero<UInt32>(11);

            Loop<Mask> loop("MyLoop", x, y);
            while (loop.cond(x < 5)) {
                scatter_reduce(ReduceOp::Add, target, UInt32(1), x);
                y += Float(x);
                x += 1;
            }

            if (j == 0) {
                jit_var_schedule(x.index());
                jit_var_schedule(y.index());
            }

            jit_assert(strcmp(y.str(), "[10, 10, 9, 7, 4, 0, 0, 0, 0, 0]") == 0);
            jit_assert(strcmp(x.str(), "[5, 5, 5, 5, 5, 5, 6, 7, 8, 9]") == 0);
            jit_assert(strcmp(target.str(), "[1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0]") == 0);
        }
    }
}

TEST_CUDA(03_side_effect_2) {
    // Tests that side effects work that don't reference loop variables
    for (uint32_t i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::LoopRecord, i != 0);
        jit_set_flag(JitFlag::LoopOptimize, i == 2);

        for (uint32_t j = 0; j < 3; ++j) {
            UInt32 x = arange<UInt32>(10);
            Float y = zero<Float>(1);
            UInt32 target = zero<UInt32>(11);

            Loop<Mask> loop("MyLoop", x, y);
            while (loop.cond(x < 5)) {
                scatter_reduce(ReduceOp::Add, target, UInt32(2), UInt32(2));
                y += Float(x);
                x += 1;
            }

            if (j == 0) {
                jit_var_schedule(x.index());
                jit_var_schedule(y.index());
            }

            jit_assert(strcmp(y.str(), "[10, 10, 9, 7, 4, 0, 0, 0, 0, 0]") == 0);
            jit_assert(strcmp(x.str(), "[5, 5, 5, 5, 5, 5, 6, 7, 8, 9]") == 0);
            jit_assert(strcmp(target.str(), "[0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0]") == 0);
        }
    }
}

TEST_CUDA(04_side_effect_masking) {
    // Tests that side effects work that don't reference loop variables
    for (uint32_t i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::LoopRecord, i != 0);
        jit_set_flag(JitFlag::LoopOptimize, i == 2);

        for (uint32_t j = 0; j < 3; ++j) {
            UInt32 x = arange<UInt32>(1000000);
            UInt32 target = zero<UInt32>(10);

            Loop<Mask> loop("MyLoop", x);
            while (loop.cond(x < 10)) {
                // This is sure to segfault if not masked correctly
                scatter_reduce(ReduceOp::Add, target, UInt32(1), x);
                x += 1;
            }

            jit_assert(strcmp(target.str(), "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]") == 0);
        }
    }
}

#if 0
TEST_CUDA(05_nested) {

    def collatz(value: p.Int):
        counter = p.Int(0)
        loop = p.Loop(value, counter)
        while (loop.cond(ek.neq(value, 1))):
            is_even = ek.eq(value & 1, 0)
            value.assign(ek.select(is_even, value // 2, 3*value + 1))
            counter += 1
        return counter

    i = p.Int(1)
    buf = ek.full(p.Int, 1000, 16)
    ek.eval(buf)

    if variant == 0:
        loop_1 = p.Loop(i)
        while loop_1.cond(i <= 10):
            ek.scatter(buf, collatz(p.Int(i)), i - 1)
            i += 1
    else:
        for i in range(1, 11):
            ek.scatter(buf, collatz(p.Int(i)), i - 1)
            i += 1

    assert buf == p.Int(0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 1000, 1000, 1000, 1000, 1000, 1000)

}
#endif
