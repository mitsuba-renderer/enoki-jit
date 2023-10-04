#include "internal.h"
#include <stdint.h>
#include <set>

extern void jitc_vcall_set_self(JitBackend backend, uint32_t value, uint32_t index);
extern void jitc_vcall_self(JitBackend backend, uint32_t *value, uint32_t *index);

extern uint32_t jitc_var_loop_init(uint32_t *indices, uint32_t n_indices);

extern uint32_t jitc_var_vcall(const char *domain, uint32_t self, uint32_t mask,
                               uint32_t n_inst, const uint32_t *inst_id,
                               uint32_t n_in, const uint32_t *in,
                               uint32_t n_out_nested,
                               const uint32_t *out_nested,
                               const uint32_t *checkpoints, uint32_t *out);

extern void jitc_vcall_upload(ThreadState *ts);

extern VCallBucket *jitc_var_vcall_reduce(JitBackend backend,
                                          const char *domain, uint32_t index,
                                          uint32_t *bucket_count_out);

/// Helper data structure used to initialize the data block consumed by a vcall
struct VCallDataRecord {
    uint32_t offset;
    uint32_t size;
    const void *src;
};

using CallablesSet = std::set<XXH128_hash_t, XXH128Cmp>;

/// Encodes information about a virtual function call
struct VCall {
    JitBackend backend;

    /// A descriptive name
    char *name = nullptr;

    /// ID of call variable
    uint32_t id = 0;

    /// Max # of distinct instances that might be referenced by the call
    uint32_t n_inst = 0;

    /// Array of size 'n_inst' with all instance IDs that might be referenced
    std::vector<uint32_t> inst_id;

    /// Array of size 'n_inst' listing the associated callable hash
    std::vector<XXH128_hash_t> inst_hash;

    /// Input variables at call site
    std::vector<uint32_t> in;
    /// Input placeholder variables
    std::vector<uint32_t> in_nested;

    /// Output variables at call site
    std::vector<uint32_t> out;
    /// Output variables *per instance*
    std::vector<uint32_t> out_nested;

    /// Per-instance offsets into side effects list
    std::vector<uint32_t> checkpoints;
    /// Compressed side effect index list
    std::vector<uint32_t> side_effects;

    /// Mapping from variable index to offset into call data
    tsl::robin_map<uint64_t, uint32_t, UInt64Hasher> data_map;
    std::vector<uint32_t> data_offset;

    uint64_t *offset = nullptr;
    size_t offset_size = 0;

    /// Storage in bytes for inputs/outputs before simplifications
    uint32_t in_count_initial = 0;
    uint32_t in_size_initial = 0;
    uint32_t out_size_initial = 0;

    /// Does this vcall need self as argument
    bool use_self = false;

    /// Set of callable hashes that can be reached from this vcall
    CallablesSet callables_set = {};

    ~VCall() {
        for (uint32_t index : out_nested)
            jitc_var_dec_ref(index);
        clear_side_effects();
        free(name);
    }

    void clear_side_effects() {
        if (checkpoints.empty() || checkpoints.back() == checkpoints.front())
            return;
        for (uint32_t index : side_effects)
            jitc_var_dec_ref(index);
        side_effects.clear();
        std::fill(checkpoints.begin(), checkpoints.end(), 0);
    }
};

// Forward declarations
extern void jitc_var_vcall_assemble(VCall *vcall, uint32_t self_reg,
                                    uint32_t mask_reg, uint32_t offset_reg,
                                    uint32_t data_reg);


extern void jitc_var_vcall_assemble_cuda(VCall *vcall, uint32_t vcall_reg,
                                         uint32_t self_reg, uint32_t mask_reg,
                                         uint32_t offset_reg, uint32_t data_reg,
                                         uint32_t n_out, uint32_t in_size,
                                         uint32_t in_align, uint32_t out_size,
                                         uint32_t out_align);

extern void jitc_var_vcall_assemble_llvm(VCall *vcall, uint32_t vcall_reg,
                                         uint32_t self_reg, uint32_t mask_reg,
                                         uint32_t offset_reg, uint32_t data_reg,
                                         uint32_t n_out, uint32_t in_size,
                                         uint32_t in_align, uint32_t out_size,
                                         uint32_t out_align);
