#include "internal.h"


struct LLVMThreadState: ThreadState{
    
    /// Fill a device memory region with constants of a given type
    void jitc_memset_async(void *ptr, uint32_t size,
                                   uint32_t isize, const void *src) override;

    /// Perform a synchronous copy operation
    void jitc_memcpy(void *dst, const void *src, size_t size) override;
    
    ~LLVMThreadState(){}
};
