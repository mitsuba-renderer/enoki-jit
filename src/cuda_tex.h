#pragma once

#include "cuda_api.h"

extern void *jitc_cuda_tex_create(size_t ndim, const size_t *shape,
                                  size_t n_channels, int format,
                                  int filter_mode, int wrap_mode);
extern void jitc_cuda_tex_get_shape(size_t ndim, const void *texture_handle,
                                    size_t *shape);
extern void jitc_cuda_tex_get_indices(const void *texture_handle,
                                      uint32_t *indices);
extern void jitc_cuda_tex_memcpy_d2t(size_t ndim, const size_t *shape,
                                     const void *src_ptr,
                                     void *dst_texture_handle);
extern void jitc_cuda_tex_memcpy_t2d(size_t ndim, const size_t *shape,
                                     const void *src_texture_handle,
                                     void *dst_ptr);
extern void jitc_cuda_tex_lookup(size_t ndim, const void *texture_handle,
                                 const uint32_t *pos, uint32_t active,
                                 uint32_t *out);
extern void jitc_cuda_tex_bilerp_fetch(size_t ndim, const void *texture_handle,
                                       const uint32_t *pos, uint32_t active,
                                       uint32_t *out);
extern void jitc_cuda_tex_destroy(void *texture_handle);
