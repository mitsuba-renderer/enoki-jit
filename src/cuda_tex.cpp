#include "cuda_tex.h"
#include "internal.h"
#include "log.h"
#include "var.h"
#include "op.h"
#include "eval.h"
#include <string.h>
#include <memory>
#include <atomic>

using StagingAreaDeleter = void (*)(void *);

struct DrJitCudaTexture {
    size_t type_size;  /// Size in bytes of underlying storage type e.g float32, float16
    size_t n_channels; /// Total number of channels
    size_t n_textures; /// Number of texture objects
    std::atomic_size_t n_referenced_textures; /// Number of referenced textures
    std::unique_ptr<CUtexObject[]> textures; /// Array of CUDA texture objects
    std::unique_ptr<uint32_t[]> indices; /// Array of indices of texture object pointers for the JIT
    std::unique_ptr<CUarray[]> arrays; /// Array of CUDA arrays

    /**
     * \brief Construct a texture object for the given number of channels.
     *
     * This constructor only allocates the necessary memory for all array
     * struct variables and defines the numbers of CUDA textures to be used for
     * the given number of channels.
     */
    DrJitCudaTexture(size_t type_size, size_t n_channels)
        : type_size(type_size), n_channels(n_channels), n_textures(1 + ((n_channels - 1) / 4)),
          n_referenced_textures(n_textures),
          textures(std::make_unique<CUtexObject[]>(n_textures)),
          indices(std::make_unique<uint32_t[]>(n_textures)),
          arrays(std::make_unique<CUarray[]>(n_textures)) {}

    /**
     * \brief Returns the number of channels from the original data that are
     * associated with the texture object at the given index in \ref textures
     */
    size_t channels(size_t index) const {
        if (index >= n_textures)
            jitc_raise("DrJitCudaTexture::channels(): invalid texture index!");

        size_t tex_channels = 4;
        if (index == n_textures - 1) {
            tex_channels = n_channels % 4;
            if (tex_channels == 0)
                tex_channels = 4;
        }

        return tex_channels;
    }

    /**
     * \brief Returns the number of channels (always either 1, 2 or 4) used
     * internally by the texture object at the given index in \ref textures
     */
    size_t channels_internal(size_t index) const {
        const size_t channels_raw = channels(index);
        return (channels_raw == 3) ? 4 : channels_raw;
    }

    /**
     * \brief Releases the texture object at the given index in \ref textures
     * and returns whether or not there is at least one texture that is still
     * not released.
     */
    bool release_texture(size_t index) {
        if (state.backends & (uint32_t) JitBackend::CUDA) {
            // Only run the following code if the CUDA context is still alive
            ThreadState *ts = thread_state(JitBackend::CUDA);
            scoped_set_context guard(ts->context);
            cuda_check(cuArrayDestroy(arrays[index]));
            cuda_check(cuTexObjectDestroy(textures[index]));
        }

        return --n_referenced_textures > 0;
    }

    /**
     * \brief Returns a JIT pointer to the ith texture object.
     *
     * We map the texture to a JIT variable, which can then be revered to by a
     * pointer. This allows us to traverse the texture for frozen functions.
     *
     * This function returns an owning reference, which has to be released.
     */
    uint32_t get_jit_pointer(uint32_t i) {
        return jitc_var_pointer(JitBackend::CUDA, jitc_var(indices[i])->data,
                                indices[i], false);
    }
};

struct TextureReleasePayload {
    DrJitCudaTexture* texture;
    size_t index;
};

void *jitc_cuda_tex_create(size_t ndim, const size_t *shape, size_t n_channels,
                           int format, int filter_mode, int wrap_mode) {
    if (ndim < 1 || ndim > 3)
        jitc_raise("jit_cuda_tex_create(): invalid texture dimension!");
    else if (n_channels == 0)
        jitc_raise("jit_cuda_tex_create(): must have at least 1 channel!");

    ThreadState *ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);

    CUDA_RESOURCE_DESC res_desc;
    memset(&res_desc, 0, sizeof(CUDA_RESOURCE_DESC));
    res_desc.resType = CU_RESOURCE_TYPE_ARRAY;

    CUDA_TEXTURE_DESC tex_desc;
    memset(&tex_desc, 0, sizeof(CUDA_TEXTURE_DESC));
    switch (filter_mode) {
        case 0:
            tex_desc.filterMode = tex_desc.mipmapFilterMode =
                CU_TR_FILTER_MODE_POINT;
            break;
        case 1:
            tex_desc.filterMode = tex_desc.mipmapFilterMode =
                CU_TR_FILTER_MODE_LINEAR;
            break;
        default:
            jitc_raise("jit_cuda_tex_create(): invalid filter mode!");
            break;
    }
    tex_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;
    for (size_t i = 0; i < 3; ++i) {
        switch (wrap_mode) {
            case 0:
                tex_desc.addressMode[i] = CU_TR_ADDRESS_MODE_WRAP;
                break;
            case 1:
                tex_desc.addressMode[i] = CU_TR_ADDRESS_MODE_CLAMP;
                break;
            case 2:
                tex_desc.addressMode[i] = CU_TR_ADDRESS_MODE_MIRROR;
                break;
            default:
                jitc_raise("jit_cuda_tex_create(): invalid wrap mode!");
                break;
        }
    }
    tex_desc.maxAnisotropy = 1;

    CUDA_RESOURCE_VIEW_DESC view_desc;
    memset(&view_desc, 0, sizeof(CUDA_RESOURCE_VIEW_DESC));
    view_desc.width = shape[0];
    view_desc.height = (ndim >= 2) ? shape[1] : 1;
    view_desc.depth = (ndim == 3) ? shape[2] : 0;

    size_t storage_format = 0;
    size_t tsize = 0;

    switch (format) {
        case 0:
            storage_format = CU_AD_FORMAT_FLOAT;
            tsize = sizeof(float);
            break;
        case 1:
            storage_format = CU_AD_FORMAT_HALF;
            tsize = sizeof(uint16_t);
            break;
        default:
            jitc_raise("jit_cuda_tex_create(): invalid data type!");
            break;
    };

    DrJitCudaTexture *texture = new DrJitCudaTexture(tsize, n_channels);
    for (size_t tex = 0; tex < texture->n_textures; ++tex) {
        const size_t tex_channels = texture->channels_internal(tex);

        CUarray array = nullptr;
        if (ndim == 1 || ndim == 2) {
            CUDA_ARRAY_DESCRIPTOR array_desc;
            memset(&array_desc, 0, sizeof(CUDA_ARRAY_DESCRIPTOR));
            array_desc.Width = shape[0];
            array_desc.Height = (ndim == 2) ? shape[1] : 1;
            array_desc.Format = (CUarray_format) storage_format;
            array_desc.NumChannels = (unsigned int) tex_channels;
            cuda_check(cuArrayCreate(&array, &array_desc));
        } else {
            CUDA_ARRAY3D_DESCRIPTOR array_desc;
            memset(&array_desc, 0, sizeof(CUDA_ARRAY3D_DESCRIPTOR));
            array_desc.Width = shape[0];
            array_desc.Height = shape[1];
            array_desc.Depth = shape[2];
            array_desc.Format = (CUarray_format) storage_format;
            array_desc.NumChannels = (unsigned int) tex_channels;
            cuda_check(cuArray3DCreate(&array, &array_desc));
        }

        res_desc.res.array.hArray = array;
        texture->arrays[tex] = array;

        if (tex_channels == 1)
            view_desc.format = format == 0 ? 
                CU_RES_VIEW_FORMAT_FLOAT_1X32 : CU_RES_VIEW_FORMAT_FLOAT_1X16;
        else if (tex_channels == 2)
            view_desc.format = format == 0 ?
                CU_RES_VIEW_FORMAT_FLOAT_2X32 : CU_RES_VIEW_FORMAT_FLOAT_2X16;
        else
            view_desc.format = format == 0 ?
                CU_RES_VIEW_FORMAT_FLOAT_4X32 : CU_RES_VIEW_FORMAT_FLOAT_4X16;

        cuda_check(cuTexObjectCreate(&(texture->textures[tex]), &res_desc,
                                     &tex_desc, &view_desc));

        // texture->indices[tex] = jitc_var_pointer(
        //     JitBackend::CUDA, (void *) texture->textures[tex], 0, 0);
        texture->indices[tex] =
            jitc_var_mem_map(JitBackend::CUDA, VarType::UInt64,
                             (void *) texture->textures[tex], 1, false);

        TextureReleasePayload *payload_ptr =
            new TextureReleasePayload({ texture, tex });

        jitc_var_set_callback(
            texture->indices[tex],
            [](uint32_t /* index */, int free, void *callback_data) {
                if (free) {
                    TextureReleasePayload& payload =
                        *((TextureReleasePayload *) callback_data);

                    DrJitCudaTexture *texture = payload.texture;
                    size_t tex = payload.index;

                    if (!texture->release_texture(tex))
                        delete texture;

                    delete &payload;
                }
            },
            (void *) payload_ptr,
            false
        );
    }

    jitc_log(LogLevel::Debug, "jitc_cuda_tex_create(): " DRJIT_PTR,
             (uintptr_t) texture);

    return (void *) texture;
}

void jitc_cuda_tex_get_shape(size_t ndim, const void *texture_handle,
                             size_t *shape) {
    if (ndim < 1 || ndim > 3)
        jitc_raise("jit_cuda_tex_get_shape(): invalid texture dimension!");

    ThreadState *ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);

    DrJitCudaTexture &texture = *((DrJitCudaTexture *) texture_handle);

    CUDA_ARRAY3D_DESCRIPTOR array_desc;
    // cuArray3DGetDescriptor can also be called on 1D and 2D arrays
    cuda_check(cuArray3DGetDescriptor(&array_desc, texture.arrays[0]));

    shape[0] = array_desc.Width;
    if (ndim >= 2)
        shape[1] = array_desc.Height;
    if (ndim == 3)
        shape[2] = array_desc.Depth;
    shape[ndim] = texture.n_channels;
}

void jitc_cuda_tex_get_indices(const void *texture_handle, uint32_t *indices) {
    if (!texture_handle)
        return;

    DrJitCudaTexture &texture = *((DrJitCudaTexture *) texture_handle);

    for (uint32_t i = 0; i < texture.n_textures; i++) {
        indices[i] = texture.indices[i];
    }
}

static std::unique_ptr<void, StagingAreaDeleter>
jitc_cuda_tex_alloc_staging_area(size_t n_texels,
                                 const DrJitCudaTexture &texture) {
    // Each texture except for the last one will need exactly 4 channels
    size_t staging_area_size =
        texture.type_size * n_texels *
        ((texture.n_textures - 1) * 4 +
         texture.channels_internal(texture.n_textures - 1));
    void *staging_area = jitc_malloc(AllocType::Device, staging_area_size);

    return std::unique_ptr<void, StagingAreaDeleter>(staging_area, jitc_free);
}

/*
 * \brief Copy texture to a staging area that is pitched over the channels, such
 * that the data for each underlying CUDA texture is partitioned.
 */
static void jitc_cuda_tex_memcpy_d2s(
    ThreadState *ts,
    const std::unique_ptr<void, StagingAreaDeleter> &staging_area,
    size_t n_texels, const void *src_ptr, const DrJitCudaTexture &dst_texture) {
    scoped_set_context guard(ts->context);

    size_t texel_size = dst_texture.n_channels * dst_texture.type_size;

    CUDA_MEMCPY2D op;
    memset(&op, 0, sizeof(CUDA_MEMCPY2D));

    for (size_t tex = 0; tex < dst_texture.n_textures; ++tex) {
        size_t texel_offset = tex * 4 * dst_texture.type_size;

        op.srcXInBytes = texel_offset;
        op.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        op.srcDevice = (CUdeviceptr) src_ptr;
        op.srcPitch = texel_size;

        op.dstXInBytes = n_texels * texel_offset;
        op.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        op.dstDevice = (CUdeviceptr) staging_area.get();
        op.dstPitch = dst_texture.channels_internal(tex) * dst_texture.type_size;

        op.WidthInBytes = dst_texture.channels(tex) * dst_texture.type_size;
        op.Height = n_texels;

        cuda_check(cuMemcpy2DAsync(&op, ts->stream));
    }
}

void jitc_cuda_tex_memcpy_d2t(size_t ndim, const size_t *shape,
                              const void *src_ptr, void *dst_texture_handle) {
    if (ndim < 1 || ndim > 3)
        jitc_raise("jit_cuda_tex_memcpy(): invalid texture dimension!");

    ThreadState *ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);

    DrJitCudaTexture &dst_texture = *((DrJitCudaTexture *) dst_texture_handle);

    size_t n_texels = shape[0];
    for (size_t dim = 1; dim < ndim; ++dim)
        n_texels *= shape[dim];

    StagingAreaDeleter noop = [](void *) {};
    std::unique_ptr<void, StagingAreaDeleter> staging_area(nullptr, noop);
    bool needs_staging_area =
        (dst_texture.n_textures > 1) || (dst_texture.n_channels == 3);
    if (needs_staging_area) {
        staging_area = jitc_cuda_tex_alloc_staging_area(n_texels, dst_texture);
        jitc_cuda_tex_memcpy_d2s(ts, staging_area, n_texels, src_ptr,
                                 dst_texture);
    }

    if (ndim == 1 || ndim == 2) {
        CUDA_MEMCPY2D op;
        memset(&op, 0, sizeof(CUDA_MEMCPY2D));

        for (size_t tex = 0; tex < dst_texture.n_textures; ++tex) {
            size_t pitch =
                shape[0] * dst_texture.channels_internal(tex) * dst_texture.type_size;

            op.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            op.srcDevice = (CUdeviceptr) src_ptr;
            op.srcPitch = pitch;
            if (needs_staging_area) {
                op.srcDevice = (CUdeviceptr) staging_area.get();
                op.srcXInBytes = tex * n_texels * 4 * dst_texture.type_size;
            }

            op.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            op.dstArray = dst_texture.arrays[tex];

            op.WidthInBytes = pitch;
            op.Height = (ndim == 2) ? shape[1] : 1;

            cuda_check(cuMemcpy2DAsync(&op, ts->stream));
        }
    } else {
        CUDA_MEMCPY3D op;
        memset(&op, 0, sizeof(CUDA_MEMCPY2D));

        for (size_t tex = 0; tex < dst_texture.n_textures; ++tex) {
            size_t pitch =
                shape[0] * dst_texture.channels_internal(tex) * dst_texture.type_size;

            op.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            op.srcDevice = (CUdeviceptr) src_ptr;
            op.srcPitch = pitch;
            op.srcHeight = shape[1];
            if (needs_staging_area) {
                op.srcXInBytes = tex * n_texels * 4 * dst_texture.type_size;
                op.srcDevice = (CUdeviceptr) staging_area.get();
            }

            op.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            op.dstArray = dst_texture.arrays[tex];

            op.WidthInBytes = pitch;
            op.Height = shape[1];
            op.Depth = shape[2];

            cuda_check(cuMemcpy3DAsync(&op, ts->stream));
        }
    }
}

/*
 * \brief Copy texture from a staging area that is pitched over the channels,
 * such that the data for each underlying CUDA texture is partitioned.
 */
static void jitc_cuda_tex_memcpy_s2d(
    ThreadState *ts,
    const std::unique_ptr<void, StagingAreaDeleter> &staging_area,
    size_t n_texels, const void *dst_ptr, const DrJitCudaTexture &src_texture) {
    scoped_set_context guard(ts->context);

    size_t texel_size = src_texture.n_channels * src_texture.type_size;

    CUDA_MEMCPY2D op;
    memset(&op, 0, sizeof(CUDA_MEMCPY2D));

    for (size_t tex = 0; tex < src_texture.n_textures; ++tex) {
        size_t texel_offset = tex * 4 * src_texture.type_size;

        op.srcXInBytes = n_texels * texel_offset;
        op.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        op.srcDevice = (CUdeviceptr) staging_area.get();
        op.srcPitch = src_texture.channels_internal(tex) * src_texture.type_size;

        op.dstXInBytes = texel_offset;
        op.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        op.dstDevice = (CUdeviceptr) dst_ptr;
        op.dstPitch = texel_size;

        op.WidthInBytes = src_texture.channels(tex) * src_texture.type_size;
        op.Height = n_texels;

        cuda_check(cuMemcpy2DAsync(&op, ts->stream));
    }
}

void jitc_cuda_tex_memcpy_t2d(size_t ndim, const size_t *shape,
                              const void *src_texture_handle, void *dst_ptr) {
    if (ndim < 1 || ndim > 3)
        jitc_raise("jit_cuda_tex_memcpy(): invalid texture dimension!");

    ThreadState *ts = thread_state(JitBackend::CUDA);
    scoped_set_context guard(ts->context);

    DrJitCudaTexture &src_texture = *((DrJitCudaTexture *) src_texture_handle);

    size_t n_texels = shape[0];
    for (size_t dim = 1; dim < ndim; ++dim)
        n_texels *= shape[dim];

    auto noop = [](void *) {};
    std::unique_ptr<void, StagingAreaDeleter> staging_area(nullptr, noop);
    bool needs_staging_area =
        (src_texture.n_textures > 1) || (src_texture.n_channels == 3);
    if (needs_staging_area) {
        staging_area = jitc_cuda_tex_alloc_staging_area(n_texels, src_texture);
    }

    if (ndim == 1 || ndim == 2) {
        CUDA_MEMCPY2D op;
        memset(&op, 0, sizeof(CUDA_MEMCPY2D));

        for (size_t tex = 0; tex < src_texture.n_textures; ++tex) {
            size_t pitch =
                shape[0] * src_texture.channels_internal(tex) * src_texture.type_size;

            op.srcMemoryType = CU_MEMORYTYPE_ARRAY;
            op.srcArray = src_texture.arrays[tex];

            op.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            op.dstDevice = (CUdeviceptr) dst_ptr;
            op.dstPitch = pitch;
            if (needs_staging_area) {
                op.dstXInBytes = tex * n_texels * 4 * src_texture.type_size;
                op.dstDevice = (CUdeviceptr) staging_area.get();
            }

            op.WidthInBytes = pitch;
            op.Height = (ndim == 2) ? shape[1] : 1;

            cuda_check(cuMemcpy2DAsync(&op, ts->stream));
        }
    } else {
        CUDA_MEMCPY3D op;
        memset(&op, 0, sizeof(CUDA_MEMCPY3D));

        for (size_t tex = 0; tex < src_texture.n_textures; ++tex) {
            size_t pitch =
                shape[0] * src_texture.channels_internal(tex) * src_texture.type_size;

            op.srcMemoryType = CU_MEMORYTYPE_ARRAY;
            op.srcArray = src_texture.arrays[tex];

            op.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            op.dstDevice = (CUdeviceptr) dst_ptr;
            op.dstPitch = pitch;
            op.dstHeight = shape[1];
            if (needs_staging_area) {
                op.dstXInBytes = tex * n_texels * 4 * src_texture.type_size;
                op.dstDevice = (CUdeviceptr) staging_area.get();
            }

            op.WidthInBytes = pitch;
            op.Height = shape[1];
            op.Depth = shape[2];

            cuda_check(cuMemcpy3DAsync(&op, ts->stream));
        }
    }

    if (needs_staging_area)
        jitc_cuda_tex_memcpy_s2d(ts, staging_area, n_texels, dst_ptr,
                                 src_texture);
}

Variable jitc_cuda_tex_check(VarType out_type, size_t ndim, const uint32_t *pos) {
    // Validate input types, determine size of the operation
    uint32_t size = 0;
    bool dirty = false, symbolic = false;
    JitBackend backend = JitBackend::None;

    if (ndim < 1 || ndim > 3)
        jitc_raise("jit_cuda_tex_check(): invalid texture dimension!");

    for (size_t i = 0; i < ndim; ++i) {
        const Variable *v = jitc_var(pos[i]);
        if ((VarType) v->type != VarType::Float32)
            jitc_raise("jit_cuda_tex_check(): type mismatch for arg. %zu (got "
                       "%s, expected %s)", i, type_name[v->type],
                       type_name[(int) VarType::Float32]);
        size = std::max(size, v->size);
        dirty |= v->is_dirty();
        symbolic |= (bool) v->symbolic;
        backend = (JitBackend) v->backend;
    }

    for (uint32_t i = 0; i < ndim; ++i) {
        const Variable *v = jitc_var(pos[i]);
        if (v->size != 1 && v->size != size)
            jitc_raise("jit_cuda_tex_check(): arithmetic involving arrays of "
                       "incompatible size!");
    }

    if (dirty) {
        jitc_eval(thread_state(backend));
        for (size_t i = 0; i < ndim; ++i) {
            if (jitc_var(pos[i])->is_dirty())
                jitc_raise_dirty_error(pos[i]);
        }
    }

    Variable v;
    v.size = size;
    v.backend = (uint32_t) backend;
    v.symbolic = symbolic;
    v.type = (uint32_t) out_type;
    return v;
}

void jitc_cuda_tex_lookup(size_t ndim, const void *texture_handle,
                          const uint32_t *pos, uint32_t active, uint32_t *out) {
    DrJitCudaTexture &tex = *((DrJitCudaTexture *) texture_handle);
    VarType out_type = VarType::Float32;
    Variable v = jitc_cuda_tex_check(out_type, ndim, pos);

    for (size_t ti = 0; ti < tex.n_textures; ++ti) {
        // Perform a fetch per texture ..
        v.kind = (uint32_t) VarKind::TexLookup;
        memset(v.dep, 0, sizeof(v.dep));
        const Variable *active_v = jitc_var(active);
        if (active_v->is_literal() && active_v->literal == 1) {
            v.dep[0] = tex.get_jit_pointer(ti);
            v.literal = 0;
        } else {
            v.literal = 1; // encode a masked operation
            uint64_t zero_i = 0;
            Ref zero = steal(jitc_var_literal((JitBackend) v.backend, VarType::Pointer, &zero_i, 1, 0));
            uint32_t pointer = tex.get_jit_pointer(ti);
            v.dep[0]         = jitc_var_select(active, pointer, zero);
            jitc_var_dec_ref(pointer);
        }
        for (size_t j = 0; j < ndim; ++j) {
            v.dep[j + 1] = pos[j];
            jitc_var_inc_ref(pos[j]);
        }
        Ref tex_load = steal(jitc_var_new(v));

        // .. and then extract components
        v.kind = (uint32_t) VarKind::Extract;
        memset(v.dep, 0, sizeof(v.dep));
        for (size_t ch = 0; ch < tex.channels(ti); ++ch) {
            v.literal = (uint64_t) ch;
            v.dep[0] = tex_load;
            jitc_var_inc_ref(tex_load);
            *out++ = jitc_var_new(v);
        }
    }
}

void jitc_cuda_tex_bilerp_fetch(size_t ndim, const void *texture_handle,
                                const uint32_t *pos, uint32_t active,
                                uint32_t *out) {
    if (ndim != 2)
        jitc_raise("jitc_cuda_tex_bilerp_fetch(): only 2D textures are supported!");

    DrJitCudaTexture &tex = *((DrJitCudaTexture *) texture_handle);
    VarType out_type = VarType::Float32;
    Variable v = jitc_cuda_tex_check(out_type, ndim, pos);

    for (size_t ti = 0; ti < tex.n_textures; ++ti) {
        for (size_t ch = 0; ch < tex.channels(ti); ++ch) {
            // Perform a fetch per texture and channel..
            v.kind = (uint32_t) VarKind::TexFetchBilerp;
            v.literal = ch;
            memset(v.dep, 0, sizeof(v.dep));
            v.dep[0] = tex.get_jit_pointer(ti);
            v.dep[1] = active;
            jitc_var_inc_ref(active);
            for (size_t j = 0; j < ndim; ++j) {
                v.dep[j + 2] = pos[j];
                jitc_var_inc_ref(pos[j]);
            }
            Ref tex_load = steal(jitc_var_new(v));

            memset(v.dep, 0, sizeof(v.dep));
            v.kind = (uint32_t) VarKind::Extract;
            for (uint32_t j = 0; j < 4; ++j) {
                // .. and then extract components
                v.literal = (uint64_t) j;
                v.dep[0] = tex_load;
                jitc_var_inc_ref(tex_load);
                *out++ = jitc_var_new(v);
            }
        }
    }
}

void jitc_cuda_tex_destroy(void *texture_handle) {
    if (!texture_handle)
        return;

    jitc_log(LogLevel::Debug, "jitc_cuda_tex_destroy(" DRJIT_PTR ")",
             (uintptr_t) texture_handle);

    DrJitCudaTexture *texture = (DrJitCudaTexture *) texture_handle;

    /* The `texture` struct can potentially be deleted when decreasing the
       reference count of the individual textures. We must hoist the number
       of textures out of the loop condition. */
    const size_t n_textures = texture->n_textures;
    for (size_t tex = 0; tex < n_textures; ++tex)
        jitc_var_dec_ref(texture->indices[tex]);
}
