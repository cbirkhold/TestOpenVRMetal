//
// MIT License
//
// Copyright (c) 2018 Chris Birkhold
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <OpenVR/OpenVR.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Watchdog.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

@import IOSurface;
@import Metal;
@import simd;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static_assert(vr::Eye_Left == 0, "!");
static_assert(vr::Eye_Right == 1, "!");

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    constexpr size_t NUM_EYES = 2;
    constexpr size_t MAX_INFLIGHT_FRAMES = 3;
    constexpr MTLPixelFormat EYE_TEXTURE_PIXEL_FORMAT = MTLPixelFormatBGRA8Unorm_sRGB;

    enum InitFramesMode_e {
        INIT_FRAMES_MODE_PER_EYE_TEXTURES,
        INIT_FRAMES_MODE_ARRAY_TEXTURES,
        INIT_FRAMES_MODE_IO_SURFACE_BACKED_TEXTURES,
    };

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------------
    // A list of tracked device poses large enough to hold the maximum.
    typedef std::array<vr::TrackedDevicePose_t, vr::k_unMaxTrackedDeviceCount> TrackedDevicePoses;

    //------------------------------------------------------------------------------
    // Per-eye information.
    typedef struct Eye_s {
        id<MTLTexture>                  m_metal_texture;
        NSUInteger                      m_metal_texture_slice;      // If the texture is an array texture.

        vr::Texture_t                   m_vr_texture;
    } Eye_t;

    //------------------------------------------------------------------------------
    // Per-frame information.
    typedef struct Frame_s {
        TrackedDevicePoses      m_render_poses;
        Eye_t                   m_eyes[NUM_EYES];
    } Frame_t;

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    std::mutex stdio_mutex;
    std::array<Frame_t, MAX_INFLIGHT_FRAMES> frames;

    id<MTLBuffer> metal_hidden_area_mesh[NUM_EYES];
    size_t metal_hidden_area_mesh_num_vertices[NUM_EYES];
    id<MTLRenderPipelineState> metal_draw_hidden_mesh;

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------------
    // This would ideally be part of OpenVR.
    const char* GetVRCompositorErrorAsEnglishDescription(vr::VRCompositorError error)
    {
        switch (error) {
            case vr::VRCompositorError_None:                         return "None (0)";
            case vr::VRCompositorError_RequestFailed:                return "RequestFailed (1)";
            case vr::VRCompositorError_IncompatibleVersion:          return "IncompatibleVersion (100)";
            case vr::VRCompositorError_DoNotHaveFocus:               return "DoNotHaveFocus (101)";
            case vr::VRCompositorError_InvalidTexture:               return "InvalidTexture (102)";
            case vr::VRCompositorError_IsNotSceneApplication:        return "IsNotSceneApplication (103)";
            case vr::VRCompositorError_TextureIsOnWrongDevice:       return "TextureIsOnWrongDevice (104)";
            case vr::VRCompositorError_TextureUsesUnsupportedFormat: return "TextureUsesUnsupportedFormat (105)";
            case vr::VRCompositorError_SharedTexturesNotSupported:   return "SharedTexturesNotSupported (106)";
            case vr::VRCompositorError_IndexOutOfRange:              return "IndexOutOfRange (107)";
            case vr::VRCompositorError_AlreadySubmitted:             return "AlreadySubmitted (108)";
            case vr::VRCompositorError_InvalidBounds:                return "InvalidBounds (109)";
            default:                                                 return "Unkown VRCompositorError";
        }
    }

    //------------------------------------------------------------------------------
    // Retrieve a tracked device property string as an std::string.
    std::string get_tracked_device_string(vr::IVRSystem* const system,
                                          vr::TrackedDeviceIndex_t device_index,
                                          vr::TrackedDeviceProperty property,
                                          vr::TrackedPropertyError* const error = nullptr)
    {
        assert(system);

        const uint32_t length = system->GetStringTrackedDeviceProperty(device_index, property, nullptr, 0, error);

        if (length == 0) {
            return std::string();
        }

        std::string result;
        result.resize((length - 1));        // 'length' includes terminator

        if (system->GetStringTrackedDeviceProperty(device_index, property, &result[0], length, error) != length) {
            assert(false);
        }

        return result;
    }

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    void export_hidden_area_outline(std::ofstream& mesh_csv,
                                    const vr::HiddenAreaMesh_t& mesh,
                                    vr::IVRSystem* const system)
    {
        vr::DistortionCoordinates_t xy;

        for (size_t i = 0; i < mesh.unTriangleCount; ++i) {
            mesh_csv << mesh.pVertexData[i].v[0] << "\t";
            mesh_csv << mesh.pVertexData[i].v[1] << "\t";

            if (not system->ComputeDistortion(vr::Eye_Left, mesh.pVertexData[i].v[0], mesh.pVertexData[i].v[1], &xy)) {
                xy = {};
            }

            mesh_csv << std::max(-1.0f, std::min(xy.rfRed[0]  , 2.0f)) << "\t";
            mesh_csv << std::max(-1.0f, std::min(xy.rfRed[1]  , 2.0f)) << "\t";
            mesh_csv << std::max(-1.0f, std::min(xy.rfGreen[0], 2.0f)) << "\t";
            mesh_csv << std::max(-1.0f, std::min(xy.rfGreen[1], 2.0f)) << "\t";
            mesh_csv << std::max(-1.0f, std::min(xy.rfBlue[0] , 2.0f)) << "\t";
            mesh_csv << std::max(-1.0f, std::min(xy.rfBlue[1] , 2.0f)) << "\n";
        }

        mesh_csv << mesh.pVertexData[0].v[0] << "\t";
        mesh_csv << mesh.pVertexData[0].v[1] << "\n";
    }

    //------------------------------------------------------------------------------
    // Export outlines of the hidden area meshes to a CSV file (X-Y scatter chart).
    void export_hidden_area_outline(const char* const path, vr::IVRSystem* const system)
    {
        std::ofstream mesh_csv(path, std::ios::out);

        mesh_csv << "Left Eye\n";

        const vr::HiddenAreaMesh_t mesh_left = system->GetHiddenAreaMesh(vr::Eye_Left , vr::k_eHiddenAreaMesh_LineLoop);
        export_hidden_area_outline(mesh_csv, mesh_left, system);

        mesh_csv << "Right Eye\n";

        const vr::HiddenAreaMesh_t mesh_right = system->GetHiddenAreaMesh(vr::Eye_Right, vr::k_eHiddenAreaMesh_LineLoop);
        export_hidden_area_outline(mesh_csv, mesh_right, system);
    }

    static void export_distortion_samples(std::ofstream& distortion_csv, size_t size, vr::EVREye eye, vr::IVRSystem *system)
    {
        vr::DistortionCoordinates_t xy;

        for (size_t y = 0; y < size; ++y) {
            const float v = (float(y) / float(size - 1));

            for (size_t x = 0; x < size; ++x) {
                const float u = (float(x) / float(size - 1));

                distortion_csv << u << "\t";
                distortion_csv << v << "\t";

                if (not system->ComputeDistortion(eye, u, v, &xy)) {
                    xy = {};
                }

                distortion_csv << std::max(-1.0f, std::min(xy.rfRed[0]  , 2.0f)) << "\t";
                distortion_csv << std::max(-1.0f, std::min(xy.rfRed[1]  , 2.0f)) << "\t";
                distortion_csv << std::max(-1.0f, std::min(xy.rfGreen[0], 2.0f)) << "\t";
                distortion_csv << std::max(-1.0f, std::min(xy.rfGreen[1], 2.0f)) << "\t";
                distortion_csv << std::max(-1.0f, std::min(xy.rfBlue[0] , 2.0f)) << "\t";
                distortion_csv << std::max(-1.0f, std::min(xy.rfBlue[1] , 2.0f)) << "\n";
            }
        }
    }

    //------------------------------------------------------------------------------
    // Export lens-distorted grids to a CSV file (X-Y scatter chart).
    void export_distortion_samples(const char* const path, vr::IVRSystem* const system)
    {
        constexpr size_t SIZE = (16 + 1);       // 16 x 16 quads

        std::ofstream distortion_csv(path, std::ios::out);

        distortion_csv << "Left Eye\n";
        export_distortion_samples(distortion_csv, SIZE, vr::Eye_Left , system);

        distortion_csv << "Right Eye\n";
        export_distortion_samples(distortion_csv, SIZE, vr::Eye_Right, system);
    }

    //------------------------------------------------------------------------------
    // Create a float2 vertex buffer for rendering the hidden area mesh of the given
    // eye as individual triangles.
    id<MTLBuffer> create_hidden_area_mesh(size_t& num_vertices,
                                          id<MTLDevice> device,
                                          id<MTLBlitCommandEncoder> blit_command_encoder,
                                          vr::IVRSystem* const system,
                                          vr::EVREye eye)
    {
        const vr::HiddenAreaMesh_t mesh = system->GetHiddenAreaMesh(eye, vr::k_eHiddenAreaMesh_Standard);

        num_vertices = (3 * mesh.unTriangleCount);
        const size_t buffer_length = (sizeof(simd_float2) * num_vertices);

        id<MTLBuffer> vertex_buffer = [device newBufferWithLength:buffer_length options:MTLResourceStorageModeShared];
        simd_float2* const vertices = static_cast<simd_float2*>(vertex_buffer.contents);

        for (size_t i = 0; i < (mesh.unTriangleCount * 3); ++i) {
            vertices[i] = simd_make_float2(mesh.pVertexData[i].v[0], mesh.pVertexData[i].v[1]);
        }

        id<MTLBuffer> hidden_mesh = [device newBufferWithLength:buffer_length options:MTLResourceStorageModePrivate];

        NSCAssert(blit_command_encoder.device == device, @"!");
        [blit_command_encoder copyFromBuffer:vertex_buffer sourceOffset:0 toBuffer:hidden_mesh destinationOffset:0 size:buffer_length];

        return hidden_mesh;
    }

    //------------------------------------------------------------------------------
    // Create a float2 vertex buffer for rendering a rectangular inset 'hidden area'
    // mesh of the given coverage as individual triangles.
    id<MTLBuffer> create_hidden_area_mesh(size_t& num_vertices,
                                          id<MTLDevice> device,
                                          id<MTLBlitCommandEncoder> blit_command_encoder,
                                          float coverage)
    {
        num_vertices = (3 * 8);
        const size_t buffer_length = (sizeof(simd_float2) * num_vertices);

        id<MTLBuffer> vertex_buffer = [device newBufferWithLength:buffer_length options:MTLResourceStorageModeShared];
        simd_float2* const vertices = static_cast<simd_float2*>(vertex_buffer.contents);

        const float coverage_2 = (coverage / 2.0f);
        const simd_float4 p = simd_make_float4(coverage_2, coverage_2, (1.0f - coverage_2), (1.0f - coverage_2));
        size_t vertex_index = 0;

        vertices[vertex_index++] = simd_make_float2(0.0, 0.0);
        vertices[vertex_index++] = simd_make_float2(p.x, p.y);
        vertices[vertex_index++] = simd_make_float2(1.0, 0.0);

        vertices[vertex_index++] = simd_make_float2(1.0, 0.0);
        vertices[vertex_index++] = simd_make_float2(p.x, p.y);
        vertices[vertex_index++] = simd_make_float2(p.z, p.y);

        vertices[vertex_index++] = simd_make_float2(1.0, 0.0);
        vertices[vertex_index++] = simd_make_float2(p.z, p.y);
        vertices[vertex_index++] = simd_make_float2(1.0, 1.0);

        vertices[vertex_index++] = simd_make_float2(1.0, 1.0);
        vertices[vertex_index++] = simd_make_float2(p.z, p.y);
        vertices[vertex_index++] = simd_make_float2(p.z, p.w);

        vertices[vertex_index++] = simd_make_float2(1.0, 1.0);
        vertices[vertex_index++] = simd_make_float2(p.z, p.w);
        vertices[vertex_index++] = simd_make_float2(0.0, 1.0);

        vertices[vertex_index++] = simd_make_float2(0.0, 1.0);
        vertices[vertex_index++] = simd_make_float2(p.z, p.w);
        vertices[vertex_index++] = simd_make_float2(p.x, p.w);

        vertices[vertex_index++] = simd_make_float2(0.0, 1.0);
        vertices[vertex_index++] = simd_make_float2(p.x, p.w);
        vertices[vertex_index++] = simd_make_float2(0.0, 0.0);

        vertices[vertex_index++] = simd_make_float2(0.0, 0.0);
        vertices[vertex_index++] = simd_make_float2(p.x, p.w);
        vertices[vertex_index++] = simd_make_float2(p.x, p.y);

        id<MTLBuffer> hidden_mesh = [device newBufferWithLength:buffer_length options:MTLResourceStorageModePrivate];

        NSCAssert(blit_command_encoder.device == device, @"!");
        [blit_command_encoder copyFromBuffer:vertex_buffer sourceOffset:0 toBuffer:hidden_mesh destinationOffset:0 size:buffer_length];

        return hidden_mesh;
    }

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    MTLTextureDescriptor* texture_desc_for_io_surface_backed_eye_texture(const std::array<size_t, 2>& size)
    {
        MTLTextureDescriptor* const texture_desc = [MTLTextureDescriptor new];
        {
            texture_desc.textureType = MTLTextureType2D;
            texture_desc.pixelFormat = EYE_TEXTURE_PIXEL_FORMAT;

            texture_desc.width  = size[0];
            texture_desc.height = size[1];

            texture_desc.storageMode = MTLStorageModeManaged;
            texture_desc.usage = (MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget);
        }

        return texture_desc;
    }

    MTLTextureDescriptor* texture_desc_for_eye_texture(const std::array<size_t, 2>& size, bool array, bool multisample)
    {
        MTLTextureDescriptor* const texture_desc = [MTLTextureDescriptor new];
        {
            if (multisample) {
                if (array) {
#if (__MAC_OS_X_VERSION_MIN_REQUIRED < 101400)
                    {
                        std::unique_lock<std::mutex> lock(stdio_mutex);
                        std::cerr << "Error: MTLTextureType2DMultisampleArray is not available!" << std::endl;
                    }

                    return nil;
#else // __MAC_OS_X_VERSION_MIN_REQUIRED
                    texture_desc.textureType = MTLTextureType2DMultisampleArray;
#endif // __MAC_OS_X_VERSION_MIN_REQUIRED
                }
                else {
                    texture_desc.textureType = MTLTextureType2DMultisample;
                }
            }
            else {
                if (array) {
                    texture_desc.textureType = MTLTextureType2DArray;
                }
                else {
                    texture_desc.textureType = MTLTextureType2D;
                }
            }

            texture_desc.pixelFormat = EYE_TEXTURE_PIXEL_FORMAT;
            texture_desc.width  = size[0];
            texture_desc.height = size[1];

            if (multisample) {
                texture_desc.sampleCount = 4;
            }

            if (array) {
                texture_desc.arrayLength = 2;
            }

            texture_desc.storageMode = MTLStorageModePrivate;
            texture_desc.usage = (MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget);
        }

        return texture_desc;
    }

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------------
    // Wrapper for WaitGetPoses() checking for errors and timeouts (which in a
    // carefully scheduled sequence like VR rendering are indicative of errors).
    bool wait_get_poses(vr::IVRCompositor* const compositor, Frame_t& frame)
    {
        Watchdog::marker("WaitGetPoses", 100);

        const vr::EVRCompositorError error = compositor->WaitGetPoses(frame.m_render_poses.data(), uint32_t(frame.m_render_poses.size()), nullptr, 0);

        if (Watchdog::reset_marker() == Watchdog::MARKER_RESULT_PREVIOUS_MARKER_EXPIRED) {
            return false;
        }

        if (error != vr::VRCompositorError_None) {
            {
                std::unique_lock<std::mutex> lock(stdio_mutex);
                std::cerr << "Error: WaitGetPoses failed: " << GetVRCompositorErrorAsEnglishDescription(error) << std::endl;
            }

            return false;
        }

        return true;
    }

    //------------------------------------------------------------------------------
    // Wrapper for Submit() checking for errors and timeouts (which in a carefully
    // scheduled sequence like VR rendering are indicative of errors).
    bool submit(vr::IVRCompositor* const compositor, const Frame_t& frame, vr::EVRSubmitFlags submit_flags)
    {
        bool result = true;

        const vr::VRTextureBounds_t bounds = { .uMin = 0.0, .vMin = 0.0, .uMax = 1.0, .vMax = 1.0 };
        vr::EVRCompositorError error = vr::VRCompositorError_None;

        for (size_t i = 0; i < 2; ++i) {
            Watchdog::marker("Submit", 2);

            error = compositor->Submit(vr::EVREye(i), &frame.m_eyes[i].m_vr_texture, &bounds, submit_flags);

            if (Watchdog::reset_marker() == Watchdog::MARKER_RESULT_PREVIOUS_MARKER_EXPIRED) {
                return false;
            }

            if (error != vr::VRCompositorError_None) {
                {
                    std::unique_lock<std::mutex> lock(stdio_mutex);
                    std::cerr << "Error: Submit failed: " << (i == 0 ? "left": "right") << " eye: ";
                    std::cerr << GetVRCompositorErrorAsEnglishDescription(error) << std::endl;
                }

                result = false;
            }
        }

        return result;
    }

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------------
    // Initialize frames with eye textures.
    bool init_frames(id<MTLDevice> device,
                     MTLTextureDescriptor* const eye_texture_desc,
                     vr::EColorSpace color_space,
                     InitFramesMode_e mode)
    {
        NSDictionary* io_surface_properties = nil;

        if (mode == INIT_FRAMES_MODE_IO_SURFACE_BACKED_TEXTURES) {
            io_surface_properties = @{
                                      (id)kIOSurfaceWidth           : @(eye_texture_desc.width),
                                      (id)kIOSurfaceHeight          : @(eye_texture_desc.height),
                                      (id)kIOSurfaceBytesPerElement : @(4),
                                      (id)kIOSurfaceIsGlobal        : @YES,                     // Deprecated but required for SteamVR (as of 2018-08-25)
                                      (id)kIOSurfacePixelFormat     : @(k32RGBAPixelFormat)     // k32BGRAPixelFormat
                                      };
        }

        for (size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
            Frame_t& frame = frames[frame_index];

            for (size_t j = 0; j < frame.m_render_poses.size(); ++j) {
                frame.m_render_poses[j].bPoseIsValid = false;
            }

            id<MTLTexture> array_texture = nil;

            if (mode == INIT_FRAMES_MODE_ARRAY_TEXTURES) {
#if (__MAC_OS_X_VERSION_MIN_REQUIRED < 101400)
                {
                    std::unique_lock<std::mutex> lock(stdio_mutex);
                    std::cerr << "Error: newSharedTextureWithDescriptor is not available!" << std::endl;
                }

                return false;
#else // __MAC_OS_X_VERSION_MIN_REQUIRED
                array_texture = [device newSharedTextureWithDescriptor:eye_texture_desc];
                array_texture.label = [NSString stringWithFormat:@"Frame %ju left/right", frame_index];
#endif // __MAC_OS_X_VERSION_MIN_REQUIRED
            }

            for (size_t eye_index = 0; eye_index < NUM_EYES; ++eye_index) {
                Eye_t& eye = frame.m_eyes[eye_index];

                if (mode == INIT_FRAMES_MODE_ARRAY_TEXTURES) {
                    eye.m_metal_texture = array_texture;
                    eye.m_metal_texture_slice = eye_index;
                }
                else {
                    if (mode == INIT_FRAMES_MODE_IO_SURFACE_BACKED_TEXTURES) {
                        const IOSurfaceRef io_surface_ref = IOSurfaceCreate((__bridge CFDictionaryRef)io_surface_properties);
                        eye.m_metal_texture = [device newTextureWithDescriptor:eye_texture_desc iosurface:io_surface_ref plane:0];
                        CFRelease(io_surface_ref);
                    }
                    else {
#if (__MAC_OS_X_VERSION_MIN_REQUIRED < 101400)
                        {
                            std::unique_lock<std::mutex> lock(stdio_mutex);
                            std::cerr << "Error: newSharedTextureWithDescriptor is not available!" << std::endl;
                        }

                        return false;
#else // __MAC_OS_X_VERSION_MIN_REQUIRED
                        eye.m_metal_texture = [device newSharedTextureWithDescriptor:eye_texture_desc];
#endif // __MAC_OS_X_VERSION_MIN_REQUIRED
                    }

                    eye.m_metal_texture.label = [NSString stringWithFormat:@"Frame %ju %s", frame_index, (eye_index == 0 ? "left" : "right")];
                    eye.m_metal_texture_slice = 0;
                }

                if (mode == INIT_FRAMES_MODE_IO_SURFACE_BACKED_TEXTURES) {
                    eye.m_vr_texture.handle = reinterpret_cast<void*>(eye.m_metal_texture.iosurface);
                    eye.m_vr_texture.eType = vr::TextureType_IOSurface;
                }
                else {
                    eye.m_vr_texture.handle = reinterpret_cast<void*>(uintptr_t(eye.m_metal_texture));
                    eye.m_vr_texture.eType = vr::TextureType_Metal;
                }

                eye.m_vr_texture.eColorSpace = color_space;
            }
        }

        return true;
    }

    //------------------------------------------------------------------------------
    // Render the given number of frames.
    bool run(id<MTLDevice> device, size_t num_frames, vr::EVRSubmitFlags submit_flags)
    {
        id<MTLCommandQueue> command_queue = [device newCommandQueue];
        vr::IVRCompositor* const compositor = vr::VRCompositor();

        //------------------------------------------------------------------------------
        // Start a capture scope.
        MTLCaptureManager* capture_manager = [MTLCaptureManager sharedCaptureManager];

        id<MTLCaptureScope> capture_scope = [capture_manager newCaptureScopeWithCommandQueue:command_queue];
        {
            capture_scope.label = @"Frame";
        }

        //------------------------------------------------------------------------------
        // Prepare a render pass descriptor (texture/slice being set below).
        MTLRenderPassDescriptor* const render_pass_desc = [MTLRenderPassDescriptor renderPassDescriptor];

        MTLRenderPassColorAttachmentDescriptor* const render_pass_color_attachment = render_pass_desc.colorAttachments[0];
        {
            render_pass_color_attachment.loadAction = MTLLoadActionClear;
            render_pass_color_attachment.storeAction = MTLStoreActionStore;
        }

        //------------------------------------------------------------------------------
        // Render frames.
        std::chrono::time_point<std::chrono::steady_clock> prev_frame_timestamp = std::chrono::steady_clock::now();

        for (size_t frame_index = 0; frame_index < num_frames; ++frame_index) {
            @autoreleasepool {
                Frame_t& frame = frames[frame_index % frames.size()];

                //------------------------------------------------------------------------------
                // Synchronize with the HMD or pace non-HMD rendering (to ~90 FPS).
                if (compositor) {
                    if (not wait_get_poses(compositor, frame)) {
                        return false;
                    }
                }
                else {
                    std::this_thread::sleep_until(prev_frame_timestamp + std::chrono::milliseconds(11));
                    prev_frame_timestamp = std::chrono::steady_clock::now();
                }

                //------------------------------------------------------------------------------
                // Render eye textures.
                [capture_scope beginScope];
                id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];

                for (size_t eye_index = 0; eye_index < NUM_EYES; ++eye_index) {
                    Eye_t& eye = frame.m_eyes[eye_index];

                    render_pass_color_attachment.texture = eye.m_metal_texture;
                    render_pass_color_attachment.slice = eye.m_metal_texture_slice;

                    if (eye_index == 0) {
                        render_pass_color_attachment.clearColor = MTLClearColorMake(1.0, 1.0, 0.0, 1.0);
                    }
                    else {
                        render_pass_color_attachment.clearColor = MTLClearColorMake(0.0, 1.0, 1.0, 1.0);
                    }

                    id<MTLRenderCommandEncoder> render_command_encoder = [command_buffer renderCommandEncoderWithDescriptor:render_pass_desc];

                    [render_command_encoder setRenderPipelineState:metal_draw_hidden_mesh];
                    [render_command_encoder setVertexBuffer:metal_hidden_area_mesh[eye_index] offset:0 atIndex:0];
                    [render_command_encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:metal_hidden_area_mesh_num_vertices[eye_index]];

                    [render_command_encoder endEncoding];
                }

                [command_buffer commit];
                [capture_scope endScope];

                //------------------------------------------------------------------------------
                // Submit this frame and make sure eye-texture rendering is scheduled before
                // letting the compositor schedule its work in the next call to WaitGetPose().
                if (compositor) {
                    if (not submit(compositor, frame, submit_flags)) {
                        return false;
                    }

                    [command_buffer waitUntilScheduled];
                }
            }
        }

        return true;
    }

    //------------------------------------------------------------------------------
    // Create render pipeline states.
    bool init_render_pipeline_states(id<MTLLibrary> library, MTLTextureDescriptor* const eye_texture_desc)
    {
        MTLRenderPipelineDescriptor* const draw_hidden_mesh = [MTLRenderPipelineDescriptor new];
        {
            draw_hidden_mesh.label = @"Hidden Mesh";

            draw_hidden_mesh.vertexFunction = [library newFunctionWithName:@"float2_uv_to_ndc_v"];
            draw_hidden_mesh.fragmentFunction = [library newFunctionWithName:@"hidden_mesh_f"];

            draw_hidden_mesh.sampleCount = eye_texture_desc.sampleCount;

            MTLRenderPipelineColorAttachmentDescriptor* const color_attachment = draw_hidden_mesh.colorAttachments[0];
            {
                color_attachment.pixelFormat = eye_texture_desc.pixelFormat;
            }
        }

        NSError* error = nil;
        metal_draw_hidden_mesh = [library.device newRenderPipelineStateWithDescriptor:draw_hidden_mesh error:&error];

        if (error || (not metal_draw_hidden_mesh)) {
            {
                std::unique_lock<std::mutex> lock(stdio_mutex);
                std::cerr << "Error: Failed to create '" << draw_hidden_mesh.label.UTF8String << "'render pipeline state!" << std::endl;
            }

            return false;
        }

        return true;
    }

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------------
    // Render frames to IOSurface backed Metal textures.
    bool render_to_io_surface_texture(id<MTLLibrary> library, size_t num_frames, const std::array<size_t, 2>& size, vr::EColorSpace color_space, vr::EVRSubmitFlags submit_flags)
    {
        std::cout << "Info: Rendering " << num_frames << " frames using per-eye IOSurface backed Metal textures" << '\n';

        MTLTextureDescriptor* const eye_texture_desc = texture_desc_for_io_surface_backed_eye_texture(size);

        if (not init_frames(library.device, eye_texture_desc, color_space, INIT_FRAMES_MODE_IO_SURFACE_BACKED_TEXTURES)) {
            return false;
        }

        if (not init_render_pipeline_states(library, eye_texture_desc)) {
            return false;
        }

        return run(library.device, num_frames, submit_flags);
    }

    //------------------------------------------------------------------------------
    // Render frames to Metal shared textures.
    bool render_to_shared_texture(id<MTLLibrary> library, size_t num_frames, const std::array<size_t, 2>& size, vr::EColorSpace color_space, vr::EVRSubmitFlags submit_flags)
    {
        std::cout << "Info: Rendering " << num_frames << " frames using per-eye Metal shared textures" << '\n';

        constexpr bool array       = false;
        constexpr bool multisample = false;

        MTLTextureDescriptor* const eye_texture_desc = texture_desc_for_eye_texture(size, array, multisample);

        if (not init_frames(library.device, eye_texture_desc, color_space, INIT_FRAMES_MODE_PER_EYE_TEXTURES)) {
            return false;
        }

        if (not init_render_pipeline_states(library, eye_texture_desc)) {
            return false;
        }

        return run(library.device, num_frames, submit_flags);
    }

    //------------------------------------------------------------------------------
    // Render frames to Metal shared array textures.
    bool render_to_shared_texture_array(id<MTLLibrary> library, size_t num_frames, const std::array<size_t, 2>& size, vr::EColorSpace color_space, vr::EVRSubmitFlags submit_flags)
    {
        std::cout << "Info: Rendering " << num_frames << " frames using a Metal shared array texture" << '\n';

        constexpr bool array       = true;
        constexpr bool multisample = false;

        MTLTextureDescriptor* const eye_texture_desc = texture_desc_for_eye_texture(size, array, multisample);

        if (not init_frames(library.device, eye_texture_desc, color_space, INIT_FRAMES_MODE_ARRAY_TEXTURES)) {
            return false;
        }

        if (not init_render_pipeline_states(library, eye_texture_desc)) {
            return false;
        }

        return run(library.device, num_frames, submit_flags);
    }

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

} // unnamed namespace

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    constexpr bool REQUIRE_HMD = false;         // We can still inspect Metal rendering without an HMD
    constexpr bool DO_NOT_USE_HMD = false;      // Optionally ingore the presence of an HMD

    //> process handle -s0 -p1 SIGINT

    std::cout << "Info: pid: " << getpid() << std::endl;

    //------------------------------------------------------------------------------
    // Setup watchdog.
    Watchdog::set_cerr_mutex([]() { stdio_mutex.lock(); }, []() { stdio_mutex.unlock(); });
    Watchdog::set_cout_mutex([]() { stdio_mutex.lock(); }, []() { stdio_mutex.unlock(); });

    //------------------------------------------------------------------------------
    // Check if runtime is installed.
    const bool is_runtime_installed = vr::VR_IsRuntimeInstalled();

    if (REQUIRE_HMD && (not is_runtime_installed)) {
        std::cout << "Error: OpenVR Runtime is NOT installed!" << std::endl;
        return EXIT_FAILURE;
    }

    //------------------------------------------------------------------------------
    // Check if HMD is present.
    const bool is_hmd_present = ((not DO_NOT_USE_HMD) && vr::VR_IsHmdPresent());

    if (REQUIRE_HMD && (not is_hmd_present)) {
        std::cout << "Error: HMD is NOT present!" << std::endl;
        return EXIT_FAILURE;
    }

    //------------------------------------------------------------------------------
    // Initialize VR system.
    vr::IVRSystem* vr_system = nullptr;
    std::array<size_t, 2> recommended_render_target_size = {};

    if (is_hmd_present) {
        vr::EVRInitError error = vr::VRInitError_None;
        vr_system = vr::VR_Init(&error, vr::VRApplication_Scene);

        if ((error != vr::VRInitError_None) || not vr_system) {
            std::unique_lock<std::mutex> lock(stdio_mutex);
            std::cerr << "Error: Failed to initialize VR system: " << vr::VR_GetVRInitErrorAsEnglishDescription(error) << "!" << std::endl;
            return EXIT_FAILURE;
        }

        atexit([]() {
            vr::VR_Shutdown();
        });

        //------------------------------------------------------------------------------
        // Keep log a bit cleaner by giving the VR sytstem a chance to complete its
        // asynchronous startup before we go on.
        std::this_thread::sleep_for(std::chrono::seconds(2));

        //------------------------------------------------------------------------------
        // Grab a few bits of VR system info.
        const std::string tracking_system_name = get_tracked_device_string(vr_system, vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_TrackingSystemName_String);
        const std::string model_number         = get_tracked_device_string(vr_system, vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_ModelNumber_String       );
        const std::string serial_number        = get_tracked_device_string(vr_system, vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_SerialNumber_String      );
        const std::string manufacturer_name    = get_tracked_device_string(vr_system, vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_ManufacturerName_String  );
        const float       display_frequency    = vr_system->GetFloatTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_DisplayFrequency_Float, nullptr);

        std::cout << "Info: " << manufacturer_name << ' ' << model_number << " (" << display_frequency << " Hz, " << tracking_system_name << ' ' << serial_number << ")\n";

        vr_system->GetRecommendedRenderTargetSize((uint32_t*)&recommended_render_target_size[0], (uint32_t*)&recommended_render_target_size[1]);

        std::cout << "Info: Recommended render target size is " << recommended_render_target_size[0] << " x " << recommended_render_target_size[1] << '\n';

        //------------------------------------------------------------------------------
        // Always show chaperon for this test.
        if ((1)) {
            vr::VRChaperone()->ForceBoundsVisible(true);
        }

        //------------------------------------------------------------------------------
        // Export hidden mesh and/or distortion infor for analysis.
        if ((0)) {
            export_hidden_area_outline("/Users/Shared/hidden_area_mesh_outline.csv", vr_system);
        }

        if ((0)) {
            export_distortion_samples("/Users/Shared/distortion_samples.csv", vr_system);
        }
    }
    else {
        recommended_render_target_size[0] = 512;
        recommended_render_target_size[1] = 512;
    }

    //------------------------------------------------------------------------------
    // Setup Metal using the same device as used by the VR system.
    id<MTLDevice> metal_device = nil;

    if (is_hmd_present) {
        static_assert(sizeof(id<MTLDevice>) == sizeof(uint64_t), "!");
        vr_system->GetOutputDevice(reinterpret_cast<uint64_t*>(uintptr_t(&metal_device)), vr::TextureType_IOSurface);
    }
    else {
        metal_device = MTLCreateSystemDefaultDevice();
    }

    if (not metal_device) {
        {
            std::unique_lock<std::mutex> lock(stdio_mutex);
            std::cerr << "Error: Failed to acquire Metal device from VR system!" << std::endl;
        }

        return EXIT_FAILURE;
    }

    std::cout << "Info: Using Metal device " << metal_device.name.UTF8String << '\n';

    //------------------------------------------------------------------------------
    // Create basic Metal objects.
    id<MTLLibrary> metal_default_library = [metal_device newDefaultLibrary];

    if (not metal_default_library) {
        {
            std::unique_lock<std::mutex> lock(stdio_mutex);
            std::cerr << "Error: Failed to load Metal default library!" << std::endl;
        }

        return EXIT_FAILURE;
    }

    //------------------------------------------------------------------------------
    // Create meshes.
    @autoreleasepool {
        id<MTLCommandQueue> command_queue = [metal_device newCommandQueue];
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLBlitCommandEncoder> blit_command_encoder = [command_buffer blitCommandEncoder];

        if (is_hmd_present) {
            metal_hidden_area_mesh[0] = create_hidden_area_mesh(metal_hidden_area_mesh_num_vertices[0], metal_device, blit_command_encoder, vr_system, vr::Eye_Left );
            metal_hidden_area_mesh[1] = create_hidden_area_mesh(metal_hidden_area_mesh_num_vertices[1], metal_device, blit_command_encoder, vr_system, vr::Eye_Right);
        }
        else {
            metal_hidden_area_mesh[0] = create_hidden_area_mesh(metal_hidden_area_mesh_num_vertices[0], metal_device, blit_command_encoder, 0.25);

            metal_hidden_area_mesh[1] = metal_hidden_area_mesh[0];
            metal_hidden_area_mesh_num_vertices[1] = metal_hidden_area_mesh_num_vertices[0];
        }

        [blit_command_encoder endEncoding];
        [command_buffer commit];
    }

    //------------------------------------------------------------------------------
    // Run tests.
    const size_t NUM_FRAMES = (90 * 10);

    if ((1)) {
        @autoreleasepool {
            render_to_io_surface_texture(metal_default_library, NUM_FRAMES, recommended_render_target_size, vr::ColorSpace_Gamma, vr::Submit_LensDistortionAlreadyApplied);
        }
    }

    if ((0)) {
        @autoreleasepool {
            render_to_shared_texture(metal_default_library, NUM_FRAMES, recommended_render_target_size, vr::ColorSpace_Gamma, vr::Submit_LensDistortionAlreadyApplied);
        }
    }

    if ((0)) {
        @autoreleasepool {
            render_to_shared_texture_array(metal_default_library, NUM_FRAMES, recommended_render_target_size, vr::ColorSpace_Gamma, vr::Submit_LensDistortionAlreadyApplied);
        }
    }

    //------------------------------------------------------------------------------
    // ...
    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
