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
#include <map>
#include <string>
#include <thread>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MetalUtils.h"
#include "OpenVRMetal.h"
#include "OpenVRUtils.h"
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
    constexpr MTLPixelFormat EYE_TEXTURE_PIXEL_FORMAT = MTLPixelFormatBGRA8Unorm;

    enum InitFrameMode_e {
        INIT_FRAME_MODE_PER_EYE_TEXTURES,
        INIT_FRAME_MODE_ARRAY_TEXTURE,
        INIT_FRAME_MODE_IO_SURFACE_BACKED_TEXTURES,
    };

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------------
    // A list of tracked device poses large enough to hold the maximum.
    typedef std::array<vr::TrackedDevicePose_t, vr::k_unMaxTrackedDeviceCount> TrackedDevicePoses;

    //------------------------------------------------------------------------------
    // Per-eye information.
    typedef struct Eye_s {
        id<MTLTexture>      m_metal_texture;
        NSUInteger          m_metal_texture_slice;      // If the texture is an array texture.

        vr::Texture_t       m_vr_texture;
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
    Frame_t frame;

    id<MTLTexture> depth_attachment_texture;
    id<MTLTexture> stencil_attachment_texture;

    std::unique_ptr<vr::metal::HiddenAreaMesh> metal_hidden_area_mesh[NUM_EYES];
    id<MTLRenderPipelineState> metal_draw_hidden_area_mesh;
    id<MTLDepthStencilState> invert_stencil_ignore_depth;

    id<MTLBuffer> cube_index_buffer;
    id<MTLRenderPipelineState> metal_draw_cubes;
    id<MTLDepthStencilState> skip_non_zero_stencil_or_greater_depth;

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
                std::cerr << "Error: WaitGetPoses failed: " << OpenVRUtils::compositor_error_as_english_description(error) << std::endl;
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
                    std::cerr << OpenVRUtils::compositor_error_as_english_description(error) << std::endl;
                }

                result = false;
            }
        }

        return result;
    }

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

#if DEBUG_PROJETION_MATRIX

    simd_float4x4 matrix_perspective_right_hand(float fovy, float aspect, float near, float far)
    {
        float ys = (1.0f / tanf(fovy * 0.5f));
        float xs = (ys / aspect);
        float zs = (far / (near - far));
        float zn = (near * zs);

        return {{
            {  xs, 0.0, 0.0,  0.0 },
            { 0.0,  ys, 0.0,  0.0 },
            { 0.0, 0.0,  zs, -1.0 },
            { 0.0, 0.0,  zn,  0.0 }
        }};
    }

#endif // DEBUG_PROJETION_MATRIX

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------------
    // Initialize frames with eye textures.
    bool init_frame(id<MTLDevice> device,
                    MTLTextureDescriptor* const eye_texture_desc,
                    vr::EColorSpace color_space,
                    InitFrameMode_e mode)
    {
        for (size_t j = 0; j < frame.m_render_poses.size(); ++j) {
            frame.m_render_poses[j].bPoseIsValid = false;
        }

        id<MTLTexture> array_texture = nil;

        if (mode == INIT_FRAME_MODE_ARRAY_TEXTURE) {
#if (__MAC_OS_X_VERSION_MIN_REQUIRED < 101400)
            {
                std::unique_lock<std::mutex> lock(stdio_mutex);
                std::cerr << "Error: newSharedTextureWithDescriptor is not available!" << std::endl;
            }

            return false;
#else // __MAC_OS_X_VERSION_MIN_REQUIRED
            array_texture = [device newSharedTextureWithDescriptor:eye_texture_desc];
            array_texture.label = @"Eye texture left/right";
#endif // ..., __MAC_OS_X_VERSION_MIN_REQUIRED
        }

        for (size_t eye_index = 0; eye_index < NUM_EYES; ++eye_index) {
            Eye_t& eye = frame.m_eyes[eye_index];

            if (mode == INIT_FRAME_MODE_ARRAY_TEXTURE) {
                eye.m_metal_texture = array_texture;
                eye.m_metal_texture_slice = eye_index;
            }
            else {
                if (mode == INIT_FRAME_MODE_IO_SURFACE_BACKED_TEXTURES) {
                    eye.m_metal_texture = vr::metal::Utils::new_io_surface_backed_eye_texture(device, eye_texture_desc);
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
#endif // ..., __MAC_OS_X_VERSION_MIN_REQUIRED
                }

                eye.m_metal_texture.label = [NSString stringWithFormat:@"Eye texture %s", (eye_index == 0 ? "left" : "right")];
                eye.m_metal_texture_slice = 0;
            }

            if (mode == INIT_FRAME_MODE_IO_SURFACE_BACKED_TEXTURES) {
                eye.m_vr_texture.handle = reinterpret_cast<void*>(eye.m_metal_texture.iosurface);
                eye.m_vr_texture.eType = vr::TextureType_IOSurface;
            }
            else {
                eye.m_vr_texture.handle = reinterpret_cast<void*>(uintptr_t(eye.m_metal_texture));
                eye.m_vr_texture.eType = vr::TextureType_Metal;
            }

            eye.m_vr_texture.eColorSpace = color_space;
        }
        return true;
    }

    //------------------------------------------------------------------------------
    // Render the given number of frames.
    bool run(id<MTLDevice> device, size_t num_frames, vr::EVRSubmitFlags submit_flags)
    {
        id<MTLCommandQueue> command_queue = [device newCommandQueue];
        vr::IVRSystem* const system = vr::VRSystem();
        vr::IVRCompositor* const compositor = vr::VRCompositor();

        //------------------------------------------------------------------------------
        // Grab per-eye projection matrices (these are constant per device).
        simd::float4x4 projection_matrices[2] = { 1.0f, 1.0f };

        if (system) {
            constexpr float near_z = 0.1;
            constexpr float far_z = 100.0;

            for (size_t eye_index = 0; eye_index < NUM_EYES; ++eye_index) {
                projection_matrices[eye_index] = vr::metal::Utils::simd_from_hmd_matrix(system->GetProjectionMatrix(vr::EVREye(eye_index), near_z, far_z));
            }
        }

        //------------------------------------------------------------------------------
        // Create a capture scope.
        MTLCaptureManager* capture_manager = [MTLCaptureManager sharedCaptureManager];

        id<MTLCaptureScope> capture_scope = [capture_manager newCaptureScopeWithCommandQueue:command_queue];
        {
            capture_scope.label = @"Frame";
        }

        //------------------------------------------------------------------------------
        // Prepare a render pass descriptor (texture/slice being set below).
        MTLRenderPassDescriptor* const render_pass_desc = [MTLRenderPassDescriptor renderPassDescriptor];
        {
            if (depth_attachment_texture) {
                render_pass_desc.depthAttachment.loadAction = MTLLoadActionClear;
                render_pass_desc.depthAttachment.storeAction = MTLStoreActionDontCare;
                render_pass_desc.depthAttachment.texture = depth_attachment_texture;
                render_pass_desc.depthAttachment.clearDepth = 1.0;
            }

            if (stencil_attachment_texture) {
                render_pass_desc.stencilAttachment.loadAction = MTLLoadActionClear;
                render_pass_desc.stencilAttachment.storeAction = MTLStoreActionDontCare;
                render_pass_desc.stencilAttachment.texture = stencil_attachment_texture;
                render_pass_desc.stencilAttachment.clearStencil = 0;
            }
        }

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
                // Grab per-eye transform matrices (these may change at runtime with the IPD).
                simd::float4x4 eye_to_head_transforms[2] = { 1.0f, 1.0f };

                if (system) {
                    for (size_t eye_index = 0; eye_index < NUM_EYES; ++eye_index) {
                        eye_to_head_transforms[eye_index] = vr::metal::Utils::simd_from_hmd_matrix(system->GetEyeToHeadTransform(vr::EVREye(eye_index)));
                    }
                }

                //------------------------------------------------------------------------------
                // Check IPD.
                static simd_float4 prev_eye_to_head_translation[2];
                bool ipd_changed = false;

                for (size_t eye_index = 0; eye_index < NUM_EYES; ++eye_index) {
                    if (not simd_equal(eye_to_head_transforms[eye_index].columns[3], prev_eye_to_head_translation[eye_index])) {
                        ipd_changed = true;
                    }

                    prev_eye_to_head_translation[eye_index] = eye_to_head_transforms[eye_index].columns[3];
                }

                if (ipd_changed) {
                    const float ipd = (fabsf(prev_eye_to_head_translation[0].x - prev_eye_to_head_translation[1].x) * 1000.0f);
                    std::cout << "Info: IPD changed to " << ipd << " [mm]\n";
                }

                //------------------------------------------------------------------------------
                // Read tracking information.
                simd::float4x4 hmd_pose(1.0f);

                if (system) {
                    const vr::TrackedDeviceIndex_t tracked_device_index_left = system->GetTrackedDeviceIndexForControllerRole(vr::TrackedControllerRole_LeftHand);
                    const vr::TrackedDeviceIndex_t tracked_device_index_right = system->GetTrackedDeviceIndexForControllerRole(vr::TrackedControllerRole_RightHand);

                    vr::HmdMatrix34_t device_to_absolute_tracking_hmd;
                    bool device_to_absolute_tracking_hmd_is_valid = false;

                    vr::HmdMatrix34_t device_to_absolute_tracking_left;
                    bool device_to_absolute_tracking_left_is_valid = false;

                    vr::HmdMatrix34_t device_to_absolute_tracking_right;
                    bool device_to_absolute_tracking_right_is_valid = false;

                    for (size_t tracked_device_index = 0; tracked_device_index < frame.m_render_poses.size(); ++tracked_device_index) {
                        const vr::TrackedDevicePose_t& render_pose = frame.m_render_poses[tracked_device_index];

                        if (not render_pose.bDeviceIsConnected || not render_pose.bPoseIsValid) {
                            continue;
                        }

                        if (render_pose.eTrackingResult != vr::TrackingResult_Running_OK) {
                            continue;
                        }

                        if (tracked_device_index == vr::k_unTrackedDeviceIndex_Hmd) {
                            hmd_pose = vr::metal::Utils::simd_from_hmd_matrix(render_pose.mDeviceToAbsoluteTracking);
                            device_to_absolute_tracking_hmd = render_pose.mDeviceToAbsoluteTracking;
                            device_to_absolute_tracking_hmd_is_valid = true;
                        }
                        else if (tracked_device_index == tracked_device_index_left) {
                            device_to_absolute_tracking_left = render_pose.mDeviceToAbsoluteTracking;
                            device_to_absolute_tracking_left_is_valid = true;
                        }
                        else if (tracked_device_index == tracked_device_index_right) {
                            device_to_absolute_tracking_right = render_pose.mDeviceToAbsoluteTracking;
                            device_to_absolute_tracking_right_is_valid = true;
                        }
                    }
                }

                //------------------------------------------------------------------------------
                // Combine transforms into a single per-eye view-projection matrix.
                simd::float4x4 eye_transforms[2] = { 1.0f, 1.0f };

                for (size_t eye_index = 0; eye_index < NUM_EYES; ++eye_index) {
                    eye_transforms[eye_index] = simd_mul(simd_mul(projection_matrices[eye_index], simd_inverse(eye_to_head_transforms[eye_index])), simd_inverse(hmd_pose));
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
                        render_pass_color_attachment.clearColor = MTLClearColorMake(0.25, 0.25, 0.25, 1.0);
                    }
                    else {
                        render_pass_color_attachment.clearColor = MTLClearColorMake(0.25, 0.25, 0.25, 1.0);
                    }

                    id<MTLRenderCommandEncoder> render_command_encoder = [command_buffer renderCommandEncoderWithDescriptor:render_pass_desc];

                    [render_command_encoder setRenderPipelineState:metal_draw_hidden_area_mesh];
                    [render_command_encoder setDepthStencilState:invert_stencil_ignore_depth];
                    [render_command_encoder setStencilReferenceValue:0];

                    metal_hidden_area_mesh[eye_index]->draw_primitives(render_command_encoder, 0);

                    simd::float4x4 transforms[4] = { 1.0f, 1.0f, 1.0f, 1.0f };

                    transforms[0].columns[3] = vector4(-0.75f, 1.75f, 0.0f, 1.0f);
                    transforms[1].columns[3] = vector4( 0.75f, 1.75f, 0.0f, 1.0f);
                    transforms[2].columns[3] = vector4(-0.75f, 0.5f , 0.0f, 1.0f);
                    transforms[3].columns[3] = vector4( 0.75f, 0.5f , 0.0f, 1.0f);

#if DEBUG_PROJETION_MATRIX
                    const float aspect = (float(eye.m_metal_texture.width) / float(eye.m_metal_texture.height));
                    const simd::float4x4 projection_matrix = matrix_perspective_right_hand((65.0f * M_PI / 180.0f), aspect, 0.1f, 100.0f);
#else // DEBUG_PROJETION_MATRIX
                    const simd::float4x4& projection_matrix = eye_transforms[eye_index];
#endif // ..., DEBUG_PROJETION_MATRIX

                    [render_command_encoder setRenderPipelineState:metal_draw_cubes];
                    [render_command_encoder setDepthStencilState:skip_non_zero_stencil_or_greater_depth];
                    [render_command_encoder setStencilReferenceValue:0];

                    [render_command_encoder setVertexBytes:&projection_matrix length:sizeof(projection_matrix) atIndex:0];
                    [render_command_encoder setVertexBytes:transforms length:sizeof(transforms) atIndex:1];
                    [render_command_encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangleStrip indexCount:((8 * 2) + 1) indexType:MTLIndexTypeUInt32 indexBuffer:cube_index_buffer indexBufferOffset:0 instanceCount:4];

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
        NSError* error = nil;

        {
            MTLRenderPipelineDescriptor* const draw_hidden_mesh = [MTLRenderPipelineDescriptor new];
            {
                draw_hidden_mesh.label = @"Hidden Mesh";

                draw_hidden_mesh.vertexFunction = [library newFunctionWithName:@"float2_uv_to_ndc_v"];
                draw_hidden_mesh.fragmentFunction = [library newFunctionWithName:@"constant_color_f"];

                draw_hidden_mesh.sampleCount = eye_texture_desc.sampleCount;

                MTLRenderPipelineColorAttachmentDescriptor* const color_attachment = draw_hidden_mesh.colorAttachments[0];
                {
                    color_attachment.pixelFormat = eye_texture_desc.pixelFormat;
                }

                if (depth_attachment_texture) {
                    draw_hidden_mesh.depthAttachmentPixelFormat = depth_attachment_texture.pixelFormat;
                }

                if (stencil_attachment_texture) {
                    draw_hidden_mesh.stencilAttachmentPixelFormat = stencil_attachment_texture.pixelFormat;
                }
            }

            metal_draw_hidden_area_mesh = [library.device newRenderPipelineStateWithDescriptor:draw_hidden_mesh error:&error];

            if (error || (not metal_draw_hidden_area_mesh)) {
                {
                    std::unique_lock<std::mutex> lock(stdio_mutex);
                    std::cerr << "Error: Failed to create '" << draw_hidden_mesh.label.UTF8String << "' render pipeline state: " << error.localizedDescription.UTF8String << std::endl;
                }

                return false;
            }

            MTLDepthStencilDescriptor* const depth_stencil_desc = [MTLDepthStencilDescriptor new];
            {
                if (stencil_attachment_texture) {
                    depth_stencil_desc.frontFaceStencil = [MTLStencilDescriptor new];
                    {
                        depth_stencil_desc.frontFaceStencil.depthStencilPassOperation = MTLStencilOperationInvert;
                    }

                    depth_stencil_desc.backFaceStencil = depth_stencil_desc.frontFaceStencil;
                }

                depth_stencil_desc.label = @"invert_stencil_ignore_depth";
            }

            invert_stencil_ignore_depth = [library.device newDepthStencilStateWithDescriptor:depth_stencil_desc];
        }

        {
            MTLRenderPipelineDescriptor* const draw_cubes = [MTLRenderPipelineDescriptor new];
            {
                draw_cubes.label = @"Cubes";

                draw_cubes.vertexFunction = [library newFunctionWithName:@"instanced_cubes_v"];
                draw_cubes.fragmentFunction = [library newFunctionWithName:@"uvw_color_f"];

                draw_cubes.sampleCount = eye_texture_desc.sampleCount;

                MTLRenderPipelineColorAttachmentDescriptor* const color_attachment = draw_cubes.colorAttachments[0];
                {
                    color_attachment.pixelFormat = eye_texture_desc.pixelFormat;
                }

                if (depth_attachment_texture) {
                    draw_cubes.depthAttachmentPixelFormat = depth_attachment_texture.pixelFormat;
                }

                if (stencil_attachment_texture) {
                    draw_cubes.stencilAttachmentPixelFormat = stencil_attachment_texture.pixelFormat;
                }
            }

            metal_draw_cubes = [library.device newRenderPipelineStateWithDescriptor:draw_cubes error:&error];

            if (error || (not metal_draw_cubes)) {
                {
                    std::unique_lock<std::mutex> lock(stdio_mutex);
                    std::cerr << "Error: Failed to create '" << draw_cubes.label.UTF8String << "' render pipeline state: " << error.localizedDescription.UTF8String << std::endl;
                }

                return false;
            }

            MTLDepthStencilDescriptor* const depth_stencil_desc = [MTLDepthStencilDescriptor new];
            {
                if (depth_attachment_texture) {
                    depth_stencil_desc.depthCompareFunction = MTLCompareFunctionLess;
                    depth_stencil_desc.depthWriteEnabled = YES;
                }

                if (stencil_attachment_texture) {
                    depth_stencil_desc.frontFaceStencil = [MTLStencilDescriptor new];
                    {
                        depth_stencil_desc.frontFaceStencil.stencilCompareFunction = MTLCompareFunctionEqual;
                    }

                    depth_stencil_desc.backFaceStencil = depth_stencil_desc.frontFaceStencil;
                }

                depth_stencil_desc.label = @"skip_non_zero_stencil_or_greater_depth";
            }

            skip_non_zero_stencil_or_greater_depth = [library.device newDepthStencilStateWithDescriptor:depth_stencil_desc];
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

        MTLTextureDescriptor* const eye_texture_desc = vr::metal::Utils::new_texture_desc_for_io_surface_backed_eye_texture(EYE_TEXTURE_PIXEL_FORMAT, size[0], size[1]);

        if (not init_frame(library.device, eye_texture_desc, color_space, INIT_FRAME_MODE_IO_SURFACE_BACKED_TEXTURES)) {
            return false;
        }

        if (not init_render_pipeline_states(library, eye_texture_desc)) {
            return false;
        }

        return run(library.device, num_frames, submit_flags);
    }

    //------------------------------------------------------------------------------
    // Render frames to Metal shared textures.
#if (__MAC_OS_X_VERSION_MIN_REQUIRED >= 101400)
    bool render_to_shared_texture(id<MTLLibrary> library, size_t num_frames, const std::array<size_t, 2>& size, vr::EColorSpace color_space, vr::EVRSubmitFlags submit_flags)
    {
        std::cout << "Info: Rendering " << num_frames << " frames using per-eye Metal shared textures" << '\n';

        constexpr bool       array        = false;
        constexpr NSUInteger sample_count = 1;

        MTLTextureDescriptor* const eye_texture_desc = vr::metal::Utils::new_texture_desc_for_eye_texture(EYE_TEXTURE_PIXEL_FORMAT, size[0], size[1], array, sample_count);

        if (not init_frame(library.device, eye_texture_desc, color_space, INIT_FRAME_MODE_PER_EYE_TEXTURES)) {
            return false;
        }

        if (not init_render_pipeline_states(library, eye_texture_desc)) {
            return false;
        }

        return run(library.device, num_frames, submit_flags);
    }
#endif // __MAC_OS_X_VERSION_MIN_REQUIRED

    //------------------------------------------------------------------------------
    // Render frames to Metal shared array textures.
#if (__MAC_OS_X_VERSION_MIN_REQUIRED >= 101400)
    bool render_to_shared_texture_array(id<MTLLibrary> library, size_t num_frames, const std::array<size_t, 2>& size, vr::EColorSpace color_space, vr::EVRSubmitFlags submit_flags)
    {
        std::cout << "Info: Rendering " << num_frames << " frames using a Metal shared array texture" << '\n';

        constexpr bool       array        = true;
        constexpr NSUInteger sample_count = 1;

        MTLTextureDescriptor* const eye_texture_desc = vr::metal::Utils::new_texture_desc_for_eye_texture(EYE_TEXTURE_PIXEL_FORMAT, size[0], size[1], array, sample_count);

        if (not init_frame(library.device, eye_texture_desc, color_space, INIT_FRAME_MODE_ARRAY_TEXTURE)) {
            return false;
        }

        if (not init_render_pipeline_states(library, eye_texture_desc)) {
            return false;
        }

        return run(library.device, num_frames, submit_flags);
    }
#endif // __MAC_OS_X_VERSION_MIN_REQUIRED

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
    std::unique_ptr<vr::metal::VRSystem> metal_vr_system;

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

        metal_vr_system.reset(new vr::metal::VRSystem(vr_system));

        //------------------------------------------------------------------------------
        // Keep log a bit cleaner by giving the VR sytstem a chance to complete its
        // asynchronous startup before we go on.
        std::this_thread::sleep_for(std::chrono::seconds(2));

        //------------------------------------------------------------------------------
        // Grab a few bits of VR system info.
        for (vr::TrackedDeviceIndex_t tracked_device_index = 0; tracked_device_index < vr::k_unMaxTrackedDeviceCount; ++tracked_device_index) {
            const vr::ETrackedDeviceClass device_class = vr_system->GetTrackedDeviceClass(tracked_device_index);

            if (device_class == vr::TrackedDeviceClass_Invalid) {
                continue;
            }

            const std::string tracking_system_name = OpenVRUtils::get_tracked_device_string(vr_system, tracked_device_index, vr::Prop_TrackingSystemName_String);
            const std::string model_number         = OpenVRUtils::get_tracked_device_string(vr_system, tracked_device_index, vr::Prop_ModelNumber_String       );
            const std::string serial_number        = OpenVRUtils::get_tracked_device_string(vr_system, tracked_device_index, vr::Prop_SerialNumber_String      );

            switch (device_class) {
                case vr::TrackedDeviceClass_HMD:               std::cout << "Info: HMD: ";                break;
                case vr::TrackedDeviceClass_Controller:        std::cout << "Info: Controller: ";         break;
                case vr::TrackedDeviceClass_GenericTracker:    std::cout << "Info: Generic Tracker: ";    break;
                case vr::TrackedDeviceClass_TrackingReference: std::cout << "Info: Tracking Reference: "; break;
                case vr::TrackedDeviceClass_DisplayRedirect:   std::cout << "Info: Display Redirect: ";   break;

                default:
                    std::cerr << "Warning: Unknown device class: " << device_class << "!" << std::endl;
                    continue;
            }

            std::cout << model_number << " (";

            if (device_class == vr::TrackedDeviceClass_HMD) {
                const float display_frequency = vr_system->GetFloatTrackedDeviceProperty(tracked_device_index, vr::Prop_DisplayFrequency_Float, nullptr);
                std::cout << display_frequency << " Hz, ";
            }

            std::cout << tracking_system_name << ' ' << serial_number << ")\n";
        }

        metal_vr_system->GetRecommendedRenderTargetSize(recommended_render_target_size[0], recommended_render_target_size[1]);

        std::cout << "Info: Recommended render target size is " << recommended_render_target_size[0] << " x " << recommended_render_target_size[1] << '\n';

        //------------------------------------------------------------------------------
        // Always show chaperon for this test.
        if ((1)) {
            vr::VRChaperone()->ForceBoundsVisible(true);
        }

        //------------------------------------------------------------------------------
        // Export hidden mesh and/or distortion infor for analysis.
        if ((0)) {
            OpenVRUtils::export_hidden_area_outline_as_csv("/Users/Shared/hidden_area_mesh_outline.csv", true, vr_system);
        }

        if ((0)) {
            OpenVRUtils::export_distortion_samples_as_csv("/Users/Shared/distortion_samples.csv", true, vr_system);
        }
    }
    else {
        recommended_render_target_size[0] = 512;
        recommended_render_target_size[1] = 512;
    }

    //------------------------------------------------------------------------------
    // Setup Metal using the same device as used by the VR system.
    id<MTLDevice> metal_device = (metal_vr_system ? metal_vr_system->GetOutputDevice() : MTLCreateSystemDefaultDevice());

    if (not metal_device) {
        {
            std::unique_lock<std::mutex> lock(stdio_mutex);
            std::cerr << "Error: Failed to acquire Metal device!" << std::endl;
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

    if ((1)) {
        MTLTextureDescriptor* const texture_desc = [MTLTextureDescriptor new];
        {
            texture_desc.textureType = MTLTextureType2D;
            texture_desc.pixelFormat = MTLPixelFormatDepth32Float_Stencil8;
            texture_desc.width = recommended_render_target_size[0];
            texture_desc.height = recommended_render_target_size[1];
            texture_desc.storageMode = MTLStorageModePrivate;
            texture_desc.usage = MTLTextureUsageRenderTarget;
        }

        depth_attachment_texture = [metal_device newTextureWithDescriptor:texture_desc];
        stencil_attachment_texture = depth_attachment_texture;
    }
    else if ((1)) {
        MTLTextureDescriptor* const texture_desc = [MTLTextureDescriptor new];
        {
            texture_desc.textureType = MTLTextureType2D;
            texture_desc.pixelFormat = MTLPixelFormatDepth32Float;
            texture_desc.width = recommended_render_target_size[0];
            texture_desc.height = recommended_render_target_size[1];
            texture_desc.storageMode = MTLStorageModePrivate;
            texture_desc.usage = MTLTextureUsageRenderTarget;
        }

        depth_attachment_texture = [metal_device newTextureWithDescriptor:texture_desc];
    }
    else {
        MTLTextureDescriptor* const texture_desc = [MTLTextureDescriptor new];
        {
            texture_desc.textureType = MTLTextureType2D;
            texture_desc.pixelFormat = MTLPixelFormatStencil8;
            texture_desc.width = recommended_render_target_size[0];
            texture_desc.height = recommended_render_target_size[1];
            texture_desc.storageMode = MTLStorageModePrivate;
            texture_desc.usage = MTLTextureUsageRenderTarget;
        }

        stencil_attachment_texture = [metal_device newTextureWithDescriptor:texture_desc];
    }

    //------------------------------------------------------------------------------
    // Create meshes.
    @autoreleasepool {
        id<MTLCommandQueue> command_queue = [metal_device newCommandQueue];
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLBlitCommandEncoder> blit_command_encoder = [command_buffer blitCommandEncoder];

        if (is_hmd_present) {
            metal_hidden_area_mesh[0] = metal_vr_system->GetHiddenAreaMesh(metal_device, vr::Eye_Left , vr::k_eHiddenAreaMesh_Standard);
            metal_hidden_area_mesh[1] = metal_vr_system->GetHiddenAreaMesh(metal_device, vr::Eye_Right, vr::k_eHiddenAreaMesh_Standard);

            metal_hidden_area_mesh[0]->move_to_private_storage(blit_command_encoder, false);
            metal_hidden_area_mesh[1]->move_to_private_storage(blit_command_encoder, false);
        }
        else {
            vr::HiddenAreaMesh_t rectangular_mesh = vr::metal::HiddenAreaMesh::create_rectangular_mesh(0.25);

            metal_hidden_area_mesh[0].reset(new vr::metal::HiddenAreaMesh(metal_device, vr::k_eHiddenAreaMesh_Standard, rectangular_mesh));
            metal_hidden_area_mesh[1].reset(new vr::metal::HiddenAreaMesh(metal_device, vr::k_eHiddenAreaMesh_Standard, rectangular_mesh));

            vr::metal::HiddenAreaMesh::destroy_rectangular_mesh(rectangular_mesh);
        }

        cube_index_buffer = [metal_device newBufferWithLength:(((8 * 2) + 1) * sizeof(uint32_t)) options:MTLResourceStorageModeManaged];
        {
            uint32_t* const cube_indices = reinterpret_cast<uint32_t*>(cube_index_buffer.contents);
            size_t index = 0;

            cube_indices[index++] = 0;
            cube_indices[index++] = 1;
            cube_indices[index++] = 2;
            cube_indices[index++] = 3;
            cube_indices[index++] = 4;
            cube_indices[index++] = 5;
            cube_indices[index++] = 6;
            cube_indices[index++] = 7;
            cube_indices[index++] = uint32_t(-1);
            cube_indices[index++] = 2;
            cube_indices[index++] = 4;
            cube_indices[index++] = 0;
            cube_indices[index++] = 6;
            cube_indices[index++] = 1;
            cube_indices[index++] = 7;
            cube_indices[index++] = 3;
            cube_indices[index++] = 5;

            assert((index * sizeof(cube_indices[0])) == cube_index_buffer.length);
            [cube_index_buffer didModifyRange:NSMakeRange(0, cube_index_buffer.length)];
        }

        cube_index_buffer = copy_to_private_storage(cube_index_buffer, blit_command_encoder, false);

        [blit_command_encoder endEncoding];
        [command_buffer commit];
    }

    //------------------------------------------------------------------------------
    // Run tests.
    const size_t NUM_FRAMES = (90 * 30);        // 90 FPS for 30 seconds

    if ((1)) {
        @autoreleasepool {
            render_to_io_surface_texture(metal_default_library, NUM_FRAMES, recommended_render_target_size, vr::ColorSpace_Linear, vr::Submit_Default);
        }
    }

#if (__MAC_OS_X_VERSION_MIN_REQUIRED >= 101400)
    if ((0)) {
        @autoreleasepool {
            render_to_shared_texture(metal_default_library, NUM_FRAMES, recommended_render_target_size, vr::ColorSpace_Gamma, vr::Submit_Default);
        }
    }

    if ((0)) {
        @autoreleasepool {
            render_to_shared_texture_array(metal_default_library, NUM_FRAMES, recommended_render_target_size, vr::ColorSpace_Linear, vr::Submit_Default);
        }
    }
#endif // __MAC_OS_X_VERSION_MIN_REQUIRED

    //------------------------------------------------------------------------------
    // ...
    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
