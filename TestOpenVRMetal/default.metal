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

#include <metal_stdlib>
using namespace metal;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct float2_uv_to_ndc_s {
    float4      position [[ position ]];
} float2_uv_to_ndc_i;

vertex float2_uv_to_ndc_i
float2_uv_to_ndc_v(device const float2* const vertices [[ buffer(0) ]], uint vid [[ vertex_id ]])
{
    float2_uv_to_ndc_i interpolators;

    interpolators.position.xy = ((vertices[vid] * 2.0f) - 1.0f);
    interpolators.position.zw = float2(0.0, 1.0);

    return interpolators;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct cube_transforms_s {
    float4      position [[ position ]];
    half3       uvw;
} cube_transforms_i;

constant float3 CUBE_VERTEX_POSITIONS[] = {
    { -0.5, -0.5, -0.5 },
    { -0.5, -0.5,  0.5 },
    { -0.5,  0.5, -0.5 },
    { -0.5,  0.5,  0.5 },
    {  0.5,  0.5, -0.5 },
    {  0.5,  0.5,  0.5 },
    {  0.5, -0.5, -0.5 },
    {  0.5, -0.5,  0.5 },
};

vertex cube_transforms_i
instanced_cubes_v(constant const float4x4& projection_matrix [[ buffer(0) ]],
                  device const float4x4* const cube_transforms [[ buffer(1) ]],
                  uint iid [[ instance_id ]],
                  uint vid [[ vertex_id ]])
{
    cube_transforms_i interpolators;

    const float4 position = float4(CUBE_VERTEX_POSITIONS[vid], 1.0);

    interpolators.position = (projection_matrix * cube_transforms[iid] * position);
    interpolators.uvw = (half3(position.xyz) + half3(0.5f));

    return interpolators;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct constant_color_s {
} constant_color_i;

fragment half4
constant_color_f(const constant_color_i interpolators [[ stage_in ]])
{
    return half4(0.0, 0.5, 0.0, 1.0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct uvw_color_s {
    half3       uvw;
} uvw_color_i;

fragment half4
uvw_color_f(const uvw_color_i interpolators [[ stage_in ]])
{
    return half4(interpolators.uvw.rgb, 1.0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
