// CMake will compile this file into AutoGenShaders.c""

#include <osgEarthProcedural/ProceduralShaders>

osgEarth::Procedural::ProceduralShaders::ProceduralShaders()
{
    TextureSplatting = "Procedural.TextureSplatting.glsl";
    _sources[TextureSplatting] = 
R"(#pragma vp_name Texture Splatter VV
#pragma vp_function oe_splat_View, vertex_view
#pragma import_defines(OE_TWEAKABLE)
#pragma import_defines(OE_SPLAT_NUM_LEVELS)
#pragma import_defines(OE_SNOW)
const int levels[2] = int[](14, 19);
flat out vec2 splat_tilexy[2];
out vec2 splat_uv[2];
// from REX SDK:
vec4 oe_terrain_scaleCoordsAndTileKeyToRefLOD(in vec2 tc, in float refLOD);
out vec4 oe_layer_tilec;
out float oe_splat_levelblend;
#ifdef OE_SNOW
out float oe_elev;
float oe_terrain_getElevation();
#endif
#ifdef OE_TWEAKABLE
#define tweakable uniform
#else
#define tweakable const
#endif
tweakable float oe_splat_blend_start = 2500.0;
tweakable float oe_splat_blend_end = 500.0;
#define MAP_TO_01(VAL,LO,HI) clamp((VAL-LO) / (HI-LO), 0.0, 1.0)
void oe_splat_View(inout vec4 vertex_view)
{
    // texture coordinates
    for (int i = 0; i < OE_SPLAT_NUM_LEVELS; ++i)
    {
        vec4 uvxy = oe_terrain_scaleCoordsAndTileKeyToRefLOD(oe_layer_tilec.st, levels[i]);
        splat_uv[i] = uvxy.xy;
        splat_tilexy[i] = uvxy.zw;
    }
    oe_splat_levelblend = MAP_TO_01(-vertex_view.z, oe_splat_blend_start, oe_splat_blend_end);
#ifdef OE_SNOW
    oe_elev = oe_terrain_getElevation();
#endif
}
[break]
#pragma vp_name Texture Splatter FS
#pragma vp_function oe_splat_Frag, fragment, 0.8
#pragma oe_use_shared_layer(OE_LIFEMAP_TEX, OE_LIFEMAP_MAT)
#pragma import_defines(OE_TEX_DIM_X)
//#pragma import_defines(OE_TEX_DIM_Y)
#define OE_TEX_DIM_Y 2
#pragma import_defines(OE_TWEAKABLE)
#pragma import_defines(OE_LIFEMAP_DIRECT)
//#pragma import_defines(OE_SNOW)
#pragma include Procedural.HexTiling.glsl
layout(binding = 5, std430) buffer SplatTextureArena {
    uint64_t texHandle[];
};
uniform float oe_texScale[128];
#define RUGGED 0
#define DENSE 1
#define LUSH 2
#define SPECIAL 3
in vec3 vp_Normal;
in vec3 vp_VertexView;
in vec3 oe_UpVectorView;
in float oe_splat_levelblend;
in vec4 oe_layer_tilec;
#pragma import_defines(OE_SPLAT_NUM_LEVELS)
flat in vec2 splat_tilexy[2];
in vec2 splat_uv[2];
#ifdef OE_SNOW
in float oe_elev;
#endif
#ifdef OE_LIFEMAP_DIRECT
    #define tweakable uniform
#else
    #define tweakable const
#endif
tweakable float dense_power = 1.0;
tweakable float lush_power = 1.0;
tweakable float rugged_power = 1.0;
tweakable float ao_power = 1.0;
tweakable float oe_snow = 0.0;
tweakable float oe_snow_min_elev = 1000.0;
tweakable float oe_snow_max_elev = 3500.0;
tweakable float oe_splat_blend_rgbh_mix = 0.8;
tweakable float oe_splat_blend_normal_mix = 0.85;
tweakable float oe_splat_brightness = 1.0;
tweakable float oe_splat_contrast = 1.0;
tweakable float oe_dense_contrast = 1.0;
uniform float oe_normal_power = 1.0;
uniform float oe_mask_alpha = 0.0;
uniform float oe_displacement_depth = 0.1;
mat3 oe_normalMapTBN;
#define MAP_TO_01(VAL,LO,HI) clamp((VAL-LO) / (HI-LO), 0.0, 1.0)
#define DECEL(V,P) (1.0-pow(1.0-(V),(P)))
#if defined(OE_LIFEMAP_DIRECT) && OE_LIFEMAP_DIRECT
#define MODIFY(V,M) clamp((M), 0.0, 1.0)
#else
#define MODIFY(V,M) clamp((V)*(M), 0.0, 1.0)
#endif
// optimized uncompressor (assumes Z is never negative)
#define UNPACK_NORMAL(P,N) N.xy = P*2.0-1.0; N.z = 1.0-abs(N.x)-abs(N.y); N /= length(N)
struct Pixel {
    vec4 rgbh;
    vec3 normal;
    vec3 material;
};
#define ROUGHNESS 0
#define AO 1
#define METAL 2
// fragment stage global PBR parameters.
struct OE_PBR {
    float roughness;
    float ao;
    float metal;
    float brightness;
    float contrast;
} oe_pbr;
// compute the splatting texture coordinate by combining the macro (tile_xy)
// and micro (local xy) components. Cannot do this in the VS because it will
// overflow the interpolator and pause pixel jitter
void get_coord(out vec2 coord, in int index, in int level)
{
    vec2 scale = vec2(oe_texScale[index]);
    vec2 a = fract(splat_tilexy[level] * scale);
    vec2 b = splat_uv[level] * scale;
    coord = a + b;
}
#pragma import_defines(OE_SPLAT_HEX_TILER)
#ifndef OE_SPLAT_HEX_TILER
#define OE_SPLAT_HEX_TILER 0
#endif
void get_pixel(out Pixel res, in int index, in vec2 coord)
{
    vec4 nnra;
#if OE_SPLAT_HEX_TILER == 1
    ht_hex2colTex_optimized(
        sampler2D(texHandle[index * 2]),
        sampler2D(texHandle[index * 2 + 1]),
        coord,
        res.rgbh,
        nnra);
#else
    res.rgbh = texture(sampler2D(texHandle[index * 2]), coord);
    nnra = texture(sampler2D(texHandle[index * 2 + 1]), coord);
#endif
    UNPACK_NORMAL(nnra.xy, res.normal);
    res.material = vec3(nnra[2], nnra[3], 0.0); // roughness, ao, metal
}
float heightAndEffectMix(in float h1, in float a1, in float h2, in float a2)
{
    // https://tinyurl.com/y5nkw2l9
    float ma = max(h1 + a1, h2 + a2) - oe_displacement_depth;
    float b1 = max(h1 + a1 - ma, 0.0);
    float b2 = max(h2 + a2 - ma, 0.0);
    return b2 / (b1 + b2);
}
void pixmix(out Pixel res, in Pixel p1, in Pixel p2, float m)
{
    res.rgbh = mix(p1.rgbh, p2.rgbh, m);
    res.normal = mix(p1.normal, p2.normal, m);
    res.material = mix(p1.material, p2.material, m);
}
void resolveRow(out Pixel result, int level, int row, float xvar)
{
    Pixel p1, p2;
    vec2 coord;
    // calulate first column index and mix factor
    float xf = xvar * (float(OE_TEX_DIM_X) - 1.0);
    float xf_floor = floor(xf);
    int x = int(xf_floor);
    float x_mix = xf - xf_floor;
    // texture index:
    int i = row * OE_TEX_DIM_X + x;
    // read both columns:
    get_coord(coord, i, level);
    get_pixel(p1, i, coord);
    i = (i%OE_TEX_DIM_X < OE_TEX_DIM_X) ? i + 1 : i;
    get_coord(coord, i, level);
    get_pixel(p2, i, coord);
    // blend them using both heightmap:
    float m = heightAndEffectMix(p1.rgbh[3], 1.0 - x_mix, p2.rgbh[3], x_mix);
    pixmix(result, p1, p2, m);
}
void resolveLevel(out Pixel result, int level, float rugged, float lush, float dense, int override_material_index)
{
    float surface_mix = dense;
    // resolve the substrate (dirt and rocks)
    Pixel substrate;
    resolveRow(substrate, level, 0, rugged);
    // resolve the surface texture (greenery and debris)
    Pixel surface;
    if (override_material_index > 0)
    {
        vec2 coord;
        get_coord(coord, override_material_index - 1, 0);
        get_pixel(surface, override_material_index - 1, coord);
        surface_mix = clamp(1.0 - DECEL(dense, 2.0), 0, 1); //  (rugged + dense + lush), 0, 1);
    }
    else
    {
        resolveRow(surface, level, 1, lush);
    }
    // use density to modulate the depth blend between the two.
    float m = heightAndEffectMix(
        substrate.rgbh[3], 1.0 - surface_mix,
        surface.rgbh[3], surface_mix);
    if (level == 0)
    {
        pixmix(result, substrate, surface, m);
    }
    else
    {
        Pixel temp;
        pixmix(temp, substrate, surface, m);
        float mat_mix = min(oe_splat_levelblend, oe_splat_blend_rgbh_mix);
        result.rgbh = mix(result.rgbh, temp.rgbh, mat_mix);
        result.normal = mix(result.normal, temp.normal, min(oe_splat_levelblend, oe_splat_blend_normal_mix));
        result.material = mix(result.material, temp.material, mat_mix);
    }
}
void oe_splat_Frag(inout vec4 quad)
{
    // sample the life map and extract the compenents:
    vec2 c = (OE_LIFEMAP_MAT * oe_layer_tilec).st;
    vec4 life = texture(OE_LIFEMAP_TEX, c);
    float rugged = MODIFY(life[RUGGED], rugged_power);
    float lush = MODIFY(life[LUSH], lush_power);
    float dense = MODIFY(life[DENSE], dense_power);
    ivec2 tfc = ivec2(min(int(c.x*256.0), 255), min(int(c.y*256.0), 255));
    vec4 life_i = texelFetch(OE_LIFEMAP_TEX, tfc, 0);
    int material_index = int(life_i[3] * 255.0f);
    // compute the pixel color:
    Pixel pixel;
    for (int level = 0; level < OE_SPLAT_NUM_LEVELS; ++level)
    {
        resolveLevel(pixel, level, rugged, lush, dense, material_index);
    }
    // apply PBR
    oe_pbr.roughness = clamp(oe_pbr.roughness * pixel.material[ROUGHNESS], 0.0, 1.0);
    oe_pbr.ao = clamp(oe_pbr.ao * pow(pixel.material[AO], ao_power), 0.0, 1.0);
    oe_pbr.metal = clamp(pixel.material[METAL], 0.0, 1.0);
    oe_pbr.brightness *= oe_splat_brightness;
    oe_pbr.contrast *= oe_splat_contrast;
    //pixel.rgbh.rgb = clamp(((pixel.rgbh.rgb - 0.5)*oe_splat_contrast + 0.5) * oe_splat_brightness, 0, 1);
    vec3 color = pixel.rgbh.rgb;
    // NORMAL
    //pixel.normal = normalize(vec3(
    //    DECEL(pixel.normal.x, normal_power),
    //    DECEL(pixel.normal.y, normal_power),
    //    pixel.normal.z));
    pixel.normal = vec3(
        DECEL(pixel.normal.x, oe_normal_power),
        DECEL(pixel.normal.y, oe_normal_power),
        pixel.normal.z);
    vp_Normal = normalize(vp_Normal + oe_normalMapTBN * pixel.normal);
    float composite = DECEL(clamp(dense + lush + rugged, 0.0, 1.0), oe_mask_alpha);
    float alpha = oe_mask_alpha > 0.0 ? composite : 1.0;
    // final color output:
    quad = vec4(color, alpha);
}
)";

    Vegetation = "Procedural.Vegetation.glsl";
    _sources[Vegetation] = 
R"(#pragma vp_function oe_vegetation_vs_view, vertex_view
#pragma import_defines(OE_WIND_TEX)
#pragma import_defines(OE_WIND_TEX_MATRIX)
#pragma import_defines(OE_NOISE_TEX_INDEX)
struct Instance
{
    mat4 xform;
    vec2 local_uv;
    uint lod;
    float visibility[2]; // per LOD
    float radius;
    float alpha_cutoff;
    uint first_lod_cmd_index;
};
layout(binding = 0, std430) buffer Instances {
    Instance instances[];
};
layout(binding = 1, std430) buffer TextureArena {
    uint64_t textures[];
};
#define NOISE_TEX sampler2D(textures[OE_NOISE_TEX_INDEX])
layout(location = 5) in vec3 flex;
// outputs
flat out uint oe_lod;
#ifdef OE_WIND_TEX
uniform sampler3D OE_WIND_TEX;
uniform mat4 OE_WIND_TEX_MATRIX;
uniform float osg_FrameTime;
#pragma import_defines(OE_TWEAKABLE)
#ifdef OE_TWEAKABLE
#define tweakable uniform
#else
#define tweakable const
#endif
tweakable float oe_wind_power = 1.0;
void oe_apply_wind(inout vec4 vertex, in int index)
{
    // scale the vert's flexibility by the model Z scale factor
    mat3 vec3xform = mat3(instances[index].xform);
    float flexibility = length(flex) * vec3xform[2][2];
    if (flexibility > 0.0 && oe_wind_power > 0.0)
    {
        vec3 center = instances[index].xform[3].xyz;
        vec2 tile_uv = instances[index].local_uv;
        // sample the wind direction and speed:
        vec4 wind = textureProj(OE_WIND_TEX, (OE_WIND_TEX_MATRIX * vertex));
        vec3 wind_vec = normalize(wind.rgb * 2.0 - 1.0);
        float speed = wind.a * oe_wind_power;
        const float rate = 0.05 * speed;
        vec4 noise_moving = textureLod(NOISE_TEX, tile_uv + osg_FrameTime * rate, 0);
        speed *= noise_moving[3];
        // final wind force vector:
        vec3 wind_force = wind_vec * speed;
        // project the wind vector onto the flex plane
        vec3 flex_plane_normal = normalize(gl_NormalMatrix * vec3xform * flex);
        float dist = dot(wind_force, flex_plane_normal);
        vec3 wind_vec_projected = wind_force - flex_plane_normal * dist;
        // move the vertex within the flex plane
        vertex.xyz += wind_vec_projected * flexibility;
    }
}
#endif
void oe_vegetation_vs_view(inout vec4 vertex)
{
    int i = gl_BaseInstance + gl_InstanceID;
    oe_lod = instances[i].lod;
#ifdef OE_WIND_TEX
    oe_apply_wind(vertex, i);
#endif
}
[break]
#pragma vp_function oe_vegetation_fs, fragment, 0.9
#pragma import_defines(OE_IS_SHADOW_CAMERA)
flat in uint oe_lod; // from VS
in vec3 vp_VertexView;
uniform float oe_veg_bbd0 = 0.5;
uniform float oe_veg_bbd1 = 0.75;
void oe_vegetation_fs(inout vec4 color)
{
#ifndef OE_IS_SHADOW_CAMERA
    // reduce the alpha on faces that are orthogonal to the view vector.
    // this makes cross-hatch impostors look better.
    // (only do this for impostor lods)
    if (oe_lod > 0)
    {
        vec3 dx = dFdx(vp_VertexView);
        vec3 dy = dFdy(vp_VertexView);
        vec3 fn = normalize(cross(dx, dy));
        float f = abs(dot(fn, normalize(vp_VertexView)));
        color.a *= clamp(mix(0, 1, (f - oe_veg_bbd0) / (oe_veg_bbd1 - oe_veg_bbd0)), 0, 1);
    }
#endif // !OE_IS_SHADOW_CAMERA
}
)";

    HexTiling = "Procedural.HexTiling.glsl";
    _sources[HexTiling] = 
R"(// Adapted and ported to GLSL from:
// https://github.com/mmikk/hextile-demo
const float ht_g_fallOffContrast = 0.6;
const float ht_g_exp = 7;
#ifndef mul
#define mul(X, Y) ((X)*(Y))
#endif
#ifndef M_PI
#define M_PI 3.1417927
#endif
#define HEX_SCALE 3.46410161
// Output:\ weights associated with each hex tile and integer centers
void ht_TriangleGrid(
    out float w1, out float w2, out float w3,
    out ivec2 vertex1, out ivec2 vertex2, out ivec2 vertex3,
    in vec2 st)
{
    // Scaling of the input
    st *= HEX_SCALE; // 2 * 1.sqrt(3);
    // Skew input space into simplex triangle grid
    const mat2 gridToSkewedGrid = mat2(1.0, -0.57735027, 0.0, 1.15470054);
    vec2 skewedCoord = mul(gridToSkewedGrid, st);
    ivec2 baseId = ivec2(floor(skewedCoord));
    vec3 temp = vec3(fract(skewedCoord), 0);
    temp.z = 1.0 - temp.x - temp.y;
    float s = step(0.0, -temp.z);
    float s2 = 2 * s - 1;
    w1 = -temp.z*s2;
    w2 = s - temp.y*s2;
    w3 = s - temp.x*s2;
    vertex1 = baseId + ivec2(s, s);
    vertex2 = baseId + ivec2(s, 1 - s);
    vertex3 = baseId + ivec2(1 - s, s);
}
// Output:\ weights associated with each hex tile and integer centers
void ht_TriangleGrid_f(
    out float w1, out float w2, out float w3,
    out vec2 vertex1, out vec2 vertex2, out vec2 vertex3,
    in vec2 st)
{
    // Scaling of the input
    st *= HEX_SCALE; // 2 * 1.sqrt(3);
    // Skew input space into simplex triangle grid
    const mat2 gridToSkewedGrid = mat2(1.0, -0.57735027, 0.0, 1.15470054);
    vec2 skewedCoord = mul(gridToSkewedGrid, st);
    vec2 baseId = floor(skewedCoord);
    vec3 temp = vec3(fract(skewedCoord), 0);
    temp.z = 1.0 - temp.x - temp.y;
    float s = step(0.0, -temp.z);
    float s2 = 2 * s - 1;
    w1 = -temp.z*s2;
    w2 = s - temp.y*s2;
    w3 = s - temp.x*s2;
    vertex1 = baseId + vec2(s, s);
    vertex2 = baseId + vec2(s, 1 - s);
    vertex3 = baseId + vec2(1 - s, s);
}
vec2 ht_hash(vec2 p)
{
    vec2 r = mat2(127.1, 311.7, 269.5, 183.3) * p;
    return fract(sin(r)*43758.5453);
}
vec2 ht_MakeCenST(ivec2 Vertex)
{
    const mat2 invSkewMat = mat2(1.0, 0.5, 0.0, 1.0 / 1.15470054);
    return mul(invSkewMat, Vertex) / HEX_SCALE;
}
mat2 ht_LoadRot2x2(ivec2 idx, float rotStrength)
{
    float angle = abs(idx.x*idx.y) + abs(idx.x + idx.y) + M_PI;
    // remap to +/-pi
    angle = mod(angle, 2 * M_PI);
    if (angle < 0) angle += 2 * M_PI;
    if (angle > M_PI) angle -= 2 * M_PI;
    angle *= rotStrength;
    float cs = cos(angle), si = sin(angle);
    return mat2(cs, -si, si, cs);
}
vec3 ht_Gain3(vec3 x, float r)
{
    // increase contrast when r>0.5 and
    // reduce contrast if less
    float k = log(1 - r) / log(0.5);
    vec3 s = 2 * step(0.5, x);
    vec3 m = 2 * (1 - s);
    vec3 res = 0.5*s + 0.25*m * pow(max(vec3(0.0), s + x * m), vec3(k));
    return res.xyz / (res.x + res.y + res.z);
}
vec3 ht_ProduceHexWeights(vec3 W, ivec2 vertex1, ivec2 vertex2, ivec2 vertex3)
{
    vec3 res;
    int v1 = (vertex1.x - vertex1.y) % 3;
    if (v1 < 0) v1 += 3;
    int vh = v1 < 2 ? (v1 + 1) : 0;
    int vl = v1 > 0 ? (v1 - 1) : 2;
    int v2 = vertex1.x < vertex3.x ? vl : vh;
    int v3 = vertex1.x < vertex3.x ? vh : vl;
    res.x = v3 == 0 ? W.z : (v2 == 0 ? W.y : W.x);
    res.y = v3 == 1 ? W.z : (v2 == 1 ? W.y : W.x);
    res.z = v3 == 2 ? W.z : (v2 == 2 ? W.y : W.x);
    return res;
}
// Input: vM is tangent space normal in [-1;1].
// Output: convert vM to a derivative.
vec2 ht_TspaceNormalToDerivative(in vec3 vM)
{
    const float scale = 1.0 / 128.0;
    // Ensure vM delivers a positive third component using abs() and
    // constrain vM.z so the range of the derivative is [-128; 128].
    const vec3 vMa = abs(vM);
    const float z_ma = max(vMa.z, scale*max(vMa.x, vMa.y));
    // Set to match positive vertical texture coordinate axis.
    const bool gFlipVertDeriv = true;
    const float s = gFlipVertDeriv ? -1.0 : 1.0;
    return -vec2(vM.x, s*vM.y) / z_ma;
}
vec2 ht_sampleDeriv(sampler2D nmap, vec2 st, vec2 dSTdx, vec2 dSTdy)
{
    // sample
    vec3 vM = 2.0*textureGrad(nmap, st, dSTdx, dSTdy).xyz - 1.0;
    return ht_TspaceNormalToDerivative(vM);
}
// Input:\ nmap is a normal map
// Input:\ r increase contrast when r>0.5
// Output:\ deriv is a derivative dHduv wrt units in pixels
// Output:\ weights shows the weight of each hex tile
void bumphex2derivNMap(
    out vec2 deriv, out vec3 weights,
    sampler2D nmap, vec2 st,
    float rotStrength, float r)
{
    vec2 dSTdx = dFdx(st), dSTdy = dFdy(st);
    // Get triangle info
    float w1, w2, w3;
    ivec2 vertex1, vertex2, vertex3;
    ht_TriangleGrid(w1, w2, w3, vertex1, vertex2, vertex3, st);
    mat2 rot1 = ht_LoadRot2x2(vertex1, rotStrength);
    mat2 rot2 = ht_LoadRot2x2(vertex2, rotStrength);
    mat2 rot3 = ht_LoadRot2x2(vertex3, rotStrength);
    vec2 cen1 = ht_MakeCenST(vertex1);
    vec2 cen2 = ht_MakeCenST(vertex2);
    vec2 cen3 = ht_MakeCenST(vertex3);
    vec2 st1 = mul(st - cen1, rot1) + cen1 + ht_hash(vertex1);
    vec2 st2 = mul(st - cen2, rot2) + cen2 + ht_hash(vertex2);
    vec2 st3 = mul(st - cen3, rot3) + cen3 + ht_hash(vertex3);
    // Fetch input
    vec2 d1 = ht_sampleDeriv(nmap, st1,
        mul(dSTdx, rot1), mul(dSTdy, rot1));
    vec2 d2 = ht_sampleDeriv(nmap, st2,
        mul(dSTdx, rot2), mul(dSTdy, rot2));
    vec2 d3 = ht_sampleDeriv(nmap, st3,
        mul(dSTdx, rot3), mul(dSTdy, rot3));
    d1 = mul(rot1, d1); d2 = mul(rot2, d2); d3 = mul(rot3, d3);
    // produce sine to the angle between the conceptual normal
    // in tangent space and the Z-axis
    vec3 D = vec3(dot(d1, d1), dot(d2, d2), dot(d3, d3));
    vec3 Dw = sqrt(D / (1.0 + D));
    Dw = mix(vec3(1.0), Dw, ht_g_fallOffContrast);	// 0.6
    vec3 W = Dw * pow(vec3(w1, w2, w3), vec3(ht_g_exp));	// 7
    W /= (W.x + W.y + W.z);
    if (r != 0.5) W = ht_Gain3(W, r);
    deriv = W.x * d1 + W.y * d2 + W.z * d3;
    weights = ht_ProduceHexWeights(W.xyz, vertex1, vertex2, vertex3);
}
// Input:\ tex is a texture with color
// Input:\ r increase contrast when r>0.5
// Output:\ color is the blended result
// Output:\ weights shows the weight of each hex tile
void ht_hex2colTex(
    out vec4 color,
    sampler2D tex,
    vec2 st,
    float rotStrength)
{
    vec2 dSTdx = dFdx(st), dSTdy = dFdy(st);
    // Get triangle info
    float w1, w2, w3;
    ivec2 vertex1, vertex2, vertex3;
    ht_TriangleGrid(w1, w2, w3, vertex1, vertex2, vertex3, st);
    mat2 rot1 = ht_LoadRot2x2(vertex1, rotStrength);
    mat2 rot2 = ht_LoadRot2x2(vertex2, rotStrength);
    mat2 rot3 = ht_LoadRot2x2(vertex3, rotStrength);
    vec2 cen1 = ht_MakeCenST(vertex1);
    vec2 cen2 = ht_MakeCenST(vertex2);
    vec2 cen3 = ht_MakeCenST(vertex3);
    vec2 st1 = mul(st - cen1, rot1) + cen1 + ht_hash(vertex1);
    vec2 st2 = mul(st - cen2, rot2) + cen2 + ht_hash(vertex2);
    vec2 st3 = mul(st - cen3, rot3) + cen3 + ht_hash(vertex3);
    // Fetch input
    vec4 c1 = textureGrad(tex, st1, dSTdx*rot1, dSTdy*rot1);
    vec4 c2 = textureGrad(tex, st2, dSTdx*rot2, dSTdy*rot2);
    vec4 c3 = textureGrad(tex, st3, dSTdx*rot3, dSTdy*rot3);
    // use luminance as weight
    const vec3 Lw = vec3(0.299, 0.587, 0.114);
    vec3 Dw = vec3(dot(c1.xyz, Lw), dot(c2.xyz, Lw), dot(c3.xyz, Lw));
    Dw = mix(vec3(1.0), Dw, ht_g_fallOffContrast);	// 0.6
    vec3 W = Dw * pow(vec3(w1, w2, w3), vec3(ht_g_exp));	// 7
    W /= (W.x + W.y + W.z);
    //if (r != 0.5) W = Gain3(W, r);
    color = W.x * c1 + W.y * c2 + W.z * c3;
    //weights = ProduceHexWeights(W.xyz, vertex1, vertex2, vertex3);
}
#define HT_HASH(X) fract(sin(mat2(127.1, 311.7, 269.5, 183.3) * X)*43758.5453)
// Hextiling function optimized for no rotations and to 
// sample and interpolate both color and material vectors
void ht_hex2colTex_optimized(
    in sampler2D color_tex,
    in sampler2D material_tex,
    in vec2 st,
    out vec4 color,
    out vec4 material)
{
    // Get triangle info
    vec3 weights;
    vec2 vertex1, vertex2, vertex3;
    ht_TriangleGrid_f(weights[0], weights[1], weights[2], vertex1, vertex2, vertex3, st);
    // randomize the sampling offsets:
    vec2 st1 = st + ht_hash(vertex1);
    vec2 st2 = st + ht_hash(vertex2);
    vec2 st3 = st + ht_hash(vertex3);
    // Use the same partial derivitives to sample all three locations
    // to avoid rendering artifacts.
#if 1
    // Fast way: replace textureGrad by manually calculating the LOD
    // and using textureLod instead (much faster than textureGrad)
    // https://solidpixel.github.io/2022/03/27/texture_sampling_tips.html
    ivec2 tex_dim;
    vec2 ddx, ddy;
    float lod;
    vec2 st_ddx = dFdx(st), st_ddy = dFdy(st);
    tex_dim = textureSize(color_tex, 0);
    ddx = st_ddx * float(tex_dim.x), ddy = st_ddy * float(tex_dim.y);
    lod = 0.5 * log2(max(dot(ddx, ddx), dot(ddy, ddy)));
    vec4 c1 = textureLod(color_tex, st1, lod);
    vec4 c2 = textureLod(color_tex, st2, lod);
    vec4 c3 = textureLod(color_tex, st3, lod);
    tex_dim = textureSize(material_tex, 0);
    ddx = st_ddx * float(tex_dim.x), ddy = st_ddy * float(tex_dim.y);
    lod = 0.5 * log2(max(dot(ddx, ddx), dot(ddy, ddy)));
    vec4 m1 = textureLod(material_tex, st1, lod);
    vec4 m2 = textureLod(material_tex, st2, lod);
    vec4 m3 = textureLod(material_tex, st3, lod);
#else
    // Original approach: use textureGrad to supply the same gradient
    // for each sample point (slow)
    vec2 ddx = dFdx(st), ddy = dFdy(st);
    vec4 c1 = textureGrad(color_tex, st1, ddx, ddy);
    vec4 c2 = textureGrad(color_tex, st2, ddx, ddy);
    vec4 c3 = textureGrad(color_tex, st3, ddx, ddy);
    vec4 m1 = textureGrad(material_tex, st1, ddx, ddy);
    vec4 m2 = textureGrad(material_tex, st2, ddx, ddy);
    vec4 m3 = textureGrad(material_tex, st3, ddx, ddy);
#endif
    // Use color's luminance as weighting factor
    const vec3 Lw = vec3(0.299, 0.587, 0.114);
    vec3 Dw = vec3(dot(c1.xyz, Lw), dot(c2.xyz, Lw), dot(c3.xyz, Lw));
    Dw = mix(vec3(1.0), Dw, ht_g_fallOffContrast);
    vec3 W = Dw * pow(weights, vec3(ht_g_exp));
    W /= (W.x + W.y + W.z);
    color = W.x * c1 + W.y * c2 + W.z * c3;
    material = W.x * m1 + W.y * m2 + W.z * m3;
}
)";
}
