#version 430
#pragma include Splat.GroundCover.Types.glsl

layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

// LUT methods generated by GroundCoverLayer.cpp
struct oe_gc_LandCoverGroup {
    int firstAssetIndex;
    int numAssets;
    float fill;
};
struct oe_gc_Asset {
    int assetId;
    int modelId;
    uint64_t modelSampler;
    uint64_t sideSampler;
    uint64_t topSampler;
    float width;
    float height;
    float sizeVariation;
    float fill;
};
bool oe_gc_getLandCoverGroup(in int zone, in int code, out oe_gc_LandCoverGroup result);
bool oe_gc_getAsset(in int index, out oe_gc_Asset result);

uniform sampler2D oe_gc_noiseTex;
#define NOISE_SMOOTH   0
#define NOISE_RANDOM   1
#define NOISE_RANDOM_2 2
#define NOISE_CLUMPY   3

// (LLx, LLy, URx, URy, tileNum
uniform float oe_tile[5];
uniform int oe_gc_zone;

uniform vec2 oe_tile_elevTexelCoeff;
uniform sampler2D oe_tile_elevationTex;
uniform mat4 oe_tile_elevationTexMatrix;
uniform float oe_GroundCover_colorMinSaturation;

#pragma import_defines(OE_LANDCOVER_TEX)
#pragma import_defines(OE_LANDCOVER_TEX_MATRIX)
uniform sampler2D OE_LANDCOVER_TEX;
uniform mat4 OE_LANDCOVER_TEX_MATRIX;

#pragma import_defines(OE_GROUNDCOVER_MASK_SAMPLER)
#pragma import_defines(OE_GROUNDCOVER_MASK_MATRIX)
#ifdef OE_GROUNDCOVER_MASK_SAMPLER
uniform sampler2D OE_GROUNDCOVER_MASK_SAMPLER;
uniform mat4 OE_GROUNDCOVER_MASK_MATRIX;
#endif

#pragma import_defines(OE_GROUNDCOVER_COLOR_SAMPLER)
#pragma import_defines(OE_GROUNDCOVER_COLOR_MATRIX)
#ifdef OE_GROUNDCOVER_COLOR_SAMPLER
  uniform sampler2D OE_GROUNDCOVER_COLOR_SAMPLER ;
  uniform mat4 OE_GROUNDCOVER_COLOR_MATRIX ;
#endif

#pragma import_defines(OE_GROUNDCOVER_PICK_NOISE_TYPE)
#ifdef OE_GROUNDCOVER_PICK_NOISE_TYPE
  int pickNoiseType = OE_GROUNDCOVER_PICK_NOISE_TYPE ;
#else
  int pickNoiseType = NOISE_RANDOM;
  //int pickNoiseType = NOISE_CLUMPY;
#endif

#ifdef OE_GROUNDCOVER_COLOR_SAMPLER

// https://stackoverflow.com/a/17897228/4218920
vec3 rgb2hsv(vec3 c)
{
    const vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    const float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

bool isLegalColor(in vec2 tilec)
{
    vec4 c = texture(OE_GROUNDCOVER_COLOR_SAMPLER, (OE_GROUNDCOVER_COLOR_MATRIX*vec4(tilec,0,1)).st);
    vec3 hsv = rgb2hsv(c.rgb);
    return hsv[1] > oe_GroundCover_colorMinSaturation;
}

#endif // OE_GROUNDCOVER_COLOR_SAMPLER

float getElevation(in vec2 tilec)
{
    vec2 elevc = tilec
       * oe_tile_elevTexelCoeff.x * oe_tile_elevationTexMatrix[0][0] // scale
       + oe_tile_elevTexelCoeff.x * oe_tile_elevationTexMatrix[3].st // bias
       + oe_tile_elevTexelCoeff.y;
    return texture(oe_tile_elevationTex, elevc).r;
}

void main()
{
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;

    vec2 offset = vec2(float(x), float(y));
    vec2 halfSpacing = 0.5 / vec2(gl_NumWorkGroups.xy);
    vec2 tilec = halfSpacing + offset / vec2(gl_NumWorkGroups.xy);

    vec4 noise = textureLod(oe_gc_noiseTex, tilec, 0);

    vec2 shift = vec2(fract(noise[1]*1.5), fract(noise[2]*1.5))*2.0-1.0;
    tilec += shift * halfSpacing;

    vec4 tilec4 = vec4(tilec, 0, 1);

#ifdef OE_GROUNDCOVER_COLOR_SAMPLER
    if (!isLegalColor(tilec))
         return;
#endif

    // sample the landcover data
    int code = int(textureLod(OE_LANDCOVER_TEX, (OE_LANDCOVER_TEX_MATRIX*tilec4).st, 0).r);    
    oe_gc_LandCoverGroup group;
    if (oe_gc_getLandCoverGroup(oe_gc_zone, code, group) == false)
        return;

    // If we're using a mask texture, sample it now:
#ifdef OE_GROUNDCOVER_MASK_SAMPLER
    float mask = texture(OE_GROUNDCOVER_MASK_SAMPLER, (OE_GROUNDCOVER_MASK_MATRIX*tilec4).st).a;
    if ( mask > 0.0 )
        return;
#endif

    // discard instances based on noise value threshold (coverage). If it passes,
    // scale the noise value back up to [0..1]
    if (noise[NOISE_SMOOTH] > group.fill)
        return;
    noise[NOISE_SMOOTH] /= group.fill;

    // select a billboard at random
    float pickNoise = 1.0-noise[pickNoiseType];
    int assetIndex = group.firstAssetIndex + int(floor(pickNoise * float(group.numAssets)));
    assetIndex = min(assetIndex, group.firstAssetIndex + group.numAssets - 1);

    // Recover the asset we randomly picked:
    oe_gc_Asset asset;
    oe_gc_getAsset(assetIndex, asset);

    // asset fill:
    if (noise[NOISE_RANDOM_2] > asset.fill)
        return;

    vec2 LL = vec2(oe_tile[0], oe_tile[1]);
    vec2 UR = vec2(oe_tile[2], oe_tile[3]);

    vec4 vertex_model = vec4(mix(LL, UR, tilec), getElevation(tilec), 1.0);

    // It's a keeper - record it to the instance buffer.
    uint i = atomicAdd(instanceHeader.count, 1);

    instance[i].tileNum = uint(oe_tile[4]);
    instance[i].vertex = vertex_model;
    instance[i].tilec = tilec;

    instance[i].fillEdge = 1.0;
    const float xx = 0.5;
    if (noise[NOISE_SMOOTH] > xx)
        instance[i].fillEdge = 1.0-((noise[NOISE_SMOOTH]-xx)/(1.0-xx));

    instance[i].modelId = asset.modelId;

    instance[i].modelSampler = asset.modelSampler;
    instance[i].sideSampler = asset.sideSampler;
    instance[i].topSampler = asset.topSampler;

     //a pseudo-random scale factor to the width and height of a billboard
    instance[i].sizeScale = 1.0 + asset.sizeVariation * (noise[NOISE_RANDOM_2]*2.0-1.0);
    instance[i].width = asset.width * instance[i].sizeScale;
    instance[i].height = asset.height * instance[i].sizeScale;

    float rotation = 6.283185 * noise[NOISE_RANDOM];
    instance[i].sinrot = sin(rotation);
    instance[i].cosrot = cos(rotation);

    instance[i].instanceId = gl_GlobalInvocationID.y * gl_NumWorkGroups.y + gl_GlobalInvocationID.x;
}
