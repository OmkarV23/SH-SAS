#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <stdint.h>
#include <cuda_runtime_api.h>

struct Octane
{
    int idx;
    float3 center;
    int    nbr[8];;  // Indices of neighboring octants
};

__device__ __forceinline__ int oct_code(bool dx, bool dy, bool dz)
{
    return (dz << 2) | (dy << 1) | dx;   // dx is LSB
}

template <typename T>
static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
{
    std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
    ptr = reinterpret_cast<T*>(offset);
    chunk = reinterpret_cast<char*>(ptr + count);
}

struct OctaneState{

    Octane* octs;

    static OctaneState fromChunk(char*& chunk, size_t P)
    {
        OctaneState state;
        obtain(chunk, state.octs, P, 128);
        return state;
    }
};


inline std::size_t bytesForOctanes(std::size_t P)
{   return sizeof(Octane) * P + 128; }


__device__ __forceinline__
uint32_t expandBits(uint32_t v)     // 10 → 101000 -> 100001 …
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __forceinline__
uint64_t morton3D(uint32_t x, uint32_t y, uint32_t z)
{
    return (uint64_t)(expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2));
}

__device__ __forceinline__ void atomicAdd_f(float* addr, float v) 
{ 
    atomicAdd(addr, v); 
}

__device__ inline void atomicAdd_vec2(glm::vec2* addr, const glm::vec2& v)
{
    atomicAdd(&(addr->x), v.x);
    atomicAdd(&(addr->y), v.y);
}

__device__ inline void atomicAdd_vec3(glm::vec3* addr, const glm::vec3& v)
{
    atomicAdd(&(addr->x), v.x);
    atomicAdd(&(addr->y), v.y);
    atomicAdd(&(addr->z), v.z);
}