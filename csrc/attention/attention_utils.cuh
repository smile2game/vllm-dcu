/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "../cuda_compat.h"
#include "attention_dtypes.h"

#include <float.h>
#include <type_traits>

namespace vllm {

inline __device__ void v_dot2_f32_f16(float& a, const uint32_t &  b,const uint32_t &  c) {
  asm volatile("v_dot2_f32_f16 %0, %1, %2, %0;": "=v"(a): "v"(b), "v"(c), "0"(a));
}

inline __device__ void v_pk_fma_f16(uint32_t& a, const uint32_t &  b,const uint32_t &  c){
   asm volatile("v_pk_fma_f16 %0, %1, %2, %3;": "=v"(a) : "v"(b), "v"(c), "v"(a));
}

inline __device__ void v_dot2_f32_f16(float& a,const uint2 &  b,const uint2 &  c) {
  v_dot2_f32_f16(a, b.x, c.x);
  v_dot2_f32_f16(a, b.y, c.y);
}

inline __device__ void v_dot2_f32_f16(float& a,const uint4 &  b,const uint4 &  c) {
  v_dot2_f32_f16(a, b.x, c.x);
  v_dot2_f32_f16(a, b.y, c.y);
  v_dot2_f32_f16(a, b.z, c.z);
  v_dot2_f32_f16(a, b.w, c.w);
}

inline __device__ float add_half2(uint32_t a){
 union {
    uint32_t u32;
    half u16[2];
  } tmp;
  tmp.u32=a;
  return static_cast<float>(tmp.u16[0]+tmp.u16[1]);
}

inline __device__ void v_pk_fma_f16x8(float& a,const uint4 &  b,const uint4 &  c) {
  uint32_t tmp = mul<uint32_t, uint32_t, uint32_t>(b.x,c.x);
  v_pk_fma_f16(tmp,b.y,c.y);
  v_pk_fma_f16(tmp,b.z,c.z);
  v_pk_fma_f16(tmp,b.w,c.w);
  a+=add_half2(tmp);
}

// Q*K^T operation. fp16
// template <int THREAD_GROUP_SIZE, typename Vec, int N, typename scalar_t, std::enable_if_t<std::is_same<scalar_t, uint16_t>::value, int> = 0>
template <int THREAD_GROUP_SIZE, typename Vec, int N>
inline __device__ float qk_dot_(const Vec (&q)[N], const Vec (&k)[N]) {
  
  float qk =0;
  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  #pragma unroll
  for (int ii = 0; ii < N; ++ii) {
    v_dot2_f32_f16(qk,q[ii],k[ii]);
  }
  // Finalize the reduction across lanes.
#pragma unroll
  for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
    qk += VLLM_SHFL_XOR_SYNC(qk, mask);
  }
  return qk;
}

// Q*K^T operation. //bf16
// template <int THREAD_GROUP_SIZE, typename Vec, int N, typename scalar_t, std::enable_if_t<!std::is_same<scalar_t, uint16_t>::value, int> = 0>
// inline __device__ float qk_dot_(const Vec (&q)[N], const Vec (&k)[N]) {

//   using A_vec = typename FloatVec<Vec>::Type;
//   A_vec qk_vec = mul<A_vec, Vec, Vec>(q[0], k[0]);
//   #pragma unroll
//   for (int ii = 1; ii < N; ++ii) {
//     qk_vec = fma(q[ii], k[ii], qk_vec);
//   }
//   float qk = sum(qk_vec);
//   // Finalize the reduction across lanes.
// #pragma unroll
//   for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
//     qk += VLLM_SHFL_XOR_SYNC(qk, mask);
//   }
//   return qk;
// }


template <typename T, int THREAD_GROUP_SIZE>
struct Qk_dot {
  template <typename Vec, int N>
  static inline __device__ float dot(const Vec (&q)[N], const Vec (&k)[N]) {
    return qk_dot_<THREAD_GROUP_SIZE>(q, k);
  }
};

}  // namespace vllm
