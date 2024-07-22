#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

#include "attention_dtypes.h"
#include "attention_utils.cuh"

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
  #include "../quantization/fp8/amd/quant_utils.cuh"
typedef __hip_bfloat16 __nv_bfloat16;
#else
  #include "../quantization/fp8/nvidia/quant_utils.cuh"
#endif

#ifndef USE_ROCM
  #define WARP_SIZE 32
#else
  #define WARP_SIZE warpSize
#endif


#include "static_switch.h"
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

namespace vllm {

// Utility function for attention softmax.
template <int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Broadcast to other threads.
  return VLLM_SHFL_SYNC(sum, 0);
}

// TODO(woosuk): Merge the last two dimensions of the grid.
// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE,
          int REUSE_KV_TIMES = 1,
          bool odd_nheads = false,
          int PARTITION_SIZE = 0,std::enable_if_t<!std::is_same<scalar_t, uint16_t>::value, int> = 0>  // Zero means no partitioning.
__device__ void paged_attention_kernel(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions,
                                 // head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_heads,                   // [num_heads]
    const int num_kv_heads,               // [num_kv_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float kv_scale, const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step) {}

// TODO(woosuk): Merge the last two dimensions of the grid.
// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE,
          int REUSE_KV_TIMES = 1,
          bool odd_nheads = false,
          int PARTITION_SIZE = 0,std::enable_if_t<std::is_same<scalar_t, uint16_t>::value, int> = 0>  // Zero means no partitioning.
__device__ void paged_attention_kernel(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions,
                                 // head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_heads,                   // [num_heads]
    const int num_kv_heads,               // [num_kv_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float kv_scale, const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step) {
  const int seq_idx = blockIdx.z;
  const int partition_idx = blockIdx.y;
  const int max_num_partitions = gridDim.y;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const int seq_len = seq_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= seq_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_seq_blocks;
  const int partition_size = USE_PARTITIONING ? PARTITION_SIZE : num_seq_blocks * BLOCK_SIZE;
  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx =
      USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx =
      MIN(start_block_idx + num_blocks_per_partition, num_seq_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx =
      MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS =
      NUM_THREADS / THREAD_GROUP_SIZE;  // Note: This assumes THREAD_GROUP_SIZE
                                        // divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
      DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  // const int warp_idx_vec = thread_idx / WARP_SIZE;
  // int warp_idx =0;
  // asm volatile("v_readfirstlane_b32 %0,%1"
  //               : "=s"(warp_idx)
  //               : "v"(warp_idx_vec)
  //               :);
  // // const int warp_idx = thread_idx / WARP_SIZE;

  // const int lane = thread_idx % WARP_SIZE;

    //const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  int warp_id_vec = threadIdx.x / WARP_SIZE; //warp id in a block
  int warp_idx =0;
  asm volatile("v_readfirstlane_b32 %0,%1"
                : "=s"(warp_idx)
                : "v"(warp_id_vec)
                :);

  // const int head_idx = blockIdx.x;
  // const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  // const float alibi_slope =
  //     alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread
  // group fetch or compute 16 bytes at a time. For example, if the size of a
  // thread group is 4 and the data type is half, then the vector size is 16 /
  // (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(32 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Quant_vec = typename Vec<cache_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in
  // the group has 0, 4, 8, ... th vectors of the query, and the second thread
  // has 1, 5, 9, ... th vectors of the query, and so on. NOTE(woosuk): Because
  // q is split from a qkv tensor, it may not be contiguous.
  // const scalar_t* q_ptr = q + seq_idx * q_stride;
  const scalar_t* q_ptr_offset = q + seq_idx * q_stride;

  __shared__ Q_vec q_vecs[REUSE_KV_TIMES * THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
// #pragma unroll
//   for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
//        i += NUM_THREAD_GROUPS) {
//     const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
//     q_vecs[thread_group_offset][i] =
//         *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
//   }
//   __syncthreads();  // TODO(naed90): possible speedup if this is replaced with a
//                     // memory wall right before we use q_vecs

  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[REUSE_KV_TIMES][2 * NUM_WARPS];
  // float (*red_smem)[2 * NUM_WARPS] = reinterpret_cast<float(*)[2 * NUM_WARPS]>(&shared_mem[10*1024]);

  // __shared__ char shared_mem[12 * 1024];
  // float* logits = reinterpret_cast<float*>(shared_mem);
  // __shared__ float red_smem[REUSE_KV_TIMES][2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(cache_t);
  float qk_max[REUSE_KV_TIMES];

  for(int reuse_kv_idx=0; reuse_kv_idx<REUSE_KV_TIMES; reuse_kv_idx++) {
      qk_max[reuse_kv_idx] = -FLT_MAX;
  }
   
  const int num_blocks_per_kv = ((num_queries_per_kv + REUSE_KV_TIMES -1) / REUSE_KV_TIMES);
  const int head_idx_soffset = (blockIdx.x / num_blocks_per_kv) * num_queries_per_kv + (blockIdx.x % num_blocks_per_kv) * REUSE_KV_TIMES;
  const int kv_head_idx = head_idx_soffset / num_queries_per_kv;
  const int q_boundary = (kv_head_idx + 1)* num_queries_per_kv;

  #pragma unroll
  for(int reuse_kv_idx=0; reuse_kv_idx<REUSE_KV_TIMES; reuse_kv_idx++) {
    const int head_idx = head_idx_soffset + reuse_kv_idx;//blockIdx.x * REUSE_KV_TIMES + reuse_kv_idx;
    const scalar_t* q_ptr = q_ptr_offset + head_idx * HEAD_SIZE;
    #pragma unroll
    for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {
      const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
      q_vecs[reuse_kv_idx*THREAD_GROUP_SIZE + thread_group_offset][i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
    }
  }
  __syncthreads(); // TODO(naed90): possible speedup if this is replaced with a memory wall right before we use q_vecs

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    // For blocksparse attention: skip computation on blocks that are not
    // attended
    for(int reuse_kv_idx=0; reuse_kv_idx<REUSE_KV_TIMES; reuse_kv_idx++) {
    const int head_idx = head_idx_soffset + reuse_kv_idx;//blockIdx.x * REUSE_KV_TIMES + reuse_kv_idx;
    if(!odd_nheads || head_idx < q_boundary) {
        // blocksparse specific vars
    int bs_block_offset;
    int q_bs_block_id;
    if constexpr (IS_BLOCK_SPARSE) { 
      // const int num_blocksparse_blocks = DIVIDE_ROUND_UP(seq_len,
      // blocksparse_block_size);
      q_bs_block_id = (seq_len - 1) / blocksparse_block_size;
      if (blocksparse_head_sliding_step >= 0)
        // sliding on q heads
        bs_block_offset =
            (tp_rank * num_heads + head_idx) * blocksparse_head_sliding_step + 1;
      else
        // sliding on kv heads
        bs_block_offset = (tp_rank * num_kv_heads + kv_head_idx) *
                              (-blocksparse_head_sliding_step) +
                          1;
    }
    if constexpr (IS_BLOCK_SPARSE) {
      const int k_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
      const bool is_remote =
          ((k_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0);
      const bool is_local =
          (k_bs_block_id > q_bs_block_id - blocksparse_local_blocks);
      if (!is_remote && !is_local) {
        for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
          const int physical_block_offset =
              (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
          const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

          if (thread_group_offset == 0) {
            // NOTE(linxihui): assign very large number to skipped tokens to
            // avoid contribution to the sumexp softmax normalizer. This will
            // not be used at computing sum(softmax*v) as the blocks will be
            // skipped.
            logits[token_idx - start_token_idx] = -FLT_MAX;
          }
        }
        continue;
      }
    }
    const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];
    const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the the thread group size is 4, then the first thread in
    // the group has 0, 4, 8, ... th vectors of the key, and the second thread
    // has 1, 5, 9, ... th vectors of the key, and so on.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];
      if(reuse_kv_idx == 0) {
        #pragma unroll
        for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
          const cache_t* k_ptr =
              k_cache + physical_block_number * kv_block_stride +
              kv_head_idx * kv_head_stride + physical_block_offset * x;
          const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
          const int offset1 = (vec_idx * VEC_SIZE) / x;
          const int offset2 = (vec_idx * VEC_SIZE) % x;

          if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
            k_vecs[j] = *reinterpret_cast<const K_vec*>(
                k_ptr + offset1 * BLOCK_SIZE * x + offset2);
          } else {
            // Vector conversion from Quant_vec to K_vec.
            Quant_vec k_vec_quant = *reinterpret_cast<const Quant_vec*>(
                k_ptr + offset1 * BLOCK_SIZE * x + offset2);
            k_vecs[j] = fp8::scaled_convert<K_vec, Quant_vec, KV_DTYPE>(
                k_vec_quant, kv_scale);
          }
        }
      }
      __builtin_amdgcn_sched_barrier(0);
      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[reuse_kv_idx*THREAD_GROUP_SIZE + thread_group_offset], k_vecs);
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - seq_len + 1) : 0;
      __builtin_amdgcn_sched_barrier(0);
      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= seq_len;
        logits[(reuse_kv_idx * partition_size) + (token_idx - start_token_idx)] = mask ? 0.f : qk;
        // Update the max value.
        qk_max[reuse_kv_idx] = mask ? qk_max[reuse_kv_idx] : fmaxf(qk_max[reuse_kv_idx], qk);
      }
    }
  }
  }
  }
  // Get the sum of the exp values.
  float exp_sum[REUSE_KV_TIMES] = {0.f};

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
  for(int reuse_kv_idx=0; reuse_kv_idx<REUSE_KV_TIMES; reuse_kv_idx++) {
    const int head_idx = head_idx_soffset + reuse_kv_idx;
    if(!odd_nheads || head_idx < q_boundary) {
      #pragma unroll
      for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
        qk_max[reuse_kv_idx] = fmaxf(qk_max[reuse_kv_idx], VLLM_SHFL_XOR_SYNC(qk_max[reuse_kv_idx], mask));
      }
      if (lane == 0) {
        red_smem[reuse_kv_idx][warp_idx] = qk_max[reuse_kv_idx];
      }
      __syncthreads();

      // TODO(woosuk): Refactor this part.
      // Get the max qk value for the sequence.
      qk_max[reuse_kv_idx] = lane < NUM_WARPS ? red_smem[reuse_kv_idx][lane] : -FLT_MAX;
    #pragma unroll
      for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
        qk_max[reuse_kv_idx] = fmaxf(qk_max[reuse_kv_idx], VLLM_SHFL_XOR_SYNC(qk_max[reuse_kv_idx], mask));
      }
      // Broadcast the max qk value to all threads.
      qk_max[reuse_kv_idx] = VLLM_SHFL_SYNC(qk_max[reuse_kv_idx], 0);

      for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
        float val = __expf(logits[(reuse_kv_idx * partition_size) + i] - qk_max[reuse_kv_idx]);
        logits[(reuse_kv_idx * partition_size) + i] = val;
        exp_sum[reuse_kv_idx] += val;
      }
      exp_sum[reuse_kv_idx] = block_sum<NUM_WARPS>(&red_smem[reuse_kv_idx][NUM_WARPS], exp_sum[reuse_kv_idx]);

      // Compute softmax.
      const float inv_sum = __fdividef(1.f, exp_sum[reuse_kv_idx] + 1e-6f);
      for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
        logits[(reuse_kv_idx * partition_size) + i] *= inv_sum;
      }
      __syncthreads();

      // If partitioning is enabled, store the max logit and exp_sum.
      if (USE_PARTITIONING && thread_idx == 0) {
        float* max_logits_ptr = max_logits +
                                seq_idx * num_heads * max_num_partitions +
                                head_idx * max_num_partitions + partition_idx;
        *max_logits_ptr = qk_max[reuse_kv_idx];
        float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions +
                              head_idx * max_num_partitions + partition_idx;
        *exp_sums_ptr = exp_sum[reuse_kv_idx];
      }
    }
  }
  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using V_quant_vec = typename Vec<cache_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD =
      DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[REUSE_KV_TIMES][NUM_ROWS_PER_THREAD];

  #pragma unroll
  for(int reuse_kv_idx=0; reuse_kv_idx<REUSE_KV_TIMES; reuse_kv_idx++) {
    #pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        accs[reuse_kv_idx][i] = 0.f;
    }
  }
  scalar_t zero_value;
  zero(zero_value);
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    V_vec v_vec;
    for(int reuse_kv_idx=0; reuse_kv_idx<REUSE_KV_TIMES; reuse_kv_idx++) {
      // NOTE(woosuk): The block number is stored in int32. However, we cast it to
      // int64 because int32 can lead to overflow when this variable is multiplied
      // by large numbers (e.g., kv_block_stride).
      // For blocksparse attention: skip computation on blocks that are not
      // attended
      // blocksparse specific vars
      const int head_idx = head_idx_soffset + reuse_kv_idx;
      int bs_block_offset;
      int q_bs_block_id;
      if constexpr (IS_BLOCK_SPARSE) {
        // const int num_blocksparse_blocks = DIVIDE_ROUND_UP(seq_len,
        // blocksparse_block_size);
        q_bs_block_id = (seq_len - 1) / blocksparse_block_size;
        if (blocksparse_head_sliding_step >= 0)
          // sliding on q heads
          bs_block_offset =
              (tp_rank * num_heads + head_idx) * blocksparse_head_sliding_step + 1;
        else
          // sliding on kv heads
          bs_block_offset = (tp_rank * num_kv_heads + kv_head_idx) *
                                (-blocksparse_head_sliding_step) +
                            1;
      }
      if constexpr (IS_BLOCK_SPARSE) {
        int v_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
        if (!((v_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0) &&
            !((v_bs_block_id > q_bs_block_id - blocksparse_local_blocks))) {
          continue;
        }
      }
      if(!odd_nheads || head_idx < q_boundary) {


      const cache_t* v_ptr = v_cache + physical_block_number * kv_block_stride
                                   + kv_head_idx * kv_head_stride;

     from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + (reuse_kv_idx * partition_size) +  token_idx - start_token_idx));
      // scalar_t* logits_vec_ptr = reinterpret_cast<scalar_t*>(&logits_vec);
      // for(int i=0;i<8;++i){
      //   from_float(*(logits_vec_ptr+i), 1000);
      // }

      if(reuse_kv_idx==0) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;

        if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
          v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        } else {
          V_quant_vec v_quant_vec =
              *reinterpret_cast<const V_quant_vec*>(v_ptr + offset);
          // Vector conversion from V_quant_vec to V_vec.
          v_vec = fp8::scaled_convert<V_vec, V_quant_vec, KV_DTYPE>(v_quant_vec,
                                                                    kv_scale);
        }
        if (block_idx == num_seq_blocks - 1) {
          // NOTE(woosuk): When v_vec contains the tokens that are out of the
          // context, we should explicitly zero out the values since they may
          // contain NaNs. See
          // https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] = token_idx + j < seq_len ? v_vec_ptr[j] : zero_value;
          }
        }
        // if(threadIdx.x==0){
        //   scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
        //   scalar_t* logits_vec_ptr = reinterpret_cast<scalar_t*>(&logits_vec);
        //   for(int i=0;i<8;++i){
        //     printf("v_vec[%d] = %f\n",i, half_to_float(v_vec_ptr[i]));
        //     // from_float(*(v_vec_ptr + i), 1000);
        //   }
        //   for(int i=0;i<8;++i){
        //     printf("logits_vec[%d] = %f\n",i,half_to_float(logits_vec_ptr[i]));
        //     // from_float(*(logits_vec_ptr + i), 1000);
        //   }
        // }
        // accs[reuse_kv_idx][i] += dot(logits_vec, v_vec);
      }
      } 
        accs[reuse_kv_idx][i] += dot(logits_vec, v_vec);
      }
      }
    }
  }

  // Perform reduction within each warp.
  #pragma unroll
  for(int reuse_kv_idx=0; reuse_kv_idx<REUSE_KV_TIMES; reuse_kv_idx++) {
    int head_idx = head_idx_soffset + reuse_kv_idx;

    if(!odd_nheads || head_idx < q_boundary) {
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[reuse_kv_idx][i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += VLLM_SHFL_XOR_SYNC(acc, mask);
    }
    accs[reuse_kv_idx][i] = acc;
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for
  // logits is reused for the output.
  __syncthreads();

  // Perform reduction across warps.
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
       float* dst = &out_smem[(reuse_kv_idx * (NUM_WARPS / 2) * HEAD_SIZE) + (warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[reuse_kv_idx][i];
        }
      }
    }
    __syncthreads();

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float* src = &out_smem[(reuse_kv_idx * (NUM_WARPS / 2) * HEAD_SIZE) + warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[reuse_kv_idx][i] += src[row_idx];
        }
      }
    }
    __syncthreads();
  }

  // Write the final output.
  if (warp_idx == 0) {
    scalar_t* out_ptr =
        out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[reuse_kv_idx][i]);
      }
    }
  }
  }
  }
}


// Grid: (num_heads, num_seqs, 1).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          int REUSE_KV_TIMES = 1,
          bool IS_BLOCK_SPARSE,
          bool odd_nheads = false>
__global__ __launch_bounds__(256,1) void paged_attention_v1_kernel(
    scalar_t* __restrict__ out,           // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_heads,               // [num_heads]    
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float kv_scale, const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step) {
      paged_attention_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,
                          KV_DTYPE, IS_BLOCK_SPARSE, REUSE_KV_TIMES, odd_nheads>(
        /* exp_sums */ nullptr, /* max_logits */ nullptr, out, q, k_cache,
        v_cache, num_heads, num_kv_heads, scale, block_tables, seq_lens,
        max_num_blocks_per_seq, alibi_slopes, q_stride, kv_block_stride,
        kv_head_stride, kv_scale, tp_rank, blocksparse_local_blocks,
        blocksparse_vert_stride, blocksparse_block_size,
        blocksparse_head_sliding_step);
}

// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE,
          int REUSE_KV_TIMES,
          int PARTITION_SIZE,
          bool odd_nheads = false>
__global__ __launch_bounds__(256,1) void paged_attention_v2_kernel(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,       // [num_seqs, num_heads,
                                          // max_num_partitions]
    scalar_t* __restrict__ tmp_out,       // [num_seqs, num_heads,
                                          // max_num_partitions, head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_heads,               // [num_heads]                                      
    const int num_kv_heads,               // [num_kv_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float kv_scale, const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step) {
        paged_attention_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,
                         KV_DTYPE, IS_BLOCK_SPARSE, REUSE_KV_TIMES, odd_nheads, PARTITION_SIZE>(
          exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_heads, num_kv_heads, scale,
          block_tables, seq_lens, max_num_blocks_per_seq, alibi_slopes, q_stride,
          kv_block_stride, kv_head_stride, kv_scale, tp_rank,
          blocksparse_local_blocks, blocksparse_vert_stride, blocksparse_block_size,
          blocksparse_head_sliding_step);
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE>
__global__ __launch_bounds__(256,1) void paged_attention_v2_reduce_kernel(
    scalar_t* __restrict__ out,            // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs, num_heads,
                                           // max_num_partitions]
    const float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                           // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_heads,
                                           // max_num_partitions, head_size]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_partitions) {
  const int num_heads = gridDim.x;
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int seq_len = seq_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(seq_len, PARTITION_SIZE);
  if (num_partitions == 1) {
    // No need to reduce. Only copy tmp_out to out.
    scalar_t* out_ptr =
        out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    const scalar_t* tmp_out_ptr =
        tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) {
      out_ptr[i] = tmp_out_ptr[i];
    }
    // Terminate the thread block.
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  // Size: 2 * num_partitions.
  extern __shared__ char shared_mem[];
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // Load max logits to shared memory.
  float* shared_max_logits = reinterpret_cast<float*>(shared_mem);
  const float* max_logits_ptr = max_logits +
                                seq_idx * num_heads * max_num_partitions +
                                head_idx * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = fmaxf(max_logit, l);
  }
  __syncthreads();

  // Get the global max logit.
  // Reduce within the warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, VLLM_SHFL_XOR_SYNC(max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  __syncthreads();
  // Reduce across warps.
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, VLLM_SHFL_XOR_SYNC(max_logit, mask));
  }
  // Broadcast the max value to all threads.
  max_logit = VLLM_SHFL_SYNC(max_logit, 0);

  // Load rescaled exp sums to shared memory.
  float* shared_exp_sums =
      reinterpret_cast<float*>(shared_mem + sizeof(float) * num_partitions);
  const float* exp_sums_ptr = exp_sums +
                              seq_idx * num_heads * max_num_partitions +
                              head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    float l = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * expf(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  __syncthreads();
  global_exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum);
  const float inv_global_exp_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);

  // Aggregate tmp_out to out.
  const scalar_t* tmp_out_ptr =
      tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
      head_idx * max_num_partitions * HEAD_SIZE;
  scalar_t* out_ptr =
      out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j) {
      acc += to_float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] *
             inv_global_exp_sum;
    }
    from_float(out_ptr[i], acc);
  }
}

}  // namespace vllm

#define LAUNCH_PAGED_ATTENTION_V1(HEAD_SIZE)                                \
  VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(                     \
      ((void*)vllm::paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE,        \
                                              BLOCK_SIZE, NUM_THREADS,      \
                                              KV_DTYPE, REUSE_KV_TIMES, IS_BLOCK_SPARSE, odd_nheads>),  \
      shared_mem_size);                                                     \
 hipLaunchKernelGGL(( vllm::paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE,        \
                                  NUM_THREADS, KV_DTYPE, REUSE_KV_TIMES, IS_BLOCK_SPARSE, odd_nheads>)   \
      , dim3(grid), dim3(block), shared_mem_size, stream,                            \
          out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_heads, num_kv_heads, \
          scale, block_tables_ptr, seq_lens_ptr, max_num_blocks_per_seq,    \
          alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,      \
          kv_scale, tp_rank, blocksparse_local_blocks,                      \
          blocksparse_vert_stride, blocksparse_block_size,                  \
          blocksparse_head_sliding_step);

// #define LAUNCH_PAGED_ATTENTION_V1(HEAD_SIZE)                                \
// vllm::paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE,        \
//                                   NUM_THREADS, KV_DTYPE, REUSE_KV_TIMES, IS_BLOCK_SPARSE, odd_nheads>   \
//       <<<dim3(grid), dim3(block)>>>(                           \
//           out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_heads, num_kv_heads, \
//           scale, block_tables_ptr, seq_lens_ptr, max_num_blocks_per_seq,    \
//           alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,      \
//           kv_scale, tp_rank, blocksparse_local_blocks,                      \
//           blocksparse_vert_stride, blocksparse_block_size,                  \
//           blocksparse_head_sliding_step);

// TODO(woosuk): Tune NUM_THREADS.
template <typename T, typename CACHE_T, int BLOCK_SIZE,
          vllm::Fp8KVCacheDataType KV_DTYPE, bool IS_BLOCK_SPARSE>
void paged_attention_v1_launcher(
    torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes, float kv_scale,
    const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);
  int num_threads = 128;
  if(num_heads!=num_kv_heads){
    num_threads =256;
  }
  int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
          : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  CACHE_T* key_cache_ptr = reinterpret_cast<CACHE_T*>(key_cache.data_ptr());
  CACHE_T* value_cache_ptr = reinterpret_cast<CACHE_T*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* seq_lens_ptr = seq_lens.data_ptr<int>();

  int padded_max_seq_len = DIVIDE_ROUND_UP(max_seq_len, BLOCK_SIZE) * BLOCK_SIZE;
  REUSEKV_SWITCH_V1(num_heads * num_seqs , [&] {
    BOOL_SWITCH((num_heads/num_kv_heads % REUSE_KV_TIMES != 0), odd_nheads, [&] {
      HEADSIZE_SWITCH(head_size, [&] {
        NUM_THREADS_SWITCH(num_threads, [&] {
          OPT_SWITCH(num_heads == num_kv_heads, [&] {
          constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
          int logits_size =  REUSE_KV_TIMES*padded_max_seq_len * sizeof(float);
          int outputs_size =  REUSE_KV_TIMES*(NUM_WARPS / 2) * head_size * sizeof(float);
          // Python-side check in vllm.worker.worker._check_if_can_support_max_seq_len
          // Keep that in sync with the logic here!
          int shared_mem_size = ::max(logits_size, outputs_size);
          if(num_heads == num_kv_heads) shared_mem_size = ::max(12 * 1024, shared_mem_size);
          // int shared_mem_size = ::max(31*1024, ::max(logits_size, outputs_size));
          // std::cout<<"shared_mem_size = "<<shared_mem_size<<std::endl;
          dim3 grid((num_heads/num_kv_heads + REUSE_KV_TIMES - 1) / REUSE_KV_TIMES*num_kv_heads, 1, num_seqs);
          dim3 block(NUM_THREADS);
          const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(query));
          const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
          LAUNCH_PAGED_ATTENTION_V1(HEAD_SIZE);
          });
        });
      });
    });
  }); 
}

#define CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, KV_DTYPE, IS_BLOCK_SPARSE)  \
  paged_attention_v1_launcher<T, CACHE_T, BLOCK_SIZE, KV_DTYPE,              \
                              IS_BLOCK_SPARSE>(                              \
      out, query, key_cache, value_cache, num_kv_heads, scale, block_tables, \
      seq_lens, max_seq_len, alibi_slopes, kv_scale, tp_rank,                \
      blocksparse_local_blocks, blocksparse_vert_stride,                     \
      blocksparse_block_size, blocksparse_head_sliding_step);

#define CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE) \
  switch (is_block_sparse) {                                               \
    case true:                                                             \
      CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, true);     \
      break;                                                               \
    case false:                                                            \
      CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, false);    \
      break;                                                               \
  }

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_V1_LAUNCHER_BLOCK_SIZE(T, CACHE_T, KV_DTYPE)         \
  switch (block_size) {                                           \
    case 16:                                                      \
      CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, 16, KV_DTYPE);        \
      break;                                                      \
    default:                                                      \
      TORCH_CHECK(false, "Unsupported block size: ", block_size); \
      break;                                                      \
  }

// // NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// // 1, 2, 4, 64, 128, 256.
// #define CALL_V1_LAUNCHER_BLOCK_SIZE(T, CACHE_T, KV_DTYPE)         \
//   switch (block_size) {                                           \
//     case 16:                                                      \
//       CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, 16, KV_DTYPE);        \
//       break;                                                      \
//       TORCH_CHECK(false, "Unsupported block size: ", block_size); \
//       break;                                                      \
//   }

void paged_attention_v1(
    torch::Tensor& out,    // [num_seqs, num_heads, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,       // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads,  // [num_heads]
    double scale,
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,      // [num_seqs]
    int64_t block_size, int64_t max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, double kv_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step) {
  const bool is_block_sparse = (blocksparse_vert_stride > 1);

  DISPATCH_BY_KV_CACHE_DTYPE(query.dtype(), kv_cache_dtype,
                             CALL_V1_LAUNCHER_BLOCK_SIZE)
}

#define LAUNCH_PAGED_ATTENTION_V2(HEAD_SIZE)                                   \
 hipLaunchKernelGGL(( vllm::paged_attention_v2_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE,           \
                                  NUM_THREADS, KV_DTYPE, IS_BLOCK_SPARSE,      \
                                  REUSE_KV_TIMES, PARTITION_SIZE, odd_nheads>)                              \
      , dim3(grid), dim3(block), shared_mem_size, stream,                               \
          exp_sums_ptr, max_logits_ptr, tmp_out_ptr, query_ptr, key_cache_ptr, \
          value_cache_ptr, num_heads, num_kv_heads, scale, block_tables_ptr,              \
          seq_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride,    \
          kv_block_stride, kv_head_stride, kv_scale, tp_rank,                  \
          blocksparse_local_blocks, blocksparse_vert_stride,                   \
          blocksparse_block_size, blocksparse_head_sliding_step);              \
 hipLaunchKernelGGL(( vllm::paged_attention_v2_reduce_kernel<T, HEAD_SIZE, NUM_THREADS,            \
                                         PARTITION_SIZE>)                       \
      , dim3(reduce_grid), dim3(block), reduce_shared_mem_size, stream,                 \
          out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, seq_lens_ptr,    \
          max_num_partitions);

template <typename T, typename CACHE_T, int BLOCK_SIZE,
          vllm::Fp8KVCacheDataType KV_DTYPE, bool IS_BLOCK_SPARSE,
          int NUM_THREADS = 256, int PARTITION_SIZE = 512>
void paged_attention_v2_launcher(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes, float kv_scale,
    const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
          : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  float* exp_sums_ptr = reinterpret_cast<float*>(exp_sums.data_ptr());
  float* max_logits_ptr = reinterpret_cast<float*>(max_logits.data_ptr());
  T* tmp_out_ptr = reinterpret_cast<T*>(tmp_out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  CACHE_T* key_cache_ptr = reinterpret_cast<CACHE_T*>(key_cache.data_ptr());
  CACHE_T* value_cache_ptr = reinterpret_cast<CACHE_T*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* seq_lens_ptr = seq_lens.data_ptr<int>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int max_num_partitions = DIVIDE_ROUND_UP(max_seq_len, PARTITION_SIZE);
  REUSEKV_SWITCH(num_heads * max_num_partitions * num_seqs , [&] {
    BOOL_SWITCH((num_heads/num_kv_heads % REUSE_KV_TIMES != 0), odd_nheads, [&] {
      HEADSIZE_SWITCH(head_size, [&] {
        OPT_SWITCH(num_heads == num_kv_heads, [&] {
        int logits_size = REUSE_KV_TIMES*PARTITION_SIZE * sizeof(float);
        int outputs_size = REUSE_KV_TIMES*(NUM_WARPS / 2) * head_size * sizeof(float);

        // For paged attention v2 kernel.
        // dim3 grid(num_heads, max_num_partitions, num_seqs);

        dim3 grid;
        grid.x = (num_heads/num_kv_heads + REUSE_KV_TIMES -1)/REUSE_KV_TIMES * num_kv_heads;
        grid.y = max_num_partitions;
        grid.z = num_seqs;
        // int shared_mem_size = ::max(1024*32, ::max(logits_size, outputs_size));
        int shared_mem_size = ::max(logits_size, outputs_size);
        // For paged attention v2 reduce kernel.
        dim3 reduce_grid(num_heads, num_seqs);
        int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);
        dim3 block(NUM_THREADS);
        const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(query));
        const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
        LAUNCH_PAGED_ATTENTION_V2(HEAD_SIZE);
        });
      });
    });
  });
}

#define CALL_V2_LAUNCHER(T, CACHE_T, BLOCK_SIZE, KV_DTYPE, IS_BLOCK_SPARSE)   \
  paged_attention_v2_launcher<T, CACHE_T, BLOCK_SIZE, KV_DTYPE,               \
                              IS_BLOCK_SPARSE>(                               \
      out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,      \
      num_kv_heads, scale, block_tables, seq_lens, max_seq_len, alibi_slopes, \
      kv_scale, tp_rank, blocksparse_local_blocks, blocksparse_vert_stride,   \
      blocksparse_block_size, blocksparse_head_sliding_step);

#define CALL_V2_LAUNCHER_SPARSITY(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE) \
  switch (is_block_sparse) {                                               \
    case true:                                                             \
      CALL_V2_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, true);     \
      break;                                                               \
    case false:                                                            \
      CALL_V2_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, false);    \
      break;                                                               \
  }

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_V2_LAUNCHER_BLOCK_SIZE(T, CACHE_T, KV_DTYPE)         \
  switch (block_size) {                                           \
    case 16:                                                      \
      CALL_V2_LAUNCHER_SPARSITY(T, CACHE_T, 16, KV_DTYPE);        \
      break;                                                      \
    default:                                                      \
      TORCH_CHECK(false, "Unsupported block size: ", block_size); \
      break;                                                      \
  }

// // NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// // 1, 2, 4, 64, 128, 256.
// #define CALL_V2_LAUNCHER_BLOCK_SIZE(T, CACHE_T, KV_DTYPE)         \
//   switch (block_size) {                                           \
//     case 16:                                                      \
//       CALL_V2_LAUNCHER_SPARSITY(T, CACHE_T, 16, KV_DTYPE);        \
//       break;                                                      \
//       TORCH_CHECK(false, "Unsupported block size: ", block_size); \
//       break;                                                      \
//   }

void paged_attention_v2(
    torch::Tensor& out,         // [num_seqs, num_heads, head_size]
    torch::Tensor& exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& max_logits,  // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor&
        tmp_out,  // [num_seqs, num_heads, max_num_partitions, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,       // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads,  // [num_heads]
    double scale,
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,      // [num_seqs]
    int64_t block_size, int64_t max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, double kv_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step) {
  const bool is_block_sparse = (blocksparse_vert_stride > 1);
  DISPATCH_BY_KV_CACHE_DTYPE(query.dtype(), kv_cache_dtype,
                             CALL_V2_LAUNCHER_BLOCK_SIZE)
}

#undef WARP_SIZE
#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP
