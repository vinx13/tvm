/*
 * Adapted from https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
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

#include <float.h>

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <type_traits>

#include "dtype_float16.h"

#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace vllm {

// Q*K^T operation.
template<int THREAD_GROUP_SIZE, typename Vec, int N>
inline __device__ float qk_dot_(const Vec (&q)[N], const Vec (&k)[N]) {
  using A_vec = typename FloatVec<Vec>::Type;
  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  A_vec qk_vec = mul<A_vec, Vec, Vec>(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    qk_vec = fma(q[ii], k[ii], qk_vec);
  }

  // Finalize the reduction across lanes.
  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
    qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
  }
  return qk;
}

template<typename T, int THREAD_GROUP_SIZE>
struct Qk_dot {
  template<typename Vec, int N>
  static inline __device__ float dot(const Vec (&q)[N], const Vec (&k)[N]) {
    return qk_dot_<THREAD_GROUP_SIZE>(q, k);
  }
};

// Utility function for attention softmax.
template<int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
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
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Broadcast to other threads.
  return __shfl_sync(uint32_t(-1), sum, 0);
}

// Grid: (num_heads, num_seqs).
template<
  typename scalar_t,
  int HEAD_SIZE,
  int BLOCK_SIZE,
  int NUM_THREADS>
__global__ void single_query_cached_kv_attention_kernel(
  scalar_t* __restrict__ out,             // [num_seqs, num_heads, head_size]
  const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
  const scalar_t* __restrict__ k_cache,   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
  const scalar_t* __restrict__ v_cache,   // [num_blocks, num_kv_heads, head_size, block_size]
  const int* __restrict__ head_mapping,   // [num_heads]
  const float scale,
  const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
  const int* __restrict__ context_lens,   // [num_seqs]
  const int max_num_blocks_per_seq,
  const float* __restrict__ alibi_slopes,  // [num_heads]
  const int q_stride,
  const int kv_block_stride,
  const int kv_head_stride) {
  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  // Note: This assumes THREAD_GROUP_SIZE divides NUM_THREADS
  constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE;
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int kv_head_idx = head_mapping[head_idx];
  const int seq_idx = blockIdx.y;
  const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread group
  // fetch or compute 16 bytes at a time.
  // For example, if the size of a thread group is 4 and the data type is half,
  // then the vector size is 16 / (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in the group
  // has 0, 4, 8, ... th vectors of the query, and the second thread has 1, 5, 9, ...
  // th vectors of the query, and so on.
  // NOTE(woosuk): Because q is split from a qkv tensor, it may not be contiguous.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads();

  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(scalar_t);
  float qk_max = -FLT_MAX;

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const int context_len = context_lens[seq_idx];
  const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
    const int physical_block_number = block_table[block_idx];

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the the thread group size is 4, then the first thread in the group
    // has 0, 4, 8, ... th vectors of the key, and the second thread has 1, 5, 9, ... th
    // vectors of the key, and so on.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const scalar_t* k_ptr = k_cache + physical_block_number * kv_block_stride
                                        + kv_head_idx * kv_head_stride
                                        + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;
        k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset],
                                                                  k_vecs);
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - context_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= context_len;
        logits[token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD = (HEAD_SIZE + NUM_ROWS_PER_ITER - 1) / NUM_ROWS_PER_ITER;

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  scalar_t zero_value;
  zero(zero_value);
  for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
    const int physical_block_number = block_table[block_idx];
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx));

    const scalar_t* v_ptr = v_cache + physical_block_number * kv_block_stride
                                    + kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        if (block_idx == num_blocks - 1) {
          // NOTE(woosuk): When v_vec contains the tokens that are out of the context,
          // we should explicitly zero out the values since they may contain NaNs.
          // See https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
#pragma unroll
          for (int j = 0; j <= V_VEC_SIZE; j++) {
            v_vec_ptr[j] = token_idx + j < context_len ? v_vec_ptr[j] : zero_value;
          }
        }
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += __shfl_xor_sync(uint32_t(-1), acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for logits
  // is reused for the output.
  __syncthreads();

  // Perform reduction across warps.
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    __syncthreads();
  }

  // Write the final output.
  if (warp_idx == 0) {
    scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}

}  // namespace vllm

namespace tvm {
namespace runtime {

#define LAUNCH_ATTENTION_KERNEL(T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS)                        \
  cudaFuncSetAttribute(                                                                       \
      vllm::single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>,   \
      cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);                          \
  vllm::single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>        \
  <<<grid, block, shared_mem_size>>>(                                                 \
    out_ptr,                                                                                  \
    query_ptr,                                                                                \
    key_cache_ptr,                                                                            \
    value_cache_ptr,                                                                          \
    head_mapping_ptr,                                                                         \
    scale,                                                                                    \
    block_tables_ptr,                                                                         \
    context_lens_ptr,                                                                         \
    max_num_blocks_per_seq,                                                                   \
    alibi_slopes_ptr,                                                                         \
    q_stride,                                                                                 \
    kv_block_stride,                                                                          \
    kv_head_stride);


template<
  typename T,
  int BLOCK_SIZE,
  int NUM_THREADS = 128>
void single_query_cached_kv_attention_launcher(
                       DLTensor* out,
                       const DLTensor* query,
                       const DLTensor* key_cache,
                       const DLTensor* value_cache,
                       const DLTensor* head_mapping,
                       float scale,
                       const DLTensor* block_tables,
                       const DLTensor* context_lens,
                       int max_context_len) {
  int num_seqs = query->shape[0];
  int num_heads = query->shape[1];
  int head_size = query->shape[2];
  int max_num_blocks_per_seq = block_tables->shape[1];
  int q_stride = query->shape[1] * query->shape[2];

  int kv_head_stride = key_cache->shape[2] * key_cache->shape[3] * key_cache->shape[4];
  int kv_block_stride = kv_head_stride * key_cache->shape[1];

  // int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  // assert(head_size % thread_group_size == 0);
  const float* alibi_slopes_ptr = nullptr;

  T* out_ptr = static_cast<T*>(out->data);
  T* query_ptr = static_cast<T*>(query->data);
  T* key_cache_ptr = static_cast<T*>(key_cache->data);
  T* value_cache_ptr = static_cast<T*>(value_cache->data);
  int* head_mapping_ptr = static_cast<int*>(head_mapping->data);
  int* block_tables_ptr = static_cast<int*>(block_tables->data);
  int* context_lens_ptr = static_cast<int*>(context_lens->data);

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_context_len = ((max_context_len + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_context_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs);
  dim3 block(NUM_THREADS);
  switch (head_size) {
    case 64:
      LAUNCH_ATTENTION_KERNEL(T, 64, BLOCK_SIZE, NUM_THREADS);
      break;
    case 80:
      LAUNCH_ATTENTION_KERNEL(T, 80, BLOCK_SIZE, NUM_THREADS);
      break;
    case 96:
      LAUNCH_ATTENTION_KERNEL(T, 96, BLOCK_SIZE, NUM_THREADS);
      break;
    case 112:
      LAUNCH_ATTENTION_KERNEL(T, 112, BLOCK_SIZE, NUM_THREADS);
      break;
    case 128:
      LAUNCH_ATTENTION_KERNEL(T, 128, BLOCK_SIZE, NUM_THREADS);
      break;
    case 256:
      LAUNCH_ATTENTION_KERNEL(T, 256, BLOCK_SIZE, NUM_THREADS);
      break;
    default:
      // TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
}

#define CALL_KERNEL_LAUNCHER(BLOCK_SIZE)                           \
  single_query_cached_kv_attention_launcher<uint16_t, BLOCK_SIZE>( \
    out,                                                            \
    query,                                                          \
    key_cache,                                                      \
    value_cache,                                                    \
    head_mapping,                                                   \
    scale,                                                          \
    block_tables,                                                   \
    context_lens,                                                   \
    max_context_len);

TVM_REGISTER_GLOBAL("tvm.contrib.vllm.single_query_cached_kv_attention")
    .set_body_typed([](const DLTensor* query,
                       const DLTensor* key_cache,
                       const DLTensor* value_cache,
                       const DLTensor* head_mapping,
                       const DLTensor* block_tables,
                       const DLTensor* context_lens,
                       int block_size,
                       const DLTensor* max_context_len_tensor,  // TODO(masahi): pass integer
                       DLTensor* out) {
        float scale = 1.0 / sqrt(query->shape[2]);
        int max_context_len = static_cast<int*>(max_context_len_tensor->data)[0];

        if (block_size == 8) {
          CALL_KERNEL_LAUNCHER(8);
        } else if (block_size == 16) {
          CALL_KERNEL_LAUNCHER(16);
        } else if (block_size == 32) {
          CALL_KERNEL_LAUNCHER(32);
        } else {
          LOG(FATAL) << "Unsupported block size: " << block_size;
        }
    });
}  // namespace runtime
}  // namespace tvm

#undef WARP_SIZE
#undef MAX
#undef MIN