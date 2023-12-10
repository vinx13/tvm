/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <optional>
#include <string>

#include "../../../3rdparty/cutlass_fpA_intB_gemm/cutlass/include/cutlass/half.h"
// clang-format off
// theses headers can't be reordered
#include "../../../3rdparty/cutlass_fpA_intB_gemm/cutlass/include/cutlass/numeric_types.h"
#include "../../../3rdparty/cutlass_fpA_intB_gemm/cutlass/include/cutlass/integer_subbyte.h"
// clang-format on
#include "../../../3rdparty/cutlass_fpA_intB_gemm/cutlass_kernels/moe_gemm/moe_gemm_kernels.h"

void compute_total_rows_before_expert(const int* sorted_indices, const int total_indices,
                                      const int num_experts, int64_t* total_rows_before_expert,
                                      cudaStream_t stream);

namespace fastertransformer {

template <typename T, typename WeightType>
void moe_gemm_bias_act(const T* A, const WeightType* B, const T* weight_scales, const T* biases,
                       T* C, int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n,
                       int64_t gemm_k, int num_experts, std::optional<std::string> activation,
                       cudaStream_t stream);
}

namespace tvm {
namespace runtime {

TVM_REGISTER_GLOBAL("cutlass.moe_gemm_f16f16")
    .set_body_typed([](NDArray x, NDArray weight, NDArray total_rows_before_expert,
                       int64_t total_rows, int64_t n, int64_t k, int64_t num_experts, NDArray out) {
      LOG(INFO) << "GEMM MOE F16F16";
      LOG(INFO) << "x: " << x->data << " weight: " << weight->data
                << " total_rows_before_expert: " << total_rows_before_expert->data
                << " total_rows: " << total_rows << " n: " << n << " k: " << k
                << " num_experts: " << num_experts << " out: " << out->data;
      //   using half = cutlass::half_t;
      fastertransformer::moe_gemm_bias_act<half, half>(
          reinterpret_cast<half*>(x->data), reinterpret_cast<half*>(weight->data), nullptr, nullptr,
          reinterpret_cast<half*>(out->data),
          reinterpret_cast<int64_t*>(total_rows_before_expert->data), total_rows, n, k, num_experts,
          std::nullopt,
          /*stream=*/nullptr /*FIXME*/);
      LOG(INFO) << "MOE OK";
    });

TVM_REGISTER_GLOBAL("cutlass.moe_gemm_s4f16")
    .set_body_typed([](NDArray x, NDArray weight, NDArray scales, NDArray total_rows_before_expert,
                       int64_t total_rows, int64_t n, int64_t k, int64_t num_experts, NDArray out) {
      fastertransformer::moe_gemm_bias_act<half, cutlass::uint4b_t>(
          reinterpret_cast<half*>(x->data), reinterpret_cast<cutlass::uint4b_t*>(weight->data),
          reinterpret_cast<half*>(scales->data), nullptr, reinterpret_cast<half*>(out->data),
          reinterpret_cast<int64_t*>(total_rows_before_expert->data), total_rows, n, k, num_experts,
          std::nullopt,
          /*stream=*/nullptr /*FIXME*/);
    });

TVM_REGISTER_GLOBAL("moe_compute_rows_before")
    .set_body_typed([](NDArray sorted_indices, NDArray total_rows_before_expert) {
      CHECK(sorted_indices->dtype.code == kDLInt && sorted_indices->dtype.bits == 32);
      CHECK(total_rows_before_expert->dtype.code == kDLInt &&
            total_rows_before_expert->dtype.bits == 64);
      CHECK(sorted_indices->ndim == 1);
      CHECK(total_rows_before_expert->ndim == 1);

      int num_experts = total_rows_before_expert->shape[0];
      compute_total_rows_before_expert(
          reinterpret_cast<int*>(sorted_indices->data), sorted_indices->shape[0], num_experts,
          reinterpret_cast<int64_t*>(total_rows_before_expert->data), nullptr);
    });

}  // namespace runtime
}  // namespace tvm
