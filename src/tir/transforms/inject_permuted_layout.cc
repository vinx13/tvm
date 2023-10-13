/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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

/*!
 * \file inject_permuted_layout.cc
 * \brief The pass for inject permuted layout.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../../runtime/thread_storage_scope.h"
#include "../../support/utils.h"
#include "ir_utils.h"
#include "simplify.h"

namespace tvm {
namespace tir {

using namespace arith;
using namespace runtime;

class PermutedLayoutInjector : private IRMutatorWithAnalyzer {
 public:
  static PrimFunc Transfrom(PrimFunc func) {
    Analyzer analyzer;
    auto fptr = func.CopyOnWrite();
    fptr->body = PermutedLayoutInjector(&analyzer)(std::move(fptr->body));
    return func;
  }

 private:
  explicit PermutedLayoutInjector(Analyzer* analyzer) : IRMutatorWithAnalyzer(analyzer) {}

  using IRMutatorWithAnalyzer::VisitExpr_;
  using IRMutatorWithAnalyzer::VisitStmt_;

  Array<PrimExpr> PermuteIndices(PrimExpr row_idx, PrimExpr col_idx, int row_size) {
    // Index after vectorizing by 8
    PrimExpr col_idx_outer = floordiv(col_idx, 8), col_idx_inner = floormod(col_idx, 8);
    PrimExpr new_col_idx_outer;
    if (row_size % 64 == 0) {
      // Use 8 * 8 permuted layout
      // Every number below corresponds to 8 consecutive fp16 number in shared mem, i.e. one read
      // Every row below corresponds to 32 banks
      // 0  1  2  3  4  5  6  7    ==>    0  1  2  3  4  5  6  7
      // 0  1  2  3  4  5  6  7    ==>    1  0  3  2  5  4  7  6
      // 0  1  2  3  4  5  6  7    ==>    2  3  0  1  6  7  4  5
      // 0  1  2  3  4  5  6  7    ==>    3  2  1  0  7  6  5  4
      // 0  1  2  3  4  5  6  7    ==>    4  5  6  7  0  1  2  3
      // 0  1  2  3  4  5  6  7    ==>    5  4  7  6  1  0  3  2
      // 0  1  2  3  4  5  6  7    ==>    6  7  4  5  2  3  0  1
      // 0  1  2  3  4  5  6  7    ==>    7  6  5  4  3  2  1  0
      auto row_idx_sub = floormod(row_idx, 8), col_idx_sub = floormod(col_idx_outer, 8);
      col_idx_sub = col_idx_sub ^ row_idx_sub;
      new_col_idx_outer = floordiv(col_idx_outer, 8) * 8 + col_idx_sub;
    } else {
      ICHECK(row_size % 32 == 0);
      // Use 8 * 4 permuted layout
      // Every number below corresponds to 8 consecutive fp16 number in shared mem, i.e. one read
      // Every row below corresponds to 16 banks
      // 0  1  2  3    ==>    0  1  2  3
      // 0  1  2  3    ==>    0  1  2  3
      // 0  1  2  3    ==>    1  0  3  2
      // 0  1  2  3    ==>    1  0  3  2
      // 0  1  2  3    ==>    2  3  0  1
      // 0  1  2  3    ==>    2  3  0  1
      // 0  1  2  3    ==>    3  2  1  0
      // 0  1  2  3    ==>    3  2  1  0
      // View with 8 elements per row:
      // 0  1  2  3  4  0  1  2  3    ==>    0  1  2  3  0  1  2  3
      // 0  1  2  3  4  0  1  2  3    ==>    1  0  3  2  1  0  3  2
      // 0  1  2  3  4  0  1  2  3    ==>    2  3  0  1  2  3  0  1
      // 0  1  2  3  4  0  1  2  3    ==>    3  2  1  0  3  2  1  0
      auto row_idx_sub = floormod(row_idx, 8), col_idx_sub = floormod(col_idx_outer, 4);
      col_idx_sub = col_idx_sub ^ floordiv(row_idx_sub, 2);
      new_col_idx_outer = floordiv(col_idx_outer, 4) * 4 + col_idx_sub;
    }
    return {row_idx, analyzer_->Simplify(new_col_idx_outer * 8 + col_idx_inner)};
  }

  static void CheckModeValid(String permuted_layout_mode) {
    static const std::vector<String> valid_modes = {"", "g2s_A", "g2s_B", "s2l_A", "s2l_B"};
    auto found = std::find(valid_modes.begin(), valid_modes.end(), permuted_layout_mode) !=
                 valid_modes.end();
    CHECK(found) << "Invalid permuted layout mode \"" << permuted_layout_mode << "\"";
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    if (op->annotations.count("permuted_layout") == 0) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    String permuted_layout_mode = Downcast<String>(op->annotations.at("permuted_layout"));
    if (permuted_layout_mode.empty()) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    CHECK(permuted_layout_mode_ == "" || permuted_layout_mode_ == permuted_layout_mode)
        << "Attribute \"permuted_layout\" is already set by block \"" << scope_block_name_
        << "\", but "
           "different from block \""
        << op->name_hint << "\"";
    CheckModeValid(permuted_layout_mode);

    permuted_layout_mode_ = permuted_layout_mode;
    scope_block_name_ = op->name_hint;

    Block blk = Downcast<Block>(IRMutatorWithAnalyzer::VisitStmt_(op));

    permuted_layout_mode_ = "";
    scope_block_name_ = "";
    return blk;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    // Case 1. Rewrite global to shared.dyn or shared
    // We assume the shape of the shared memory is [..., row_size, col_size],
    // where row_size is divisible by 64, or divisible by 32 and col_size is divisible by 2.
    auto store = Downcast<BufferStore>(IRMutatorWithAnalyzer::VisitStmt_(op));

    if (permuted_layout_mode_ == "" || !support::StartsWith(permuted_layout_mode_, "g2s")) {
      return store;
    }

    auto buffer = store->buffer;
    auto scope = StorageScope::Create(GetPtrStorageScope(buffer->data));
    if (scope.rank != StorageRank::kShared) {
      return store;
    }

    CHECK(buffer->shape.size() >= 2)
        << "The dimension of Buffer \"" << buffer->name << "\" with shape " << buffer->shape
        << " should be at least 2";

    auto dim = buffer->shape.size();
    auto buffer_row_size = buffer->shape[dim - 1].as<IntImmNode>()->value;
    auto buffer_col_size = buffer->shape[dim - 2].as<IntImmNode>()->value;

    if (buffer_row_size % 32 != 0) {
      LOG(FATAL) << "Permuted Layout for Buffer \"" << buffer->name << "\" with shape "
                 << buffer->shape
                 << " is not supported since its second dimension is not divisible by 32";
    } else if (buffer_row_size % 64 == 32 && buffer_col_size % 2 != 0) {
      LOG(FATAL) << "Permuted Layout for Buffer \"" << buffer->name << "\" with shape "
                 << buffer->shape
                 << " is not supported since its first dimension is not divisible by 2 and "
                    "second dimension is not divisible by 64";
    }

    // Store the buffer row size to for later handling of T.ptx_ldmatrix
    if (permuted_layout_mode_.at(4) == 'A') {
      buffer_row_size_A_ = buffer_row_size;
    } else {
      buffer_row_size_B_ = buffer_row_size;
    }

    // Mutate the last two indices
    auto store_node = store.CopyOnWrite();
    PrimExpr row_idx = store_node->indices[dim - 2];
    PrimExpr col_idx = store_node->indices[dim - 1];
    auto new_indices = PermuteIndices(row_idx, col_idx, buffer_row_size);
    store_node->indices.Set(dim - 2, new_indices[0]);
    store_node->indices.Set(dim - 1, new_indices[1]);

    return store;
  }

  PrimExpr VisitExpr_(const CallNode* op) {
    // Case 2. Rewrite shared or shared.dyn to local. We only consider T.ptx_ldmatrix, which
    // has the form T.ptx_ldmatrix(..., smem_ptr, smem_offset) and
    // smem_ptr == T.tvm_access_ptr(ptype, data, offset, extent, rw_mask)
    auto call = Downcast<Call>(IRMutatorWithAnalyzer::VisitExpr_(op));

    if (!call->op.same_as(builtin::ptx_ldmatrix())) {
      return call;
    }
    if (permuted_layout_mode_ == "" || !support::StartsWith(permuted_layout_mode_, "s2l")) {
      return call;
    }

    auto access_ptr_node = call->args[5].as<CallNode>();
    PrimExpr smem_offset = call->args[6];

    CHECK(access_ptr_node && access_ptr_node->op.same_as(builtin::tvm_access_ptr()))
        << "Invalid ptx_ldmatrix for permuted layout: " << GetRef<Call>(op);

    // The 2th arg of T.tvm_access_ptr call is offset, we set it to 0 and accumulate it to
    // smem_offset
    auto access_ptr = GetRef<Call>(access_ptr_node);
    auto new_access_ptr = access_ptr.CopyOnWrite();
    smem_offset += new_access_ptr->args[2];
    new_access_ptr->args.Set(2, 0);

    // Retrieve the previous set row size
    int buffer_row_size;
    if (permuted_layout_mode_.at(4) == 'A') {
      CHECK(buffer_row_size_A_ != -1) << "Buffer row size for A is not set";
      buffer_row_size = buffer_row_size_A_;
      buffer_row_size_A_ = -1;
    } else {
      CHECK(buffer_row_size_B_ != -1) << "Buffer row size for B is not set";
      buffer_row_size = buffer_row_size_B_;
      buffer_row_size_B_ = -1;
    }

    // Convert offset to 2-dimension, reindex it and convert it back
    PrimExpr row_idx = floordiv(smem_offset, buffer_row_size);
    PrimExpr col_idx = floormod(smem_offset, buffer_row_size);

    auto new_indices = PermuteIndices(row_idx, col_idx, buffer_row_size);
    auto new_offset = analyzer_->Simplify(new_indices[0] * buffer_row_size + new_indices[1]);

    auto new_call = call.CopyOnWrite();
    new_call->args.Set(5, access_ptr);
    new_call->args.Set(6, new_offset);

    return call;
  }

  String permuted_layout_mode_ = "";
  String scope_block_name_;
  int buffer_row_size_A_ = -1;
  int buffer_row_size_B_ = -1;
};

namespace transform {

Pass InjectPermutedLayout() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return PermutedLayoutInjector::Transfrom(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectPermutedLayout", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InjectPermutedLayout").set_body_typed(InjectPermutedLayout);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
