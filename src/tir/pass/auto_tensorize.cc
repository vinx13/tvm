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

/*!
 * \file auto_tensorize.cc
 */
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/te/tensor_intrin.h>
#include <tvm/te/operation.h>

namespace tvm {
namespace tir {

class AutoTensorizeMutator : public StmtExprMutator {
 public:
  explicit AutoTensorizeMutator(const Array<te::TensorIntrin> &tensor_intrins) : tensor_intrins_(tensor_intrins) {
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (tir::attr::IsPragmaKey(op->attr_key) && op->attr_key == "pragma_tensorize_hint") {
      const Stmt &body = op->body;
      LOG(INFO) << body;
      for (const auto &intrin : tensor_intrins_) {
        if (MatchTensorIntrin(body, intrin)) {
          return Tensorize(body, intrin);
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  private:
   Array<te::TensorIntrin> tensor_intrins_;

  bool MatchTensorIntrin(const Stmt& stmt, const te::TensorIntrin &intrin) {
    return true;  // FIXME
  }

  Stmt Tensorize()
};

Stmt AutoTensorize(Stmt stmt, te::Schedule schedule, Array<te::TensorIntrin> tensor_intrins) {
  auto op = tensor_intrins[0]->op.as<te::ComputeOpNode>();
  LOG(INFO) << op->body;
  LOG(INFO) << stmt;
  return AutoTensorizeMutator(tensor_intrins)(stmt);
}

}  // namespace tir
}  // namespace tvm
