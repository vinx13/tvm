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
 * \brief Lower the usage of TF32 in CUDA tensor core.

   This pass will add conversions from float to
 * tfloat32 after wmma::load_matrix_sync.
 * \file lower_tf32_tensor_core.cc
 */

#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../runtime/thread_storage_scope.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

class TF32TensorCoreLowerer : public StmtMutator {
 private:
  PrimExpr GetWMMAFragmentSize(const runtime::StorageScope& scope, const Var& fragment,
                               const PrimExpr& m, const PrimExpr& n, const PrimExpr& k) {
    if (scope.rank == runtime::StorageRank::kWMMAMatrixA) {
      return m * k;
    } else if (scope.rank == runtime::StorageRank::kWMMAMatrixB) {
      return k * n;
    } else if (scope.rank == runtime::StorageRank::kWMMAAccumulator) {
      return m * n;
    } else {
      LOG(FATAL) << "Unsupported WMMA fragment: " << fragment;
      return 0;
    }
  }

  Stmt EmitWMMAFragmentFloatToTF32Cast(const runtime::StorageScope& scope, Var& fragment,
                                       const PrimExpr& m, const PrimExpr& n, const PrimExpr& k,
                                       const PrimExpr& fragment_index) {
    PrimExpr wmma_size = GetWMMAFragmentSize(scope, fragment, m, n, k);
    Var element_index = Var("e");
    PrimExpr fp32_element = Call(DataType::Float(32), builtin::tvm_wmma_get_element(),
                                 {fragment, fragment_index, element_index});
    PrimExpr tf32_element = Call(
        DataType::Float(32), builtin::call_pure_extern(),
        {StringImm("__float_to_tf32"), fp32_element});  // the storage format of tf32 is still fp32
    Stmt set_element = Evaluate(Call(DataType::Handle(), builtin::tvm_wmma_set_element(),
                                     {fragment, fragment_index, element_index, tf32_element}));
    return For(element_index, Integer(0), wmma_size, ForKind::kSerial, set_element);
  }

  Stmt VisitStmt_(const EvaluateNode* op) final {
    static const Op& load_matrix_sync = builtin::tvm_load_matrix_sync();

    Evaluate evaluate = Downcast<Evaluate>(StmtMutator::VisitStmt_(op));
    if (const CallNode* call = evaluate->value.as<CallNode>();
        call != nullptr && call->op.same_as(load_matrix_sync)) {
      Var fragment = Downcast<Var>(call->args[0]);
      runtime::StorageScope scope = runtime::StorageScope::Create(GetPtrStorageScope(fragment));
      if ((scope.rank == runtime::StorageRank::kWMMAMatrixA ||
           scope.rank == runtime::StorageRank::kWMMAMatrixB) &&
          scope.tag == ".tf32") {
        PrimExpr m = call->args[1];
        PrimExpr n = call->args[2];
        PrimExpr k = call->args[3];
        PrimExpr fragment_index = call->args[4];
        Stmt stmt_after = EmitWMMAFragmentFloatToTF32Cast(scope, fragment, m, n, k, fragment_index);
        return SeqStmt({std::move(evaluate), stmt_after});
      }
    }
    return std::move(evaluate);
  }
};

PrimFunc LowerTF32TensorCore(PrimFunc f) {
  auto fptr = f.CopyOnWrite();
  fptr->body = TF32TensorCoreLowerer()(std::move(fptr->body));
  return f;
}

namespace transform {

Pass LowerTF32TensorCore() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LowerTF32TensorCore(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerTF32TensorCore", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerTF32TensorCore").set_body_typed(LowerTF32TensorCore);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
