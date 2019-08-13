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
 * Copyright (c) 2018 by Contributors
 * \file remove_add_zeros.cc
 * \brief Remove add zeros
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/transform.h>
#include "pattern_util.h"

namespace tvm {
namespace relay {

class AddZerosRemover : public ExprMutator {
 public:
  bool ShapeMatch(const TensorTypeNode* lhs, const TensorTypeNode* rhs) {
    if (lhs->shape.size() != rhs->shape.size()) {
      return false;
    }
    for (size_t i = 0; i < lhs->shape.size(); i++) {
      const auto* x = as_const_int(lhs->shape[i]);
      const auto* y = as_const_int(rhs->shape[i]);
      if (!(x && y && *x == *y)) {
        return false;
      }
    }
    return true;
  }

  Expr VisitExpr_(const CallNode* n) {
    static const Op& add = Op::Get("add");
    static const Op& zeros = Op::Get("zeros");
    auto new_n = ExprMutator::VisitExpr_(n);
    auto new_call = new_n.as<CallNode>();
    CHECK(new_call);
    if (new_call->op.same_as(add)) {
      const auto* lhs_type = n->args[0]->type_as<TensorTypeNode>();
      const auto* rhs_type = n->args[1]->type_as<TensorTypeNode>();
      const auto* lhs = n->args[0].as<CallNode>();
      const auto* rhs = n->args[1].as<CallNode>();
      if (lhs && lhs->op.same_as(zeros) && ShapeMatch(lhs_type, rhs_type)) {
        return new_call->args[1];
      } else if (rhs && rhs->op.same_as(zeros) && ShapeMatch(lhs_type, rhs_type)) {
        return new_call->args[0];
      }
    }
    return new_n;
  }
};

Expr RemoveAddZeros(const Expr& e) {
  return AddZerosRemover().Mutate(e);
}

namespace transform {

Pass RemoveAddZeros() {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
    [=](Function f, Module m, PassContext pc) {
    return Downcast<Function>(RemoveAddZeros(f));
  };
  return CreateFunctionPass(pass_func, 3, "RemoveAddZeros",
                            {ir::StringImm::make("InferType")});
}

TVM_REGISTER_API("relay._transform.RemoveAddZeros")
.set_body_typed(CanonicalizeOps);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
