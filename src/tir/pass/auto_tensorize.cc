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
#include "ir_util.h"

namespace tvm {
namespace tir {

class TensorIntrinMatcher : public StmtExprVisitor {
 public:
  TensorIntrinMatcher(const te::TensorIntrin &tensor_intrin) : tensor_intrin_(tensor_intrin), match_(true) {
  }

  bool Match(const Stmt &stmt) {
    VisitStmt(stmt);
    LOG(INFO) << "Match? " << match_;
    return match_;
  }

 protected:
  void VisitStmt(const Stmt & stmt) final {
    if (!match_) return;
    StmtExprVisitor::VisitStmt(stmt);
  }

  void VisitStmt_(const IfThenElseNode* op) final {
    match_ = false;
  }

 private:
  te::TensorIntrin tensor_intrin_;
  bool match_;
};

class TensorIntrinRewritter : public StmtExprMutator {
 public:
  explicit TensorIntrinRewritter(const te::TensorIntrin &intrin, const std::unordered_map<TensorKey, Region>& bounds) : intrin_(intrin), bounds_(bounds) {

  }

 protected:
  Stmt VisitStmt(const Stmt &stmt) final {
    auto result = StmtExprMutator::VisitStmt(stmt);
    if (result.same_as(stmt)) {
      return result;
    }
    return result;
  }

  Stmt VisitStmt_(const ProvideNode *op) final {
      // find C = C + A*B
      auto add = op->value.as<AddNode>();
      if (!add) {
        return StmtExprMutator::VisitStmt_(op);
      }
      auto mul = add->b.as<MulNode>();
      if (!mul) {
        return StmtExprMutator::VisitStmt_(op);
      }
      auto lhs = mul->a.as<CallNode>();
      auto rhs = mul->b.as<CallNode>();
      if (!lhs || !rhs) {
        return StmtExprMutator::VisitStmt_(op);
      }

      // auto shape_a = GetShape(lhs);
      // auto shape_b = GetShape(rhs);
      // LOG(INFO) << shape_a;
      // LOG(INFO) << shape_b;

      Stmt nop = EvaluateNode::make(0);
      auto gen_bind = [&](const CallNode *src, const Buffer &dst_buf) {
        auto src_tensor = Downcast<te::Operation>(src->func).output(0);
        Array<ObjectRef> bind_spec{dst_buf, src_tensor};
        auto shape = GetShape(src);
        LOG(INFO) << "Shape " << shape;
        Array<PrimExpr> tuple;

        for (size_t i = 0; i < src->args.size(); i++) {
          tuple.push_back(src->args[i]);
          tuple.push_back(shape[i]);
        }

        LOG(INFO) << "Tuple " << tuple;
        return AttrStmtNode::make(
            bind_spec, tir::attr::buffer_bind_scope,
            CallNode::make(DataType::Handle(), tir::intrinsic::tvm_tuple, tuple,
                           CallNode::Intrinsic),
            nop);
      };

      std::vector<Stmt> input_bind_nest, output_bind_nest;
      input_bind_nest.push_back(gen_bind(lhs, intrin_->buffers[0]));
      input_bind_nest.push_back(gen_bind(rhs, intrin_->buffers[1]));

      return MergeNest(input_bind_nest, intrin_->body);

      // ObjectPtr<te::TensorNode> tensor_node = make_object<te::TensorNode>();
      // tensor_node->value_index = lhs->value_index;
      // tensor_node->op = Downcast<te::Operation>(lhs->func);
      // tensor_node->shape = shape;
      // tensor_node->dtype = datatype;
      // Tensor tensor(tensor_node);

      // const CallNode *arg1 = op->args[0].as<CallNode>();
      // const CallNode *arg2 = op->args[1].as<CallNode>();

      // CHECK(arg1);
      // CHECK(arg2);
      // LOG(INFO) << "FUNC" << arg1->func;
      //    return StmtExprMutator::VisitStmt_(op);
  }

 private:

 Array<PrimExpr> GetShape(const CallNode *op) {
    Region bound = bounds_.at(TensorKey{op->func, op->value_index});
    Array<PrimExpr> shape;
    for (const auto &r : bound) {
      shape.push_back(r->extent);
    }
    return shape;
  }

   te::TensorIntrin intrin_;
   std::unordered_map<TensorKey, Region> bounds_;
};

class AutoTensorizeMutator : public StmtExprMutator {
 public:
  explicit AutoTensorizeMutator(const Array<te::TensorIntrin> &tensor_intrins)
      : tensor_intrins_(tensor_intrins) {}

 protected:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (tir::attr::IsPragmaKey(op->attr_key) &&
        op->attr_key == "pragma_tensorize_hint") {
      const Stmt &body = op->body;
      LOG(INFO) << body;
      for (const auto &intrin : tensor_intrins_) {
        if (MatchTensorIntrin(body, intrin)) {
          // return Tensorize(body, intrin);
          return TensorIntrinRewritter(intrin, bounds_)(body);
          // tensorize_scope_ = true;
          // auto result = StmtExprMutator::VisitStmt_(op);
          // auto result_attr_stmt = result.as<AttrStmtNode>();
          // if (result_attr_stmt && op->body.same_as(result_attr_stmt->body)) {
          //   return result_attr_stmt->body;
          // }
          // tensorize_scope_ = false;
          // return result;
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }


  Stmt VisitStmt_(const RealizeNode *op) final {
    TensorKey key{op->func, op->value_index};
    bounds_[key] = op->bounds;
    return StmtExprMutator::VisitStmt_(op);
  }

 private:




  Array<te::TensorIntrin> tensor_intrins_;
  bool tensorize_scope_ = false;
  std::unordered_map<TensorKey, Region> bounds_;

  bool MatchTensorIntrin(const Stmt &stmt, const te::TensorIntrin &intrin) {
    return TensorIntrinMatcher(intrin).Match(stmt);
  }

  Stmt Tensorize(const Stmt &stmt, const te::TensorIntrin &intrin) {
    // bind buffers
    Stmt nop = EvaluateNode::make(0);
    std::vector<Stmt> input_bind_nest, output_bind_nest;
    for (size_t i = 0; i < intrin->inputs.size(); ++i) {
      te::Tensor tensor = intrin->inputs[i];
      Buffer buffer = intrin->buffers[i];
      Array<ObjectRef> bind_spec{buffer, tensor};
      // auto it = in_region.find(tensor);
      // const Array<Range> &region = it->second;
      Array<PrimExpr> tuple;
      // for (const Range r : region) {
      //   tuple.push_back(r->min);
      //   tuple.push_back(r->extent);
      // }
      input_bind_nest.emplace_back(AttrStmtNode::make(
          bind_spec, tir::attr::buffer_bind_scope,
          CallNode::make(DataType::Handle(), tir::intrinsic::tvm_tuple, tuple,
                         CallNode::Intrinsic),
          nop));
    }

    // for (size_t i = intrin->inputs.size(); i < intrin->buffers.size(); ++i)
    // {
    //   te::Tensor tensor = stage->op.output(i - intrin->inputs.size());
    //   Buffer buffer = intrin->buffers[i];
    //   Array<ObjectRef> bind_spec{buffer, tensor};
    //   output_bind_nest.emplace_back(AttrStmtNode::make(
    //       bind_spec, tir::attr::buffer_bind_scope,
    //       CallNode::make(DataType::Handle(), tir::intrinsic::tvm_tuple,
    //       tuple,
    //                      CallNode::Intrinsic),
    //       nop));
    // }
    auto result = MergeNest(input_bind_nest, intrin->body);
    LOG(INFO) << "Result\n" << result;
    return result;
    // return tensor_intrin->body;
  }
  std::vector<Stmt> loop_nests_;
};

Stmt AutoTensorize(Stmt stmt, te::Schedule schedule,
                   Array<te::TensorIntrin> tensor_intrins) {
  auto op = tensor_intrins[0]->op.as<te::ComputeOpNode>();
  LOG(INFO) << op->body;
  LOG(INFO) << stmt;

  return AutoTensorizeMutator(tensor_intrins)(stmt);
}

}  // namespace tir
}  // namespace tvm
