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
 * \file apply_block_bound_predicate.cc
 * \brief Apply the block iter bound predicate to loops.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../arith/pattern_match.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

class BoundPredicateParserSimplifier : public ExprMutator {
 public:
  explicit BoundPredicateParserSimplifier(Map<Var, PrimExpr> binding_map,
                                          Map<Var, arith::IntSet>* bound_intset)
      : binding_map_(std::move(binding_map)), bound_intset_(bound_intset) {}

 private:
  PrimExpr VisitExpr(const PrimExpr& expr) final {
    if (expr->IsInstance<AndNode>() || expr->IsInstance<LTNode>() || expr->IsInstance<GENode>()) {
      return ExprMutator::VisitExpr(expr);
    }
    ICHECK(false) << "InternalError: PrimExpr \"" << expr
                  << "\" is not supposed to appear as a bound predicate";
    throw;
  }

  PrimExpr VisitExpr_(const LTNode* lt) final {
    const VarNode* var = lt->a.as<VarNode>();
    if (!var) {
      ICHECK(false) << "InternalError: LHS of logical expression here is required to be variables";
    }
    Optional<PrimExpr> binding = binding_map_.Get(GetRef<Var>(var));
    if (!binding.defined()) {
      ICHECK(false) << "InternalError: The LHS variable is supposed to be a block iterator";
    }
    const VarNode* loop_var = binding.value().as<VarNode>();
    if (!loop_var) {
      return GetRef<PrimExpr>(lt);
    }

    arith::IntSet intset =
        bound_intset_->Get(GetRef<Var>(loop_var)).value_or(arith::IntSet::Everything());
    intset = arith::Intersect(
        {intset, arith::IntSet::FromRange(Range(min_value(lt->b.dtype()), lt->b))});
    bound_intset_->Set(GetRef<Var>(loop_var), intset);
    return const_true();
  }

  PrimExpr VisitExpr_(const GENode* ge) final {
    const VarNode* var = ge->a.as<VarNode>();
    if (!var) {
      ICHECK(false) << "InternalError: LHS of logical expression here is required to be variables";
    }
    Optional<PrimExpr> binding = binding_map_.Get(GetRef<Var>(var));
    if (!binding.defined()) {
      ICHECK(false) << "InternalError: The LHS variable is supposed to be a block iterator";
    }
    const VarNode* loop_var = binding.value().as<VarNode>();
    if (!loop_var) {
      return GetRef<PrimExpr>(ge);
    }

    arith::IntSet intset =
        bound_intset_->Get(GetRef<Var>(loop_var)).value_or(arith::IntSet::Everything());
    intset = arith::Intersect(
        {intset, arith::IntSet::FromRange(Range(ge->b, max_value(ge->b.dtype())))});
    bound_intset_->Set(GetRef<Var>(loop_var), intset);
    return const_true();
  }

  Map<Var, PrimExpr> binding_map_;
  Map<Var, arith::IntSet>* bound_intset_;
};

/*!
 * \brief Narrow the extents of some loops by checking whether some constraints in the block iter
 * bound predicates can be directly applied on the loops.
 */
class LoopExtentMutator : public StmtMutator {
 private:
  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    // Step 1. Mutate recursively.
    BlockRealize new_realize = Downcast<BlockRealize>(StmtMutator::VisitStmt_(realize));
    // Step 2. If the block has no "require_block_var_bound_predicate" annotation, skip this block.
    Block block = new_realize->block;
    const Optional<ObjectRef>& bound_predicate =
        block->annotations.Get(tir::attr::require_block_var_bound_predicate);
    if (!bound_predicate.defined()) {
      return new_realize;
    }
    // Step 3. Make a mapping from block iters to bindings.
    Map<Var, PrimExpr> binding_map;
    ICHECK_EQ(block->iter_vars.size(), new_realize->iter_values.size());
    int n_iter = static_cast<int>(block->iter_vars.size());
    for (int i = 0; i < n_iter; ++i) {
      binding_map.Set(block->iter_vars[i]->var, new_realize->iter_values[i]);
    }
    // Step 4. Parse the bound predicate, removing constraints on the block vars whose binding are
    // single vars.
    PrimExpr new_predicate = BoundPredicateParserSimplifier(
        binding_map, &bound_intset_)(Downcast<PrimExpr>(bound_predicate.value()));
    // Step 5. Update the block annotation and update the new block-realize.
    ObjectPtr<BlockNode> p_new_block = CopyOnWrite(block.get());
    if (ana_.CanProveEqual(new_predicate, const_true())) {
      p_new_block->annotations.erase(tir::attr::require_block_var_bound_predicate);
    } else {
      p_new_block->annotations.Set(tir::attr::require_block_var_bound_predicate, new_predicate);
    }
    ObjectPtr<BlockRealizeNode> p_new_realize = CopyOnWrite(new_realize.get());
    p_new_realize->block = Block(p_new_block);

    return BlockRealize(p_new_realize);
  }

  Stmt VisitStmt_(const ForNode* loop) final {
    // Step 1. Mutate recursively.
    For new_loop = Downcast<For>(StmtMutator::VisitStmt_(loop));
    // Step 2. Check whether this loop has a bound intset. If not, return the new loop.
    Optional<arith::IntSet> intset = bound_intset_.Get(new_loop->loop_var);
    if (!intset.defined()) {
      return new_loop;
    }
    // Step 3. Update the new loop's `min` and `extent` according to the extent.
    PrimExpr new_min = max(new_loop->min, intset.value().min());
    PrimExpr new_extent = min(new_loop->min + new_loop->extent, intset.value().max() + 1) - new_min;
    // Step 4. Update the new loop.
    ObjectPtr<ForNode> p_new_loop = CopyOnWrite(new_loop.get());
    p_new_loop->min = ana_.Simplify(new_min);
    p_new_loop->extent = ana_.Simplify(new_extent);

    return For(p_new_loop);
  }

  /*! \brief The bounds of loop vars, provided by the block iter bound predicate */
  Map<Var, arith::IntSet> bound_intset_;
  /*! \brief The analyzer */
  arith::Analyzer ana_;
};

PrimFunc ApplyBlockBoundPredicate(PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = LoopExtentMutator()(f->body);
    return f;
  } else {
    return f;
  }
}

namespace transform {

Pass ApplyBlockBoundPredicate() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return ApplyBlockBoundPredicate(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.ApplyBlockBoundPredicate", {});
}

TVM_REGISTER_GLOBAL("tir.transform.ApplyBlockBoundPredicate")
    .set_body_typed(ApplyBlockBoundPredicate);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
