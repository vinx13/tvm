/*!
 * Copyright (c) 2019 by Contributors
 *
 * \file eliminate_common_subexpr.cc
 * \brief Combine common subexpressions.
 *
 * This is an optimization pass that eliminates common subexpressions. During the pass, it tries
 * to replace an expression with a previously appeared expression with the same input and
 * attributes. The fskip callback argument allows us to skip specific expressions.
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include <unordered_map>
#include "./pattern_util.h"

namespace tvm {
namespace relay {

class CommonSubexprEliminator : public ExprMutator {
 public:
  explicit CommonSubexprEliminator(runtime::TypedPackedFunc<bool(Expr)> fskip): fskip_(fskip) {}

  Expr VisitExpr_(const CallNode* call) final {
    static auto op_stateful = Op::GetAttr<TOpIsStateful>("TOpIsStateful");
    Expr new_expr = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_expr.as<CallNode>();
    CHECK(new_call);
    const OpNode* op = new_call->op.as<OpNode>();
    AttrsEqual attrs_equal;

    if (new_call->args.size() == 0 || op == nullptr || op_stateful.get(GetRef<Op>(op), false)) {
      return new_expr;
    }
    if (fskip_ != nullptr && fskip_(new_expr)) {
      return new_expr;
    }

    auto it = expr_map_.find(new_call->args[0]);
    if (it != expr_map_.end()) {
      for (const CallNode* candidate : it->second) {
        bool is_equivalent = true;
        if (!new_call->op.same_as(candidate->op)) continue;
        for (size_t i = 0; i < new_call->args.size(); i++) {
          if (!new_call->args[i].same_as(candidate->args[i]) &&
              !IsEqualScalar(new_call->args[i], candidate->args[i]) &&
              !attrs_equal(new_call->attrs, candidate->attrs)) {
            is_equivalent = false;
            break;
          }
        }
        if (!is_equivalent) continue;
        return GetRef<Call>(candidate);
      }
    }
    expr_map_[new_call->args[0]].push_back(new_call);
    return new_expr;
  }

  std::unordered_map<Expr, std::vector<const CallNode*>, NodeHash, NodeEqual> expr_map_;
  runtime::TypedPackedFunc<bool(Expr)> fskip_;
};

Expr EliminateCommonSubexpr(const Expr& expr, PackedFunc callback) {
  return CommonSubexprEliminator(callback)(expr);
}

TVM_REGISTER_API("relay._ir_pass.eliminate_common_subexpr")
.set_body_typed<Expr(Expr, PackedFunc)>(EliminateCommonSubexpr);

}  // namespace relay
}  // namespace tvm
