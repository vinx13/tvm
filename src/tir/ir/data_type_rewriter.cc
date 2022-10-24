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
 * \file data_type_rewriter.cc
 * \brief Rewrite the data type of expressions.
 */

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "./functor_common.h"

namespace tvm {
namespace tir {

Stmt DataTypeLegalizer::VisitStmt_(const ForNode* op) {
  Stmt s = StmtExprMutator::VisitStmt_(op);
  op = s.as<ForNode>();
  ICHECK(op != nullptr) << "Expected type to be ForNode, but get " << s->GetTypeKey();
  PrimExpr e = VisitExpr(op->loop_var);
  Var var = Downcast<Var>(e);
  return For(var, cast(var.dtype(), op->min), cast(var.dtype(), op->extent), op->kind, op->body,
             op->thread_binding, op->annotations);
}

Stmt DataTypeLegalizer::VisitStmt_(const BlockRealizeNode* op) {
  BlockRealize realize = Downcast<BlockRealize>(StmtExprMutator::VisitStmt_(op));
  Array<PrimExpr> new_iter_values;
  bool changed = false;
  for (int i = 0; i < static_cast<int>(op->iter_values.size()); ++i) {
    auto dtype = realize->block->iter_vars[i]->var->dtype;
    if (op->iter_values[i]->dtype != dtype) {
      new_iter_values.push_back(cast(dtype, realize->iter_values[i]));
      changed = true;
    } else {
      new_iter_values.push_back(realize->iter_values[i]);
    }
  }
  if (changed) {
    realize.CopyOnWrite()->iter_values = std::move(new_iter_values);
  }
  return std::move(realize);
}

Stmt DataTypeLegalizer::VisitStmt_(const BlockNode* op) {
  Block new_block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
  Array<IterVar> new_iter_vars = MutateArray(new_block->iter_vars, [this](const IterVar& iter) {
    auto dtype = iter->var.dtype();
    if (iter->dom->min->dtype != dtype || iter->dom->extent->dtype != dtype) {
      IterVar new_iter = iter;
      new_iter.CopyOnWrite()->dom =
          Range(cast(dtype, iter->dom->min), cast(dtype, iter->dom->extent));
      return new_iter;
    } else {
      return iter;
    }
  });
  if (!op->iter_vars.same_as(new_iter_vars)) {
    new_block.CopyOnWrite()->iter_vars = std::move(new_iter_vars);
  }
  return std::move(new_block);
}

Stmt DataTypeLegalizer::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == attr::thread_extent || op->attr_key == attr::virtual_thread) {
    Stmt s = StmtExprMutator::VisitStmt_(op);
    op = s.as<AttrStmtNode>();
    ICHECK(op != nullptr) << "Expected type to be AttrStmtNode"
                          << ", but get " << s->GetTypeKey();
    const IterVarNode* iv = op->node.as<IterVarNode>();
    ICHECK(iv != nullptr) << "Expected type to be IterVarNode"
                          << ", but get " << op->node->GetTypeKey();
    PrimExpr e = VisitExpr(iv->var);
    Var var = Downcast<Var>(e);
    if (ivmap_.find(iv) == ivmap_.end()) {
      Range dom = iv->dom;
      if (dom.defined()) {
        PrimExpr extend = dom->extent;
        ICHECK(extend.dtype().is_int() && var.dtype().is_int());
        if (var.dtype().bits() != extend.dtype().bits()) {
          DataType dtype = var.dtype();
          dom = Range(cast(dtype, dom->min), cast(dtype, extend), dom->span);
        }
      }
      ivmap_[iv] = IterVar(dom, var, iv->iter_type, iv->thread_tag);
    }
    return AttrStmt(ivmap_[iv], op->attr_key, cast(var.dtype(), op->value), op->body);
  }
  return StmtExprMutator::VisitStmt_(op);
}

PrimExpr DataTypeLegalizer::VisitExpr_(const SelectNode* op) {
  PrimExpr condition = this->VisitExpr(op->condition);
  PrimExpr true_value = this->VisitExpr(op->true_value);
  PrimExpr false_value = this->VisitExpr(op->false_value);
  if (condition.same_as(op->condition) && true_value.same_as(op->true_value) &&
      false_value.same_as(op->false_value) && true_value.dtype() == false_value.dtype()) {
    return GetRef<PrimExpr>(op);
  } else {
    int bits = std::max(true_value.dtype().bits(), false_value.dtype().bits());
    DataType dtype = true_value.dtype().with_bits(bits);
    if (true_value.dtype() != dtype) true_value = cast(dtype, true_value);
    if (false_value.dtype() != dtype) false_value = cast(dtype, false_value);
    return Select(condition, true_value, false_value);
  }
}

PrimExpr DataTypeLegalizer::VisitExpr_(const RampNode* op) {
  PrimExpr base = VisitExpr(op->base);
  PrimExpr stride = VisitExpr(op->stride);
  if (base.same_as(op->base) && stride.same_as(op->stride) && base.dtype() == stride.dtype()) {
    return GetRef<PrimExpr>(op);
  } else {
    ICHECK(base.dtype().is_int() && stride.dtype().is_int());
    int bits = std::max(base.dtype().bits(), stride.dtype().bits());
    DataType dtype = base.dtype().with_bits(bits);
    if (base.dtype() != dtype) base = cast(dtype, base);
    if (stride.dtype() != dtype) stride = cast(dtype, stride);
    return Ramp(base, stride, op->lanes);
  }
}

#define DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(OP, FUNC)                 \
  PrimExpr DataTypeLegalizer::VisitExpr_(const OP* op) {                  \
    PrimExpr a = this->VisitExpr(op->a);                                  \
    PrimExpr b = this->VisitExpr(op->b);                                  \
    if (op->a.same_as(a) && op->b.same_as(b) && a.dtype() == b.dtype()) { \
      return GetRef<PrimExpr>(op);                                        \
    } else {                                                              \
      return FUNC(a, b);                                                  \
    }                                                                     \
  }

DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(AddNode, operator+);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(SubNode, operator-);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MulNode, operator*);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(DivNode, div);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(ModNode, truncmod);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(FloorDivNode, floordiv);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(FloorModNode, floormod);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MinNode, min);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MaxNode, max);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(EQNode, operator==);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(NENode, operator!=);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(LENode, operator<=);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(LTNode, operator<);  // NOLINT(*)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(GTNode, operator>);  // NOLINT(*)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(GENode, operator>=);

#undef DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH

PrimExpr DataTypeLegalizer::VisitExpr_(const CallNode* op) {
  PrimExpr e = StmtExprMutator::VisitExpr_(op);
  op = e.as<CallNode>();
  static const Op& builtin_pow_ = Op::Get("tir.pow");
  ICHECK(op != nullptr) << "Expected type to be CallNode"
                        << ", but get " << e->GetTypeKey();
  if (op->op.same_as(builtin::shift_right())) {
    return op->args[0] >> op->args[1];
  } else if (op->op.same_as(builtin::shift_left())) {
    return op->args[0] << op->args[1];
  } else if (op->op.same_as(builtin::bitwise_and())) {
    return op->args[0] & op->args[1];
  } else if (op->op.same_as(builtin::bitwise_or())) {
    return op->args[0] | op->args[1];
  } else if (op->op.same_as(builtin::bitwise_xor())) {
    return op->args[0] ^ op->args[1];
  } else if (op->op.same_as(builtin_pow_)) {
    return pow(op->args[0], op->args[1]);
  } else if (op->op.same_as(builtin::if_then_else())) {
    return if_then_else(op->args[0], op->args[1], op->args[2]);
  }
  return e;
}

Stmt IndexDataTypeRewriter::VisitStmt_(const AllocateNode* op) {
  bool is_index = is_index_;
  is_index_ = true;
  auto new_extents = op->extents.Map([this](const PrimExpr& e) { return this->VisitExpr(e); });
  auto new_cond = VisitExpr(op->condition);
  is_index_ = is_index;
  auto new_body = this->VisitStmt(op->body);
  if (!new_extents.same_as(op->extents) || !new_cond.same_as(op->condition) ||
      !new_body.same_as(op->body)) {
    Allocate new_allocate = GetRef<Allocate>(op);
    auto* n = new_allocate.CopyOnWrite();
    n->extents = std::move(new_extents);
    n->condition = std::move(new_cond);
    n->body = std::move(new_body);
    return std::move(new_allocate);
  } else {
    return GetRef<Stmt>(op);
  }
}

Stmt IndexDataTypeRewriter::VisitStmt_(const DeclBufferNode* op) {
  Buffer new_buffer = VisitBuffer(op->buffer);
  DeclBuffer decl_buffer = Downcast<DeclBuffer>(StmtExprMutator::VisitStmt_(op));
  if (!new_buffer.same_as(op->buffer)) {
    decl_buffer.CopyOnWrite()->buffer = new_buffer;
  }
  return std::move(decl_buffer);
}

Stmt IndexDataTypeRewriter::VisitStmt_(const BlockNode* op) {
  Array<Buffer> new_alloc_buffers =
      op->alloc_buffers.Map([this](const Buffer& buffer) { return this->VisitBuffer(buffer); });
  Array<MatchBufferRegion> new_match_buffers =
      op->match_buffers.Map([this](const MatchBufferRegion& match_buffer_region) {
        Buffer new_buffer = this->VisitBuffer(match_buffer_region->buffer);
        BufferRegion new_buffer_region = this->VisitBufferRegion(match_buffer_region->source);
        if (!new_buffer.same_as(match_buffer_region->buffer) ||
            !new_buffer_region.same_as(match_buffer_region->source)) {
          return MatchBufferRegion(new_buffer, new_buffer_region);
        } else {
          return match_buffer_region;
        }
      });
  Array<BufferRegion> new_reads = op->reads.Map(
      [this](const BufferRegion& buffer_region) { return this->VisitBufferRegion(buffer_region); });
  Array<BufferRegion> new_writes = op->writes.Map(
      [this](const BufferRegion& buffer_region) { return this->VisitBufferRegion(buffer_region); });
  Array<IterVar> new_iter_vars =
      op->iter_vars.Map([this](const IterVar& iter_var) { return this->VisitIterVar(iter_var); });

  Block new_block = Downcast<Block>(Parent::VisitStmt_(op));
  if (!new_block.same_as(GetRef<Block>(op)) || !new_alloc_buffers.same_as(op->alloc_buffers) ||
      !new_match_buffers.same_as(op->match_buffers) || !new_reads.same_as(op->reads) ||
      !new_writes.same_as(op->writes) | new_iter_vars.same_as(op->iter_vars)) {
    BlockNode* n = new_block.CopyOnWrite();
    n->alloc_buffers = std::move(new_alloc_buffers);
    n->match_buffers = std::move(new_match_buffers);
    n->reads = std::move(new_reads);
    n->writes = std::move(new_writes);
    n->iter_vars = std::move(new_iter_vars);
  }
  return std::move(new_block);
}

Buffer IndexDataTypeRewriter::GetRemappedBuffer(const Buffer& buffer) {
  if (auto it = buffer_remap_.find(buffer); it != buffer_remap_.end()) {
    return (*it).second;
  }
  return buffer;
}

IterVar IndexDataTypeRewriter::VisitIterVar(const IterVar& iter_var) {
  bool is_index = is_index_;
  is_index_ = true;
  Var new_var = Downcast<Var>(VisitExpr(iter_var->var));
  PrimExpr min = VisitExpr(iter_var->dom->min);
  PrimExpr extent = VisitExpr(iter_var->dom->extent);
  is_index_ = is_index;
  if (!new_var.same_as(iter_var->var) || !min.same_as(iter_var->dom->min) ||
      !extent.same_as(iter_var->dom->extent)) {
    IterVar new_iter_var = iter_var;
    IterVarNode* n = new_iter_var.CopyOnWrite();
    n->var = std::move(new_var);
    n->dom = Range(min, extent);
    return new_iter_var;
  }
  return iter_var;
}

Buffer IndexDataTypeRewriter::VisitBuffer(const Buffer& buffer) {
  bool is_index = is_index_;

  is_index_ = true;
  Array<PrimExpr> new_shape =
      buffer->shape.Map([&](const PrimExpr& e) { return this->VisitExpr(e); });
  Array<PrimExpr> new_strides =
      buffer->strides.Map([&](const PrimExpr& e) { return this->VisitExpr(e); });
  auto new_elem_offset = VisitExpr(buffer->elem_offset);
  is_index_ = is_index;

  if (!buffer->shape.same_as(new_shape) || !buffer->strides.same_as(new_strides) || !buffer->elem_offset.same_as(new_elem_offset)) {
    Buffer new_buffer = buffer;
    BufferNode* new_buffer_node = new_buffer.CopyOnWrite();
    new_buffer_node->shape = std::move(new_shape);
    new_buffer_node->strides = std::move(new_strides);
    new_buffer_node->elem_offset = std::move(new_elem_offset);
    buffer_remap_.Set(buffer, new_buffer);
    return new_buffer;
  } else {
    return buffer;
  }
}

BufferRegion IndexDataTypeRewriter::VisitBufferRegion(const BufferRegion& buffer_region) {
  Buffer remapped_buffer = GetRemappedBuffer(buffer_region->buffer);

  bool is_index = is_index_;
  is_index_ = true;
  auto new_region = buffer_region->region.Map([&](const Range& range) {
    return Range::FromMinExtent(this->VisitExpr(range->min), this->VisitExpr(range->extent));
  });
  is_index_ = is_index;

  if (!remapped_buffer.same_as(buffer_region->buffer) ||
      !new_region.same_as(buffer_region->region)) {
    return BufferRegion(remapped_buffer, new_region);
  } else {
    return buffer_region;
  }
}

Stmt IndexDataTypeRewriter::VisitStmt_(const BufferStoreNode* op) {
  BufferStore store = GetRef<BufferStore>(op);

  Buffer new_buffer = GetRemappedBuffer(op->buffer);
  auto value = this->VisitExpr(op->value);
  auto indices = VisitIndices(op->indices);

  if (!new_buffer.same_as(op->buffer) || !value.same_as(op->value) ||
      !indices.same_as(op->indices)) {
    auto writer = store.CopyOnWrite();
    writer->buffer = new_buffer;
    writer->value = value;
    writer->indices = indices;
  }

  return std::move(store);
}

PrimExpr IndexDataTypeRewriter::VisitExpr_(const BufferLoadNode* op) {
  BufferLoad load = GetRef<BufferLoad>(op);

  Buffer new_buffer = GetRemappedBuffer(op->buffer);
  auto indices = VisitIndices(op->indices);

  if (!new_buffer.same_as(op->buffer) || !indices.same_as(op->indices)) {
    auto writer = load.CopyOnWrite();
    writer->indices = indices;
    writer->buffer = new_buffer;
  }

  return std::move(load);
}

Array<PrimExpr> IndexDataTypeRewriter::VisitIndices(Array<PrimExpr> indices) {
  bool is_index = is_index_;
  is_index_ = true;

  auto fmutate = [this](const PrimExpr& index) { return this->VisitExpr(index); };
  indices.MutateByApply(fmutate);

  is_index_ = is_index;

  return indices;
}

Stmt IndexDataTypeRewriter::VisitStmt_(const IfThenElseNode* op) {
  IfThenElse updated = Downcast<IfThenElse>(Parent::VisitStmt_(op));
  is_condition_ = true;
  PrimExpr cond = VisitExpr(op->condition);
  is_condition_ = false;
  if (!cond.same_as(op->condition)) {
    return std::move(IfThenElse(cond, updated->then_case, updated->else_case));
  }
  return std::move(updated);
}

Stmt IndexDataTypeRewriter::VisitStmt_(const ForNode* op) {
  bool is_index = is_index_;
  is_index_ = true;
  Var new_loop_var = Downcast<Var>(VisitExpr(op->loop_var));
  PrimExpr min = VisitExpr(op->min);
  PrimExpr extent = VisitExpr(op->extent);
  is_index_ = is_index;

  if (!new_loop_var.same_as(op->loop_var) || !min.same_as(op->min) || !extent.same_as(op->extent)) {
    For new_for = GetRef<For>(op);
    auto* n = new_for.CopyOnWrite();
    n->loop_var = new_loop_var;
    n->min = min;
    n->extent = extent;
    n->body = this->VisitStmt(op->body);
    return std::move(new_for);
  } else {
    return Parent::VisitStmt_(op);
  }
}

// PrimExpr IndexDataTypeRewriter::VisitExpr_(const VarNode* op) {
//   if (auto opt_new_var = var_remap_.Get(GetRef<Var>(op)).defined()) {
//     return opt_new_var.value();
//   }
//   Var new_var = Downcast<Var>(Parent::VisitExpr_(op));
//   if (!new_var.same_as(GetRef<Var>(op))) {
//     var_remap_.Set(GetRef<Var>(op), new_var);
//   }
//   return std::move(new_var);
// }

// PrimExpr IndexDataTypeRewriter::VisitExpr_(const SizeVarNode* op) {
//   if (auto opt_new_var = var_remap_.Get(GetRef<Var>(op)).defined()) {
//     return opt_new_var.value();
//   }
//   SizeVar new_var = Downcast<SizeVar>(Parent::VisitExpr_(op));
//   if (!new_var.same_as(GetRef<SizeVar>(op))) {
//     var_remap_.Set(GetRef<SizeVar>(op), new_var);
//   }
//   return std::move(new_var);
// }

#define DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(OP, FUNC)                          \
  PrimExpr IndexDataTypeRewriter::VisitExpr_(const OP* op) {                        \
    bool is_index = is_index_;                                                      \
    bool rewrite = is_condition_ && op->a->dtype.is_int() && op->b->dtype.is_int(); \
    if (rewrite) {                                                                  \
      is_index_ = true;                                                             \
    }                                                                               \
    auto result = Parent::VisitExpr_(op);                                           \
    is_index_ = is_index;                                                           \
    return std::move(result);                                                       \
  }

DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(EQNode, operator==);
DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(NENode, operator!=);
DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(LENode, operator<=);
DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(LTNode, operator<);  // NOLINT(*)
DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(GTNode, operator>);  // NOLINT(*)
DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(GENode, operator>=);

PrimExpr IndexDataTypeRewriter::VisitExpr_(const CallNode* op) {
  // handle if_then_else condition
  if (op->op.same_as(builtin::if_then_else())) {
    bool is_condition = is_condition_;
    is_condition_ = true;
    PrimExpr cond = VisitExpr(op->args[0]);
    is_condition_ = is_condition;
    return if_then_else(cond, VisitExpr(op->args[1]), VisitExpr(op->args[2]));
  }
  return Parent::VisitExpr_(op);
}

#undef DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH

}  // namespace tir
}  // namespace tvm
