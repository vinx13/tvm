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

#include "create_primfunc.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ir/name_supply.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../../tir/ir/functor_common.h"
#include "../../tir/transforms/ir_utils.h"
#include "../schedule/graph.h"

namespace tvm {
namespace tir {

/*! \brief The helper mutator that transforms ProducerLoad to BufferLoad */
class ProducerToBufferTransformer : public StmtExprMutator {
 public:
  explicit ProducerToBufferTransformer(const std::unordered_map<te::Tensor, Buffer>& tensor2buffers)
      : tensor2buffers_(tensor2buffers) {}

  PrimExpr VisitExpr_(const ProducerLoadNode* op) final {
    auto visited_op = Downcast<ProducerLoad>(StmtExprMutator::VisitExpr_(op));
    te::Tensor tensor = Downcast<te::Tensor>(visited_op->producer);
    auto it = tensor2buffers_.find(tensor);
    ICHECK(it != tensor2buffers_.end()) << "IndexError: Cannot find the tensor " << tensor;
    const Buffer& buffer = it->second;
    return BufferLoad(buffer, visited_op->indices);
  }

 private:
  /*! \brief The Map from Operations to buffers */
  const std::unordered_map<te::Tensor, Buffer>& tensor2buffers_;
};

/*! \brief Helper data structure to store information. */
struct CreateFuncInfo {
  /*! \brief The Tensor arg_list. */
  Array<te::Tensor> arg_list;
  /*! \brief The map from each Tensor to its corresponding buffer. */
  std::unordered_map<te::Tensor, Buffer> tensor2buffers;
  /*! \brief The transformer from ProducerLoad to BufferLoad. */
  ProducerToBufferTransformer transformer;
  /*! \brief The buffers should be allocated at function root. */
  Array<Buffer> root_alloc;
  /*! \brief The NameSupply to make block name unique. */
  NameSupply name_supply = NameSupply("");

  String FreshName(String base_name) { return name_supply->FreshName(base_name); }

  explicit CreateFuncInfo(Array<te::Tensor> arg_list)
      : arg_list(std::move(arg_list)), transformer(tensor2buffers) {}

  bool IsArg(const te::Tensor& tensor) const {
    return std::any_of(arg_list.begin(), arg_list.end(),
                       [&tensor](const te::Tensor& arg) { return tensor == arg; });
  }
};

class LayoutFreePlaceholdersNormalizer : public StmtMutator {
 public:
  PrimFunc Process(PrimFunc func) {
    for (int i = 0, n = func->params.size(); i < n; ++i) {
      if (const auto* v = func->params[i].as<VarNode>()) {
        if (Optional<Buffer> buffer = func->buffer_map.Get(GetRef<Var>(v))) {
          buffer2index_[buffer.value()] = i;
        }
      }
    }
    PrimFuncNode* f = func.CopyOnWrite();
    f->body = VisitStmt(std::move(f->body));
    if (this->layout_free_buffer_indices_.empty()) {
      return func;
    }
    Array<Integer> indices;
    indices.reserve(this->layout_free_buffer_indices_.size());
    for (int i : this->layout_free_buffer_indices_) {
      indices.push_back(Integer(i));
    }
    return WithAttr(std::move(func), tir::attr::layout_free_buffers, indices);
  }

  Stmt VisitStmt_(const BlockNode* _block) final {
    Block block = Downcast<Block>(StmtMutator::VisitStmt_(_block));
    if (Optional<ObjectRef> ann = block->annotations.Get(topi_attr)) {
      Array<Buffer> new_buffers;
      for (Buffer buffer : Downcast<Array<Buffer>>(ann)) {
        auto it = buffer2index_.find(buffer);
        if (it != buffer2index_.end()) {
          layout_free_buffer_indices_.insert(it->second);
        } else {
          new_buffers.push_back(buffer);
        }
      }
      block.CopyOnWrite()->annotations.Set(topi_attr, new_buffers);
    }
    return std::move(block);
  }

  std::unordered_map<tir::Buffer, int, ObjectPtrHash, ObjectPtrEqual> buffer2index_;
  std::set<int> layout_free_buffer_indices_;
  String topi_attr = "layout_free_placeholders";
};

// class DataTypeNormalizer : public IndexDataTypeRewriter {
//  public:
//   Stmt VisitStmt_(const BlockNode* op) final {
//     auto new_iters = op->iter_vars.Map([&](const IterVar& iter) {
//       if (iter->var->dtype != target_data_type_) {
//         IterVar new_iter = iter;
//         IterVarNode* new_iter_node = new_iter.CopyOnWrite();
//         new_iter_node->var = iter->var.copy_with_dtype(target_data_type_);
//         var_remap_.Set(iter->var, new_iter->var);
//         new_iter_node->dom = Range::FromMinExtent(cast(target_data_type_, iter->dom->min),
//                                                   cast(target_data_type_, iter->dom->extent));
//         return new_iter;
//       }
//       return iter;
//     });
//     // Block access region is not handled. It assumes block access region will be added by later
//     and
//     // they should be empty at this point.
//     ICHECK(op->reads.empty());
//     ICHECK(op->writes.empty());

//     if (!new_iters.same_as(op->iter_vars)) {
//       Block new_block = GetRef<Block>(op);
//       new_block.CopyOnWrite()->iter_vars = std::move(new_iters);
//       return IndexDataTypeRewriter::VisitStmt_(new_block.get());
//     }
//     return IndexDataTypeRewriter::VisitStmt_(op);
//   }

//   Stmt VisitStmt_(const ForNode* op) final {
//     if (op->loop_var->dtype != target_data_type_) {
//       auto new_loop_var = op->loop_var.copy_with_dtype(target_data_type_);
//       var_remap_.Set(op->loop_var, new_loop_var);
//     }
//     return IndexDataTypeRewriter::VisitStmt_(op);
//   }

//   PrimExpr VisitExpr_(const VarNode* op) final {
//     if (auto it = var_remap_.find(GetRef<Var>(op)); it != var_remap_.end()) {
//       return (*it).second;
//     }
//     return GetRef<Var>(op);
//   }

//   PrimExpr VisitExpr_(const IntImmNode* op) final {
//     if (is_index_) {
//       return cast(target_data_type_, GetRef<IntImm>(op));
//     }
//     return GetRef<IntImm>(op);
//   }

//   // Buffer VisitBuffer(const Buffer& buffer) {
//   //   auto new_shape = buffer->shape.Map([this](const PrimExpr& e) { return
//   cast(target_data_type_,
//   //   e); }); if (!new_shape.same_as(buffer->shape)) {
//   //     Buffer new_buffer = buffer;
//   //     new_buffer.CopyOnWrite()->shape = std::move(new_shape);
//   //     return new_buffer;
//   //   }
//   //   return buffer;
//   // }

//   // PrimExpr VisitExpr_(const BufferLoadNode* op) final {
//   //   auto new_buffer = VisitBuffer(op->buffer);
//   //   if (!new_buffer.same_as(op->buffer)) {
//   //     BufferLoad new_load = GetRef<BufferLoad>(op);
//   //     new_load.CopyOnWrite()->buffer = std::move(new_buffer);
//   //     return IndexDataTypeRewriter::VisitExpr_(new_load.get());
//   //   }
//   //   return IndexDataTypeRewriter::VisitExpr_(op);
//   // }

//   // Stmt VisitStmt_(const BufferStoreNode* op) final {
//   //   auto new_buffer = VisitBuffer(op->buffer);
//   //   if (!new_buffer.same_as(op->buffer)) {
//   //     BufferStore new_store = GetRef<BufferStore>(op);
//   //     new_store.CopyOnWrite()->buffer = std::move(new_buffer);
//   //     return IndexDataTypeRewriter::VisitStmt_(new_store.get());
//   //   }
//   //   return IndexDataTypeRewriter::VisitStmt_(op);
//   // }

//   DataType target_data_type_{DataType::Int(64)};
//   Map<Var, Var> var_remap_;
// };

BlockRealize GenerateBlockFromTensors(const te::ComputeOp& compute_op,
                                      const Array<te::Tensor>& tensors, Array<PrimExpr> bindings,
                                      PrimExpr expr_body, CreateFuncInfo* info,
                                      arith::Analyzer* analyzer) {
  // Step 1. Push_back data_par axis and reduce_axis into block_vars.
  Array<IterVar> iter_vars;
  std::unordered_map<const VarNode*, PrimExpr> var_map;
  iter_vars.reserve(compute_op->axis.size() + compute_op->reduce_axis.size());
  auto f_push_block_vars = [&iter_vars, &var_map, &analyzer](const Array<IterVar>& iters) {
    for (IterVar iter_var : iters) {
      // Create new var
      Var new_var(iter_var->var->name_hint, iter_var->var->dtype);
      var_map[iter_var->var.get()] = new_var;

      const PrimExpr& dom_min = analyzer->Simplify(iter_var->dom->min);
      const PrimExpr& dom_extent = analyzer->Simplify(iter_var->dom->extent);
      iter_vars.push_back(IterVar(Range::FromMinExtent(dom_min, dom_extent), new_var,
                                  iter_var->iter_type, iter_var->thread_tag, iter_var->span));
    }
  };
  f_push_block_vars(compute_op->axis);
  f_push_block_vars(compute_op->reduce_axis);

  // Step 2.
  //  - Declare buffers
  //  - Update `op2buffers`
  //  - Add the non-argument tensors to `alloc_buffer` of the root block
  Array<Buffer> buffers;
  for (const te::Tensor& tensor : tensors) {
    Buffer buffer = decl_buffer(tensor->shape, tensor->dtype, tensor->GetNameHint(), "global");
    info->tensor2buffers[tensor] = buffer;
    buffers.push_back(buffer);

    if (!info->IsArg(tensor)) {
      info->root_alloc.push_back(info->tensor2buffers[tensor]);
    }
  }

  // Step 3. Calculate indices for BufferStore
  Array<PrimExpr> indices;
  indices.reserve(compute_op->axis.size());
  for (const IterVar& iter_var : compute_op->axis) {
    auto it = var_map.find(iter_var->var.get());
    ICHECK(it != var_map.end());
    indices.push_back(it->second);
  }

  // Step 4. Create block body.
  String block_name{nullptr};
  Optional<Stmt> init = NullOpt;
  Stmt body;
  if (const auto* reduce = expr_body.as<ReduceNode>()) {
    // Case 1. Reduce compute
    block_name = info->FreshName(compute_op->name);
    int n_buffers = buffers.size();

    Array<PrimExpr> lhs;
    Array<PrimExpr> rhs;
    lhs.reserve(n_buffers);
    rhs.reserve(n_buffers);

    // Make the LHS operands and RHS operands:
    //  - A LHS operand is the buffer storing the reduction result, with corresponding indices.
    //  - A RHS operand is the value to be reduced.
    for (int i = 0; i < n_buffers; ++i) {
      const PrimExpr& left = BufferLoad(buffers[i], indices);
      const PrimExpr& right =
          analyzer->Simplify(Substitute(info->transformer(reduce->source[i]), var_map));
      lhs.push_back(left);
      rhs.push_back(right);
      ICHECK_EQ(left->dtype, right->dtype);
    }

    Array<Var> temp_vars;
    Array<Stmt> body_stmts;
    Array<Stmt> init_stmts;
    temp_vars.reserve(n_buffers);
    body_stmts.reserve(n_buffers);
    init_stmts.reserve(n_buffers);

    // - When there is only one buffer, we directly create a BufferStore which stores "combiner(lhs,
    //   rhs)" into the target buffer position.
    // - In case there are multiple buffers, to avoid incorrect results, we create some intermediate
    //   variables and use LetStmts to bind the variables with "combiner(lhs, rhs)". After that, we
    //   then store the value of the variables into the target buffer positions.
    for (int i = 0; i < n_buffers; ++i) {
      const Buffer& buffer = buffers[i];
      init_stmts.push_back(BufferStore(buffer, reduce->combiner->identity_element[i], indices));
      PrimExpr value{nullptr};
      if (n_buffers > 1) {
        temp_vars.push_back(Var("v_" + buffer->name, PrimType(lhs[i].dtype())));
        value = temp_vars.back();
      } else {
        value = reduce->combiner.get()->operator()(lhs, rhs)[i];
      }
      body_stmts.push_back(BufferStore(buffer, value, indices));
    }

    init = SeqStmt::Flatten(init_stmts);
    body = SeqStmt::Flatten(body_stmts);
    if (n_buffers > 1) {
      // When there are multiple buffers, we wrap the body with LetStmts.
      for (int i = n_buffers - 1; i >= 0; --i) {
        PrimExpr value = reduce->combiner.get()->operator()(lhs, rhs)[i];
        body = LetStmt(temp_vars[i], std::move(value), std::move(body));
      }
    }
  } else {
    // Case 2. Data parallel compute
    ICHECK_EQ(tensors.size(), 1);
    block_name = info->FreshName(tensors[0]->GetNameHint());
    const PrimExpr& compute_body = Substitute(info->transformer(expr_body), var_map);
    body = BufferStore(info->tensor2buffers[tensors[0]], analyzer->Simplify(compute_body), indices);
  }

  // Step 5. Add script_parsing_detect_access attr for auto complete the whole IR.
  Map<String, ObjectRef> annotations;
  auto mutate_attr = [&info](const ObjectRef& value) -> ObjectRef {
    if (const auto* tensor_value = value.as<te::TensorNode>()) {
      return info->tensor2buffers.at(GetRef<te::Tensor>(tensor_value));
    } else {
      return value;
    }
  };

  for (const auto& pair : compute_op->attrs) {
    const String& key = pair.first;
    const ObjectRef& value = pair.second;
    // TensorIR will not allow Tensor data structure
    if (value->IsInstance<ArrayNode>()) {
      const auto array_value = Downcast<Array<ObjectRef>>(value);
      annotations.Set(key, array_value.Map(mutate_attr));
    } else {
      annotations.Set(key, mutate_attr(value));
    }
  }
  // Set script_parsing_detect_access
  annotations.Set(tir::attr::script_parsing_detect_access, IntImm(DataType::Int(32), 3));
  if (iter_vars.empty()) {
    IterVar iter(Range::FromMinExtent(0, 1), Var("vi", DataType::Int(32)), IterVarType::kDataPar);
    PrimExpr binding(0);
    iter_vars.push_back(iter);
    bindings.push_back(binding);
  }

  // Step 6. Create Block and BlockRealize.
  return BlockRealize(/*iter_values=*/std::move(bindings),
                      /*predicate=*/Bool(true),
                      /*block=*/
                      Block(/*iter_vars=*/std::move(iter_vars),
                            /*reads=*/{},
                            /*writes=*/{},
                            /*name_hint=*/block_name,
                            /*body=*/std::move(body),
                            /*init=*/std::move(init),
                            /*alloc_buffers=*/{},
                            /*match_buffers=*/{},
                            /*annotations=*/std::move(annotations)));
}

Stmt GenerateStmtFromCompute(const te::ComputeOp& compute_op, CreateFuncInfo* info,
                             arith::Analyzer* analyzer) {
  // Step 1. Creating loop vars for block bindings.
  Array<IterVar> axes = compute_op->axis;
  axes.insert(axes.end(), compute_op->reduce_axis.begin(), compute_op->reduce_axis.end());
  Array<PrimExpr> bindings;
  for (size_t i = 0; i < axes.size(); ++i) {
    const IterVar& axis = axes[i];
    int bits = std::max(axis->dom->min.dtype().bits(), axis->dom->extent.dtype().bits());
    bindings.push_back(Var("i" + std::to_string(i), runtime::DataType::Int(bits)));
  }
  // Step 2. Generate block bodies.
  Array<Stmt> seq_stmt;
  if (compute_op->body[0]->IsInstance<ReduceNode>()) {
    auto f_reducer_equal = [](const ReduceNode* a, const ReduceNode* b) -> bool {
      return a->combiner.same_as(b->combiner) &&    //
             a->source.same_as(b->source) &&        //
             a->axis.same_as(b->axis) &&            //
             a->condition.same_as(b->condition) &&  //
             ((a->init.empty() && b->init.empty()) || a->init.same_as(b->init));
    };

    PrimExpr expr_body = compute_op->body[0];
    Array<te::Tensor> tensors = {compute_op.output(0)};
    const tir::ReduceNode* reduce = expr_body.as<tir::ReduceNode>();
    // specially handle reduction inline for multiplre reductions.
    for (size_t k = 1; k < compute_op->body.size(); ++k) {
      const tir::ReduceNode* reduce_ = compute_op->body[k].as<tir::ReduceNode>();
      ICHECK(reduce_);
      ICHECK(f_reducer_equal(reduce_, reduce))
          << "The Reduce inputs of ComputeOp should have the same attribute except value_index";
      tensors.push_back(compute_op.output(k));
    }

    seq_stmt.push_back(GenerateBlockFromTensors(compute_op, tensors, bindings, std::move(expr_body),
                                                info, analyzer));
  } else {
    for (int i = 0; i < compute_op->num_outputs(); ++i) {
      const te::Tensor& tensor = compute_op.output(i);
      PrimExpr expr_body = compute_op->body[i];
      seq_stmt.push_back(GenerateBlockFromTensors(compute_op, {tensor}, bindings,
                                                  std::move(expr_body), info, analyzer));
    }
  }

  Stmt body = SeqStmt::Flatten(seq_stmt);

  // Step 3. Generate loop nesting.
  for (size_t i = axes.size(); i > 0; --i) {
    const IterVar& axis = axes[i - 1];
    PrimExpr dom_min = analyzer->Simplify(axis->dom->min);
    PrimExpr dom_extent = analyzer->Simplify(axis->dom->extent);
    const Var& loop_var = Downcast<Var>(bindings[i - 1]);
    body = For(loop_var, dom_min, dom_extent, ForKind::kSerial, body);
  }

  // body = DataTypeNormalizer()(body);

  return body;
}

Stmt GenerateStmtFromExternOp(const te::ExternOp& extern_op, CreateFuncInfo* info) {
  // Step 1. Check all inputs are visited before and update var_map.
  std::unordered_map<const VarNode*, PrimExpr> var_map;
  ICHECK_EQ(extern_op->inputs.size(), extern_op->input_placeholders.size());
  for (size_t i = 0; i < extern_op->inputs.size(); ++i) {
    const Buffer& placeholder = extern_op->input_placeholders[i];
    const te::Tensor& input_tensor = extern_op->inputs[i];
    auto it = info->tensor2buffers.find(input_tensor);
    ICHECK(it != info->tensor2buffers.end());
    var_map[placeholder->data.get()] = it->second->data;
  }

  // Step 2. Update info with its output tensor and placeholder buffer.
  ICHECK_EQ(extern_op->num_outputs(), extern_op->output_placeholders.size());
  for (int i = 0; i < extern_op->num_outputs(); ++i) {
    const Buffer& placeholder = extern_op->output_placeholders[i];
    const te::Tensor& output_tensor = extern_op.output(i);
    info->tensor2buffers[output_tensor] = placeholder;
    if (!info->IsArg(output_tensor)) {
      info->root_alloc.push_back(placeholder);
    }
  }

  // Step 3. Collect Access Region
  Array<BufferRegion> reads, writes;
  for (const te::Tensor& tensor : extern_op->inputs) {
    // We have ICHECK before so it is not needed here.
    reads.push_back(BufferRegion::FullRegion(info->tensor2buffers[tensor]));
  }
  for (const Buffer& buffer : extern_op->output_placeholders) {
    writes.push_back(BufferRegion::FullRegion(buffer));
  }

  Stmt body = Substitute(extern_op->body, var_map);

  // Step 4. Generate opaque block as body.
  return BlockRealize(/*iter_values=*/{},
                      /*predicate=*/Bool(true),
                      /*block=*/
                      Block(/*iter_vars=*/{},
                            /*reads=*/std::move(reads),
                            /*writes=*/std::move(writes),
                            /*name_hint=*/info->FreshName(extern_op->name),
                            /*body=*/std::move(body),
                            /*init=*/NullOpt,
                            /*alloc_buffers=*/{},
                            /*match_buffers=*/{},
                            /*annotations=*/extern_op->attrs));
}

Array<te::Operation> CollectOrderedOps(const Array<te::Tensor>& arg_list) {
  Array<te::Operation> arg_ops;
  for (const te::Tensor& arg : arg_list) {
    arg_ops.push_back(arg->op);
  }
  te::ReadGraph g = te::CreateReadGraph(arg_ops);
  Array<te::Operation> order = te::PostDFSOrder(arg_ops, g);

  for (const te::Operation& op : order) {
    if (!(op->IsInstance<te::PlaceholderOpNode>() || op->IsInstance<te::ComputeOpNode>() ||
          op->IsInstance<te::ExternOpNode>()))
      LOG(FATAL) << "TypeError: Unsupported Operation: " << op->GetTypeKey() << ". "
                 << "Only te.placeholder and te.compute are allowed for now.";
  }
  return order;
}

void InitializeBufferBinds(const Array<te::Operation>& ordered_ops, CreateFuncInfo* info) {
  // Process any TE operations which contain user defined buffers
  for (const auto& op : ordered_ops) {
    // Initialize the tensor2buffer binds map with buffers defined by the te.extern
    if (const auto* extern_op = op.as<te::ExternOpNode>()) {
      ICHECK_EQ(extern_op->inputs.size(), extern_op->input_placeholders.size());
      for (size_t i = 0; i < extern_op->inputs.size(); ++i) {
        const te::Tensor& input = extern_op->inputs[i];
        const Buffer& buffer = extern_op->input_placeholders[i];
        info->tensor2buffers[input] = buffer;
      }
    }
  }
}

void RewriteStageToBlock(const te::Operation& op, CreateFuncInfo* info, Array<Stmt>* root_stmts,
                         arith::Analyzer* analyzer) {
  if (const auto* placeholder = op.as<te::PlaceholderOpNode>()) {
    // Case 1. PlaceholderOp (te.placeholder)
    ICHECK_EQ(op->num_outputs(), 1);
    const te::Tensor& tensor = op.output(0);
    // Check op is in op list
    ICHECK(info->IsArg(tensor));
    // Declare a buffer for any argument tensors without a pre-existing
    // buffer declaration recorded in the tensor2buffer binds map
    if (info->tensor2buffers.count(tensor) == 0) {
      const Buffer& buffer =
          decl_buffer(placeholder->shape, placeholder->dtype, placeholder->name, "global");
      info->tensor2buffers[tensor] = buffer;
    }
  } else if (const auto* compute_op = op.as<te::ComputeOpNode>()) {
    // Case 2. ComputeOp (te.compute)
    root_stmts->push_back(
        GenerateStmtFromCompute(GetRef<te::ComputeOp>(compute_op), info, analyzer));
  } else if (const auto extern_op = op.as<te::ExternOpNode>()) {
    // Case 3. ExternOp (te.extern)
    root_stmts->push_back(GenerateStmtFromExternOp(GetRef<te::ExternOp>(extern_op), info));
  } else {
    ICHECK(false) << "TypeError: Unsupported Operation: " << op->GetTypeKey() << ". "
                  << "Only te.placeholder and te.compute are allowed for now.";
  }
}

PrimFunc GenerateAndCompletePrimFunc(const Array<te::Tensor>& arg_list,
                                     const Array<Stmt>& root_stmts, CreateFuncInfo* info) {
  Array<Var> parameters;
  Map<Var, Buffer> buffer_map;
  for (const te::Tensor& tensor : arg_list) {
    Var arg("var_" + tensor->GetNameHint(), PrimType(DataType::Handle()));
    parameters.push_back(arg);
    auto it = info->tensor2buffers.find(tensor);
    ICHECK(it != info->tensor2buffers.end());
    buffer_map.Set(arg, it->second);
  }
  PrimFunc func = WithAttrs(PrimFunc(/*params=*/std::move(parameters),
                                     /*body=*/SeqStmt::Flatten(root_stmts),
                                     /*ret_type=*/VoidType(),
                                     /*buffer_map=*/std::move(buffer_map)),
                            {{"global_symbol", String("main")}, {"tir.noalias", Bool(true)}});
  const auto* complete = runtime::Registry::Get("script.Complete");
  ICHECK(complete);
  func = (*complete)(std::move(func), info->root_alloc);
  return func;
}

// class BufferShapeDtypeNormalizer : public StmtExprMutator {
//  public:
//   static PrimFunc Rewrite(PrimFunc func) {
//     BufferShapeDtypeNormalizer normalizer;
//     Map<Var, Buffer> new_buffer_map = func->buffer_map;
//     for (const auto& [var, buffer] : func->buffer_map) {
//       new_buffer_map.Set(var, normalizer.VisitBuffer(buffer));
//     }
//     PrimFuncNode* new_func = func.CopyOnWrite();
//     new_func->buffer_map = std::move(new_buffer_map);
//     new_func->body = normalizer(std::move(new_func->body));
//     return func;
//   }

//   Buffer VisitBuffer(const Buffer& buffer) {
//     Array<PrimExpr> new_shape =
//         buffer->shape.Map([this](const PrimExpr& expr) { return cast(target_dtype_, expr); });
//     if (!new_shape.same_as(buffer->shape)) {
//       Buffer new_buffer = buffer;
//       new_buffer.CopyOnWrite()->shape = new_shape;
//       buffer_remap_.Set(buffer, new_buffer);
//       return new_buffer;
//       // rewrite stride?
//     }
//     return buffer;
//   }

//   PrimExpr VisitExpr_(const BufferLoadNode* op) final {
//     BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
//     if (auto it = buffer_remap_.find(load->buffer); it != buffer_remap_.end()) {
//       load.CopyOnWrite()->buffer = (*it).second;
//     }
//     return std::move(load);
//   }

//   Stmt VisitStmt_(const BufferStoreNode* op) final {
//     BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
//     if (auto it = buffer_remap_.find(store->buffer); it != buffer_remap_.end()) {
//       store.CopyOnWrite()->buffer = (*it).second;
//     }
//     return std::move(store);
//   }

//   Stmt VisitStmt_(const BlockNode* op) final {
//     Array<Buffer> new_alloc_buffers =
//         op->alloc_buffers.Map([this](const Buffer& buffer) { return VisitBuffer(buffer); });
//     ICHECK(op->match_buffers.empty());  // not supported
//     Block new_block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
//     auto f_mutate_buffer_region = [this](const BufferRegion& region) {
//       if (auto it = buffer_remap_.find(region->buffer); it != buffer_remap_.end()) {
//         BufferRegion new_region = region;
//         new_region.CopyOnWrite()->buffer = (*it).second;
//         return std::move(new_region);
//       }
//       return region;
//     };
//     BlockNode* new_block_node = new_block.CopyOnWrite();
//     new_block_node->alloc_buffers = std::move(new_alloc_buffers);
//     new_block_node->reads = new_block_node->reads.Map(f_mutate_buffer_region);
//     new_block_node->writes = new_block_node->writes.Map(f_mutate_buffer_region);
//     new_block_node->annotations = VisitBlockAnnotations(new_block_node->annotations);
//     return new_block;
//   }

//   Map<String, ObjectRef> VisitBlockAnnotations(const Map<String, ObjectRef>& annotations) {
//     auto new_annotations = annotations;

//     std::function<ObjectRef(const ObjectRef&)> f_mutate_obj =
//         [this, &f_mutate_obj](const ObjectRef& obj) -> ObjectRef {
//       if (obj->IsInstance<BufferNode>()) {
//         if (auto it = buffer_remap_.find(Downcast<Buffer>(obj)); it != buffer_remap_.end()) {
//           return (*it).second;
//         }
//         return obj;
//       } else if (obj->IsInstance<ArrayNode>()) {
//         return Downcast<Array<ObjectRef>>(obj).Map(f_mutate_obj);
//       }
//       return obj;
//     };
//     for (const auto& [key, value] : annotations) {
//       auto new_value = f_mutate_obj(value);
//       if (!new_value.same_as(value)) {
//         new_annotations.Set(key, new_value);
//       }
//     }
//     return new_annotations;
//   }

//  private:
//   DataType target_dtype_{DataType::Int(64)};
//   Map<Buffer, Buffer> buffer_remap_;
// };

class Normalizer : public IndexDataTypeRewriter {
 public:
  Normalizer(DataType target_data_type) : target_data_type_(std::move(target_data_type)) {}
  PrimFunc Rewrite(PrimFunc func) {
    Map<Var, Buffer> new_buffer_map = func->buffer_map;
    for (const auto& [var, buffer] : func->buffer_map) {
      new_buffer_map.Set(var, VisitBuffer(buffer));
    }
    PrimFuncNode* new_func = func.CopyOnWrite();
    new_func->buffer_map = std::move(new_buffer_map);
    new_func->body = VisitStmt(std::move(new_func->body));
    return func;
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    Block new_block = Downcast<Block>(IndexDataTypeRewriter::VisitStmt_(op));
    auto new_annotations = VisitBlockAnnotations(new_block->annotations);
    if (!new_annotations.same_as(new_block->annotations)) {
      new_block.CopyOnWrite()->annotations = std::move(new_annotations);
    }
    return std::move(new_block);
  }

  Map<String, ObjectRef> VisitBlockAnnotations(const Map<String, ObjectRef>& annotations) {
    auto new_annotations = annotations;

    std::function<ObjectRef(const ObjectRef&)> f_mutate_obj =
        [this, &f_mutate_obj](const ObjectRef& obj) -> ObjectRef {
      if (!obj.defined()) {
        return obj;
      }
      if (obj->IsInstance<BufferNode>()) {
        Buffer buffer = Downcast<Buffer>(obj);
        if (Buffer new_buffer = GetRemappedBuffer(buffer); !new_buffer.same_as(buffer)) {
          return new_buffer;
        }
      } else if (obj->IsInstance<ArrayNode>()) {
        return Downcast<Array<ObjectRef>>(obj).Map(f_mutate_obj);
      }
      return obj;
    };
    for (const auto& [key, value] : annotations) {
      auto new_value = f_mutate_obj(value);
      if (!new_value.same_as(value)) {
        new_annotations.Set(key, new_value);
      }
    }
    return new_annotations;
  }

  PrimExpr VisitExpr_(const IntImmNode* op) final {
    if (is_enabled_) {
      ICHECK_LE(op->value, Downcast<Integer>(max_value(target_data_type_))->value);
      return cast(target_data_type_, GetRef<IntImm>(op));
    }
    return GetRef<IntImm>(op);
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    if (auto it = var_remap_.find(GetRef<Var>(op)); it != var_remap_.end()) {
      return (*it).second;
    }
    if (is_enabled_) {
      Var new_var = GetRef<Var>(op).copy_with_dtype(target_data_type_);
      var_remap_.Set(GetRef<Var>(op), new_var);
      return std::move(new_var);
    }
    return GetRef<PrimExpr>(op);
  }

  PrimExpr VisitExpr_(const SizeVarNode* op) final {
    if (auto it = var_remap_.find(GetRef<Var>(op)); it != var_remap_.end()) {
      return (*it).second;
    }
    if (is_enabled_) {
      ICHECK_LE(op->dtype.bits(), target_data_type_.bits());
      Var new_var = GetRef<Var>(op).copy_with_dtype(target_data_type_);
      var_remap_.Set(GetRef<Var>(op), new_var);
      return std::move(new_var);
    }
    return GetRef<PrimExpr>(op);
  }

  DataType target_data_type_ = DataType::Int(64);
};

PrimFunc CreatePrimFuncWithConstants(const Array<te::Tensor>& arg_list,
                                     const Array<runtime::NDArray>& constants,
                                     std::optional<DataType> index_dtype_override) {
  // Infomations used in CreatePrimFunc and its sub-functions.
  CreateFuncInfo info(arg_list);
  // Root body stmts.
  Array<Stmt> root_stmts;
  // Analyzer
  arith::Analyzer analyzer;

  // Step 1. Create ordered array of operations and validate they are supported.
  Array<te::Operation> order = CollectOrderedOps(arg_list);

  // Step 2. Initialize buffer binds map
  InitializeBufferBinds(order, &info);

  // Step 3. Rewrite compute stages into blocks.
  for (const te::Operation& op : order) {
    RewriteStageToBlock(op, &info, &root_stmts, &analyzer);
  }

  // Step 4. Create func and complete prim func.
  auto func = GenerateAndCompletePrimFunc(arg_list, root_stmts, &info);
  func = tir::BindParams(func, constants);
  // LOG(INFO)  <<"Rewrite";
  // func = BufferShapeDtypeNormalizer::Rewrite(std::move(func));
  if (index_dtype_override.has_value()) {
    LOG(INFO) << index_dtype_override.value();
    func = Normalizer(index_dtype_override.value()).Rewrite(std::move(func));
  }
  // LOG(INFO) << "OK";
  // LOG(INFO) << func;
  auto result = LayoutFreePlaceholdersNormalizer().Process(std::move(func));
  // LOG(INFO) << "OK";/
  return result;
}

PrimFunc CreatePrimFunc(const Array<te::Tensor>& arg_list,
                        std::optional<DataType> index_dtype_override) {
  return CreatePrimFuncWithConstants(arg_list, {}, index_dtype_override);
}

TVM_REGISTER_GLOBAL("te.CreatePrimFunc").set_body([](TVMArgs args, TVMRetValue* ret) {
  Array<te::Tensor> arg_list = args[0];
  std::optional<DataType> index_dtype_override{std::nullopt};
  // Add conversion to make std::optional compatible with FFI.
  if (args[1].type_code() != kTVMNullptr) {
    LOG(INFO) << args[1].type_code();
    index_dtype_override = args[1].operator DataType();
  }
  LOG(INFO) << runtime::ArgTypeCode2Str(args[1].type_code());
  *ret = CreatePrimFunc(arg_list, index_dtype_override);
});

}  // namespace tir
}  // namespace tvm
