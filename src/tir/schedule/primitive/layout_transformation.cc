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
#include "../utils.h"

namespace tvm {
namespace tir {

class TransformLayoutRewriter : private StmtExprMutator {
 public:
  /*!
   * \brief Rewrite the access to the buffer after the transformation
   * \param scope_stmt The parent statement that contains all accesses to the target buffer
   * \param old_buffer The target buffer before transformation
   * \param new_buffer The new buffer after transformation
   * \param index_map The transformation applied to the buffer
   * \return The new AST rooting at the original parent scope and the map from the old block to the
   * new block
   */
  static std::pair<Stmt, Map<Block, Block>> Rewrite(const Stmt& scope_stmt,
                                                    const Buffer& old_buffer,
                                                    const Buffer& new_buffer,
                                                    const IndexMap& index_map) {
    TransformLayoutRewriter rewriter(old_buffer, new_buffer, index_map);
    Stmt result = rewriter(scope_stmt);
    return {result, rewriter.block_sref_reuse_};
  }

 private:
  TransformLayoutRewriter(const Buffer& old_buffer, const Buffer& new_buffer,
                          const IndexMap& index_map)
      : old_buffer_(old_buffer),
        new_buffer_(new_buffer),
        index_map_(index_map),
        buffer_data_to_buffer_{{new_buffer->data, new_buffer}} {}

  void RewriteBufferAccess(Buffer* buffer, Array<PrimExpr>* indices, arith::Analyzer* analyzer) {
    *buffer = new_buffer_;
    *indices = index_map_->MapIndices(*indices);
    indices->MutateByApply([&](const PrimExpr& expr) { return analyzer->Simplify(expr); });
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    BufferLoad buffer_load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    if (buffer_load->buffer.same_as(old_buffer_)) {
      auto* n = buffer_load.CopyOnWrite();
      RewriteBufferAccess(&n->buffer, &n->indices, &analyzer_);
    }
    return std::move(buffer_load);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore buffer_store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    if (buffer_store->buffer.same_as(old_buffer_)) {
      auto* n = buffer_store.CopyOnWrite();
      RewriteBufferAccess(&n->buffer, &n->indices, &analyzer_);
    }
    return std::move(buffer_store);
  }

  void RewriteAccessRegion(Array<BufferRegion>* old_access_regions,
                           const Array<BufferRegion>& infered_access_regions) {
    auto fmutate = [this, &infered_access_regions](const BufferRegion& buffer_region) {
      if (buffer_region->buffer.same_as(old_buffer_)) {
        ICHECK(infered_access_regions.size() == 1);
        return infered_access_regions[0];
      }
      return buffer_region;
    };
    (*old_access_regions).MutateByApply(fmutate);
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    for (const IterVar& iter : op->iter_vars) {
      analyzer_.Bind(iter->var, iter->dom);
    }
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    auto infered_access_regions = GetBlockReadWriteRegion(block, buffer_data_to_buffer_);
    auto* n = block.CopyOnWrite();
    RewriteAccessRegion(&n->reads, infered_access_regions[0]);
    RewriteAccessRegion(&n->writes, infered_access_regions[1]);
    block_sref_reuse_.Set(GetRef<Block>(op), block);
    return std::move(block);
  }

  const Buffer& old_buffer_;
  const Buffer& new_buffer_;
  const IndexMap& index_map_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Block, Block> block_sref_reuse_;
  arith::Analyzer analyzer_;
};

class BufferIsSubregionError : public ScheduleError {
 public:
  explicit BufferIsSubregionError(IRModule mod, Buffer buffer) : mod_(mod), buffer_(buffer) {}

  String FastErrorString() const final {
    return "ScheduleError: The input buffer is defined in `match_buffer` of a block, it is expected"
           " to be a function parameter or allocated by a block";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "ScheduleError: The input buffer " << buffer_->name << " is defined in `match_buffer` of "
       << "a block, it is expected to be a function parameter or allocated by a block.";
    return os.str();
  }

  Array<ObjectRef> LocationsOfInterest() const final { return {}; }
  IRModule mod() const final { return mod_; }

 private:
  IRModule mod_;
  Buffer buffer_;
};

void TransformLayout(ScheduleState self, const StmtSRef& block_sref, int buffer_index,
                     BufferIndexType buffer_index_type, const IndexMap& index_map) {
  const BlockNode* block_ptr = TVM_SREF_TO_BLOCK(block_ptr, block_sref);
  Buffer old_buffer =
      GetNthAccessBuffer(self, GetRef<Block>(block_ptr), buffer_index,
                         buffer_index_type == BufferIndexType::kRead ? false : true);
  Optional<StmtSRef> defining_site_sref;
  bool is_alloc;
  std::tie(defining_site_sref, is_alloc) = GetBufferDefiningSite(block_sref, old_buffer);
  if (defining_site_sref.defined() && !is_alloc) {
    throw BufferIsSubregionError(self->mod, old_buffer);
  }

  StmtSRef scope_sref = defining_site_sref.defined()
                            ? defining_site_sref.value()
                            : GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);
  const BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_block, scope_sref);

  // Step 1: Infer the shape of the new buffer
  ObjectPtr<BufferNode> new_buffer_node = make_object<BufferNode>(*(old_buffer.get()));
  new_buffer_node->shape = index_map->MapShape(old_buffer->shape);
  Buffer new_buffer{new_buffer_node};

  // Step 2: Rewrite access indices and regions of the buffer
  Stmt new_stmt;
  Map<Block, Block> block_sref_reuse;
  std::tie(new_stmt, block_sref_reuse) = TransformLayoutRewriter::Rewrite(
      GetRef<Block>(scope_block), old_buffer, new_buffer, index_map);
  Block new_scope_block = Downcast<Block>(new_stmt);

  // Step 3: Rewrite alloc_buffer of the block or buffer_map of the PrimFunc.
  if (defining_site_sref.defined()) {
    auto* n = new_scope_block.CopyOnWrite();
    n->alloc_buffers.MutateByApply([&old_buffer, &new_buffer](const Buffer& buffer) {
      if (buffer.same_as(old_buffer)) {
        return new_buffer;
      }
      return buffer;
    });
    block_sref_reuse.Set(GetRef<Block>(scope_block), new_scope_block);
  } else {
    GlobalVar g_var;
    GetRootPrimFunc(self->mod, scope_block, &g_var);
    IRModuleNode* new_mod = self->mod.CopyOnWrite();
    MapNode* new_map = new_mod->functions.CopyOnWrite();
    PrimFunc ref_new_func = Downcast<PrimFunc>(std::move(new_map->at(g_var)));
    PrimFuncNode* new_func = ref_new_func.CopyOnWrite();
    MapNode* new_buffer_map = new_func->buffer_map.CopyOnWrite();
    for (auto it = new_buffer_map->begin(); it != new_buffer_map->end(); ++it) {
      if ((*it).second.same_as(old_buffer)) {
        (*it).second = new_buffer;
      }
    }
    new_map->at(g_var) = std::move(ref_new_func);
  }

  // Step 4: Replace the scope block with the new block
  self->Replace(scope_sref, new_scope_block, block_sref_reuse);
}

class BlockVarTypeDetector : public ExprVisitor {
 public:
  static IterVarType Detect(
      const PrimExpr& expr,
      std::unordered_map<Var, IterVarType, ObjectPtrHash, ObjectPtrEqual>* block_var_type_map) {
    BlockVarTypeDetector detector(block_var_type_map);
    detector(expr);
    ICHECK(detector.found_);
    return detector.iter_type_;
  }

 private:
  explicit BlockVarTypeDetector(
      std::unordered_map<Var, IterVarType, ObjectPtrHash, ObjectPtrEqual>* block_var_type_map)
      : block_var_type_map_(block_var_type_map) {}

  void VisitExpr_(const VarNode* var) final {
    const auto& it = block_var_type_map_->find(GetRef<Var>(var));
    ICHECK(it != block_var_type_map_->end());
    if (!found_) {
      found_ = true;
      iter_type_ = it->second;
    } else {
      ICHECK_EQ(iter_type_, it->second);
    }
  }

 private:
  bool found_{false};
  IterVarType iter_type_;
  std::unordered_map<Var, IterVarType, ObjectPtrHash, ObjectPtrEqual>* block_var_type_map_;
};

class IndexSimplifer : public arith::IRMutatorWithAnalyzer {
 public:
  static Stmt SimplifyIndex(const Stmt& stmt) {
    arith::Analyzer analyzer;
    IndexSimplifer simplifier(&analyzer);
    return simplifier(stmt);
  }

 private:
  explicit IndexSimplifer(arith::Analyzer* analyzer) : IRMutatorWithAnalyzer(analyzer) {}

  using IRMutatorWithAnalyzer::VisitExpr_;
  using IRMutatorWithAnalyzer::VisitStmt_;

  void SimplifyAccessRegion(Array<BufferRegion>* old_access_regions, arith::Analyzer* analyzer_) {
    auto fmutate = [this, &analyzer_](const BufferRegion& buffer_region) {
      std::vector<Range> new_buffer_region;
      for (const auto& range : buffer_region->region) {
        new_buffer_region.push_back(Range::FromMinExtent(analyzer_->Simplify(range->min),
                                                         analyzer_->Simplify(range->extent)));
      }
      return BufferRegion(buffer_region->buffer, new_buffer_region);
    };
    (*old_access_regions).MutateByApply(fmutate);
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    for (const IterVar& iter_var : op->iter_vars) {
      analyzer_->Bind(iter_var->var, iter_var->dom);
    }
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    auto* n = block.CopyOnWrite();
    SimplifyAccessRegion(&n->reads, analyzer_);
    SimplifyAccessRegion(&n->writes, analyzer_);
    return std::move(block);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    std::vector<PrimExpr> new_indices;
    for (const PrimExpr& index : op->indices) {
      new_indices.push_back(analyzer_->Simplify(analyzer_->Simplify(index)));
    }
    return BufferStore(op->buffer, value, new_indices);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    std::vector<PrimExpr> new_indices;
    for (const PrimExpr& index : op->indices) {
      new_indices.push_back(analyzer_->Simplify(index));
    }
    return BufferLoad(op->buffer, new_indices);
  }
};

void TransformBlockLayout(ScheduleState self, const StmtSRef& block_sref,
                          const IndexMap& index_map) {
  const BlockNode* block_ptr = TVM_SREF_TO_BLOCK(block_ptr, block_sref);
  ICHECK_EQ(block_ptr->iter_vars.size(), index_map->initial_indices.size());

  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);
  const BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_block, scope_sref);

  // Affine binding
  CheckAffineBinding(self, GetRef<Block>(block_ptr));

  // Loop A Chain
  const StmtSRefNode* loop;
  for (loop = block_sref->parent; loop->parent != scope_sref.get();) {
    const ForNode* outer = loop->parent->StmtAs<ForNode>();
    const ForNode* inner = loop->StmtAs<ForNode>();
    ICHECK(outer != nullptr && inner != nullptr);
    ICHECK(outer->body.get() == inner);
    loop = loop->parent;
  }

  // Collect old block info, check only parallel and reduce
  arith::Analyzer analyzer;
  std::vector<PrimExpr> block_vars;
  std::unordered_map<Var, Range, ObjectPtrHash, ObjectPtrEqual> block_var_range_map;
  std::unordered_map<Var, IterVarType, ObjectPtrHash, ObjectPtrEqual> block_var_type_map;
  std::vector<PrimExpr> block_var_range_array;
  for (const auto& iter_var : block_ptr->iter_vars) {
    ICHECK(iter_var->iter_type == kDataPar || iter_var->iter_type == kCommReduce);
    block_vars.push_back(iter_var->var);
    block_var_range_map[iter_var->var] = iter_var->dom;
    block_var_type_map[iter_var->var] = iter_var->iter_type;
    block_var_range_array.push_back(analyzer.Simplify(iter_var->dom->min + iter_var->dom->extent));
  }
  Array<PrimExpr> res = index_map->MapIndices(block_vars);

  // New block info
  std::vector<PrimExpr> new_block_vars;
  std::vector<IterVarType> new_block_var_types;
  Array<PrimExpr> new_range = index_map->MapShape(block_var_range_array);
  std::vector<IterVar> new_block_iters;
  for (size_t i = 0; i < index_map->final_indices.size(); ++i) {
    new_block_vars.push_back(Var("bv", DataType::Int(32)));
    new_block_var_types.push_back(BlockVarTypeDetector::Detect(res[i], &block_var_type_map));
    new_block_iters.emplace_back(Range::FromMinExtent(0, new_range[i]),
                                 Downcast<Var>(new_block_vars[i]), new_block_var_types[i], "");
  }
  // New binding values
  DiagnosticContext diag_ctx(DiagnosticContext::Default(IRModule()));
  auto detect_res =
      arith::DetectIterMap(res, block_var_range_map, Bool(true), true, &analyzer);
  auto inverse_map = arith::InverseAffineIterMap(detect_res, new_block_vars);
  for (const auto& iter_var : block_ptr->iter_vars) {
    if (inverse_map.find(iter_var->var) == inverse_map.end()) {
      inverse_map.Set(iter_var->var, 0);
    }
  }

  // Make new block
  Block new_block = Downcast<Block>(Substitute(GetRef<Block>(block_ptr), inverse_map));
  new_block.CopyOnWrite()->iter_vars = new_block_iters;
  new_block = Downcast<Block>(IndexSimplifer::SimplifyIndex(new_block));

  // Make new loop vars
  std::vector<PrimExpr> new_loop_vars;
  for (size_t i = 0; i < index_map->final_indices.size(); ++i) {
    new_loop_vars.push_back(Var("ax", DataType::Int(32)));
  }

  // Make new block realize
  BlockRealize block_realize = GetBlockRealize(self, block_sref);
  BlockRealize new_block_realize(new_loop_vars, Bool(true), new_block);

  Stmt body = new_block_realize;
  for (int i = index_map->final_indices.size() - 1; i >= 0; --i) {
    body = For(Downcast<Var>(new_loop_vars[i]), 0, new_range[i], ForKind::kSerial, body);
  }

  self->Replace(GetRef<StmtSRef>(loop), body, {{GetRef<Block>(block_ptr), new_block}});
}

/******** InstructionKind Registration ********/

struct TransformLayoutTraits : public UnpackedInstTraits<TransformLayoutTraits> {
  static constexpr const char* kName = "TransformLayout";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 3;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, Integer buffer_index,
                                      Integer buffer_index_type, IndexMap index_map) {
    return sch->TransformLayout(block_rv, buffer_index,
                                static_cast<BufferIndexType>(buffer_index_type->value), index_map);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, Integer buffer_index,
                                 Integer buffer_index_type, IndexMap index_map) {
    PythonAPICall py("transform_layout");
    py.Input("block", block_rv);
    py.Input("buffer_index", buffer_index);
    py.Input("buffer_index_type",
             BufferIndexType2Str(static_cast<BufferIndexType>(buffer_index_type->value)));
    py.Input("index_map", index_map->ToPythonString());
    return py.Str();
  }

 public:
  static ObjectRef AttrsAsJSON(const Array<ObjectRef>& attrs) {
    Array<ObjectRef> attrs_record;
    attrs_record.reserve(kNumAttrs);
    attrs_record.push_back(attrs[0]);
    attrs_record.push_back(attrs[1]);
    attrs_record.push_back(String(::tvm::SaveJSON(attrs[2])));
    return std::move(attrs_record);
  }

  static Array<ObjectRef> AttrsFromJSON(const ObjectRef& attrs_record_) {
    Array<ObjectRef> attrs_record = Downcast<Array<ObjectRef>>(attrs_record_);
    Array<ObjectRef> attrs;
    attrs.push_back(attrs_record[0]);
    attrs.push_back(attrs_record[1]);
    attrs.push_back(::tvm::LoadJSON(Downcast<String>(attrs_record[2])));
    return attrs;
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct TransformBlockLayoutTraits : public UnpackedInstTraits<TransformBlockLayoutTraits> {
  static constexpr const char* kName = "TransformBlockLayout";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, IndexMap index_map) {
    return sch->TransformBlockLayout(block_rv, index_map);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, IndexMap index_map) {
    PythonAPICall py("transform_block_layout");
    py.Input("block", block_rv);
    py.Input("index_map", index_map->ToPythonString());
    return py.Str();
  }

 public:
  static ObjectRef AttrsAsJSON(const Array<ObjectRef>& attrs) {
    Array<ObjectRef> attrs_record;
    attrs_record.reserve(kNumAttrs);
    attrs_record.push_back(String(::tvm::SaveJSON(attrs[0])));
    return std::move(attrs_record);
  }

  static Array<ObjectRef> AttrsFromJSON(const ObjectRef& attrs_record_) {
    Array<ObjectRef> attrs_record = Downcast<Array<ObjectRef>>(attrs_record_);
    Array<ObjectRef> attrs;
    attrs.push_back(::tvm::LoadJSON(Downcast<String>(attrs_record[0])));
    return attrs;
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(TransformLayoutTraits);
TVM_REGISTER_INST_KIND_TRAITS(TransformBlockLayoutTraits);

}  // namespace tir
}  // namespace tvm
