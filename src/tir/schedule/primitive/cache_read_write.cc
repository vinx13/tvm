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

/******** Error Classes ********/

class NotSingleWriteBlock : public ScheduleError {
 public:
  explicit NotSingleWriteBlock(IRModule mod, Buffer buffer, Array<StmtSRef> write_blocks)
      : mod_(std::move(mod)), buffer_(std::move(buffer)) {
    ICHECK_GT(write_blocks.size(), 1);
    write_blocks_.reserve(write_blocks.size());
    for (const StmtSRef& block_sref : write_blocks) {
      const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
      write_blocks_.push_back(GetRef<Block>(block));
    }
  }

  String FastErrorString() const final {
    return "ScheduleError: The buffer is allowed to be written by single block.";
  }

  String DetailRenderTemplate() const final {
    size_t k = write_blocks_.size();
    return "The buffer " + buffer_->name + " is expected to be written by single block, but got " +
           std::to_string(k) + " blocks who write it.";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final {
    return {write_blocks_.begin(), write_blocks_.end()};
  }

 private:
  IRModule mod_;
  Buffer buffer_;
  Array<Block> write_blocks_;
};

/******** Helper Functions/Classes ********/

/*! \brief The auxiliary info used for the insertion point and content of the cache stage. */
struct CacheStageInfo {
  /*! \brief The buffer to be read. */
  Buffer read_buffer;
  /*! \brief The buffer to be written. */
  Buffer write_buffer;
  /*! \brief The buffer allocation to be inserted into the block signature. */
  Buffer alloc;
  /*! \brief The AST node whose body is where the cache stage should be inserted. */
  StmtSRef loc_sref;
  /*! \brief The index to insert the cache_read/cache_write stage. */
  size_t loc_pos;
  /*! \brief The cache_read/cache_write stage to be inserted. */
  Stmt cache_stage;
  /*! \brief The map used for ScheduleStateNode::Replace. */
  Map<Block, Block> block_reuse;
};

/*! \brief Return the buffer region realted with the buffer */
Optional<BufferRegion> GetBufferRegionFromBuffer(const Array<BufferRegion>& buffer_regions,
                                                 const Buffer& buffer) {
  Optional<BufferRegion> res = NullOpt;
  for (const auto& region : buffer_regions) {
    if (region->buffer.same_as(buffer)) {
      ICHECK(!res.defined());
      res = region;
    }
  }
  return res;
}

/*!
 * \brief Create a loop nest that represents cache copy (cache_read / cache_write) from read buffer
 *        to write buffer.
 * \note This function will store the stmt with loop nesting to the CacheStageInfo, but only return
 *        the inside block.
 * \param cache_region The cached copy region.
 * \param info The cache stage information, which will be updated in the function.
 * \param storage_scope The storage scope of the cached buffer (only used in naming here)
 * \returns A block indicating the body of the loop nesting.
 */
Block MakeCacheStage(const BufferRegion& cache_region, CacheStageInfo* info,
                     const String& storage_scope) {
  // loop variables
  std::vector<Var> loop_vars;
  // bindings in block realize
  std::vector<PrimExpr> iter_values;
  // Create loop vars and block vars' binding_value
  for (const Range& axis_range : cache_region->region) {
    Var loop_var("ax" + std::to_string(loop_vars.size()), axis_range->extent.dtype());
    loop_vars.push_back(loop_var);
    iter_values.push_back(axis_range->min + loop_var);
  }
  // block variables
  Array<IterVar> block_vars;
  // block access region for read/write buffers
  Region access_region;
  // indices used in block body
  Array<PrimExpr> access_indices;
  // Create block vars, block's accessed region and accessing indices
  for (const PrimExpr& dim : cache_region->buffer->shape) {
    Var var("v" + std::to_string(access_indices.size()), dim.dtype());
    block_vars.push_back(IterVar(/*dom=*/Range::FromMinExtent(0, dim),
                                 /*var=*/var,
                                 /*IterVarType=*/kDataPar));
    access_indices.push_back(var);
    access_region.push_back(Range::FromMinExtent(var, make_const(var.dtype(), 1)));
  }

  // Create the body block:
  //   reads = [read_buffer[access_region]]
  //   writes = [write_buffer[access_region]]
  //     write_buffer[access_indices] = read_buffer[access_indices]
  Block block(
      /*iter_vars=*/std::move(block_vars),
      /*reads=*/{BufferRegion(info->read_buffer, access_region)},
      /*writes=*/{BufferRegion(info->write_buffer, access_region)},
      /*name_hint=*/cache_region->buffer->name + "_" + storage_scope,
      /*body=*/
      BufferStore(info->write_buffer, BufferLoad(info->read_buffer, access_indices),
                  access_indices),
      /*init=*/NullOpt,
      /*alloc_buffers=*/{},
      /*match_buffers=*/{},
      /*annotations=*/{});
  // Create the block realize node
  Stmt body = BlockRealize(/*values=*/iter_values,
                           /*predicate=*/const_true(),
                           /*block=*/block);
  // Create surrounding loops
  for (size_t i = loop_vars.size(); i >= 1; --i) {
    body = For(/*loop_var=*/loop_vars[i - 1],
               /*min=*/0,
               /*extent=*/cache_region->region[i - 1]->extent,
               /*kind=*/ForKind::kSerial,
               /*body=*/body);
  }
  info->cache_stage = std::move(body);
  return block;
}

Block MakeReIndexStage(const BlockNode* block, CacheStageInfo* info,
                       const std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>& covered,
                       const Array<PrimExpr>& original_indices, int buffer_index,
                       bool is_write_index) {
  // loop variables
  std::vector<Var> loop_vars;
  // bindings in block realize
  std::vector<PrimExpr> iter_values;
  // create loop vars and block vars' binding_value
  for (const IterVar& iter : block->iter_vars) {
    Var loop_var("ax" + std::to_string(loop_vars.size()));
    loop_vars.push_back(loop_var);
    iter_values.push_back(loop_var);
  }
  // block vars
  Array<IterVar> block_vars;
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectEqual> block_var_replace_map;
  // block access region of reindexed buffer and original buffer
  Region reindex_region, original_region;
  // indices used in block body
  Array<PrimExpr> reindex_indices;
  // Create block vars, block's accessed region and accessing indices
  for (const IterVar& iter : block->iter_vars) {
    Var var("v" + std::to_string(block_vars.size()));
    bool flag = covered.count(iter->var);
    block_vars.push_back(IterVar(/*dom=*/flag ? iter->dom : Range::FromMinExtent(0, 1),
                                 /*var=*/var,
                                 /*IterVarType=*/kDataPar));
    if (flag) {
      reindex_indices.push_back(var);
      reindex_region.push_back(Range::FromMinExtent(var, 1));
    }
    block_var_replace_map[iter->var] = var;
  }
  BufferRegion buffer_region =
      is_write_index ? block->writes[buffer_index] : block->reads[buffer_index];
  original_region = Substitute(buffer_region->region, block_var_replace_map);
  Array<PrimExpr> new_indices;
  for (const auto& original_index : original_indices) {
    new_indices.push_back(Substitute(original_index, block_var_replace_map));
  }
  // Create the body block
  Block new_block(
      /*iter_vars=*/block_vars,
      /*reads=*/
      {BufferRegion(info->read_buffer, is_write_index ? reindex_region : original_region)},
      /*writes=*/
      {BufferRegion(info->write_buffer, is_write_index ? original_region : reindex_region)},
      /*name_hint=*/buffer_region->buffer->name + "_reindex",
      /*body=*/
      BufferStore(info->write_buffer,
                  BufferLoad(info->read_buffer, is_write_index ? reindex_indices : new_indices),
                  is_write_index ? new_indices : reindex_indices),
      /*init=*/NullOpt,
      /*alloc_buffers=*/{},
      /*match_buffers=*/{},
      /*annotations=*/{});
  // Create the block realize node
  Stmt body = BlockRealize(/*values=*/iter_values,
                           /*predicate=*/const_true(),
                           /*block=*/new_block);
  // Create surrounding loops
  for (size_t i = loop_vars.size(); i >= 1; --i) {
    body = For(/*loop_var=*/loop_vars[i - 1],
               /*min=*/block_vars[i - 1]->dom->min,
               /*extent=*/block_vars[i - 1]->dom->extent,
               /*kind=*/ForKind::kSerial,
               /*body=*/body);
  }
  info->cache_stage = std::move(body);
  return new_block;
}

/*!
 * \brief Recalculate the `affine_binding` flag of a specifc block
 * \param block_sref The sref to the specific block
 */
bool CalculateAffineFlag(const ScheduleState& self, const StmtSRef& block_sref) {
  if (block_sref->parent == nullptr) {
    return true;
  }
  arith::Analyzer analyzer;
  StmtSRef parent_sref = GetRef<StmtSRef>(block_sref->parent);
  return IsAffineBinding(/*realize=*/GetBlockRealize(self, block_sref),
                         /*loop_var_ranges=*/LoopDomainOfSRefTreePath(parent_sref),
                         /*analyzer=*/&analyzer);
}

/*!
 * \brief Insert the cache_read/cache_write stage into the specific position
 * \param stmt A sequence of statements or a single statement that the new stage is inserted in
 * \param pos The position where the cache stage is inserted
 * \param stage The stage to be inserted
 * \return A SeqStmt, the result after insertion
 */
SeqStmt InsertCacheStage(const Stmt& stmt, int pos, const Stmt& stage) {
  if (const auto* seq_stmt = stmt.as<SeqStmtNode>()) {
    ObjectPtr<SeqStmtNode> result = make_object<SeqStmtNode>(*seq_stmt);
    result->seq.insert(result->seq.begin() + pos, stage);
    return SeqStmt(result);
  }
  if (pos == 0) {
    return SeqStmt({stage, stmt});
  }
  ICHECK_EQ(pos, 1);
  return SeqStmt({stmt, stage});
}

/*!
 * \brief Get the only writer block of the input buffer in a given scope block.
 * \param self The state of the schedule
 * \param scope_sref The scope block where the write is considered
 * \param buffer The queried buffer
 * \return The sref of the only writer of the input buffer in the given scope,
 *         or `NullOpt` if no block writes it in the scope.
 * \throw NotSingleWriteBlock if there are more than one intrested block.
 */
Optional<StmtSRef> GetOnlyWriteBlock(ScheduleState self, const StmtSRef& scope_sref,
                                     const Buffer& buffer) {
  BlockScope scope = self->GetBlockScope(scope_sref);
  auto it = scope->buffer_writers.find(buffer);
  if (it == scope->buffer_writers.end()) {
    return NullOpt;
  } else {
    const Array<StmtSRef>& block_srefs = it->second;
    ICHECK(!block_srefs.empty());
    if (block_srefs.size() > 1) {
      throw NotSingleWriteBlock(self->mod, buffer, block_srefs);
    }
    return block_srefs[0];
  }
}

/*!
 * \brief Get the buffer region under the sref tree path [dom_low_inclusive, dom_high_exclusive)
 * \param self The state of the schedule.
 * \param buffer_region The buffer region to be analyzed.
 * \param block_sref The sref of the block related to the region.
 * \param dom_low_inclusive The lowest node in the sref tree path.
 * \param dom_high_exclusive The highest node in the sref tree path.
 * \return The relaxed buffer region.
 */
BufferRegion RelaxBufferRegion(ScheduleState self, const BufferRegion& buffer_region,
                               const StmtSRef& block_sref, const StmtSRef& dom_low_inclusive,
                               const StmtSRef& dom_high_exclusive) {
  BlockRealize realize = GetBlockRealize(self, block_sref);
  Map<Var, PrimExpr> binding = GetBindings(realize);
  const Buffer& buffer = buffer_region->buffer;
  arith::Analyzer analyzer;
  BufferRegion subst_region = BufferRegion(buffer, Substitute(buffer_region->region, binding));
  Array<arith::IntSet> int_sets = AnalyzeRegionUpperBound(
      /*region=*/subst_region,
      /*predicate=*/realize->predicate,
      /*dom_low_inclusive=*/dom_low_inclusive,
      /*dom_high_exclusive=*/dom_high_exclusive,
      /*analyzer=*/&analyzer);
  ICHECK_EQ(buffer_region->region.size(), int_sets.size());

  Region region;
  region.reserve(int_sets.size());
  for (size_t i = 0; i < int_sets.size(); ++i) {
    region.push_back(int_sets[i].CoverRange(Range::FromMinExtent(0, buffer->shape[i])));
  }
  return BufferRegion(buffer, region);
}

/*! \brief Detect the insertion position of the new cache stage */
class CacheLocDetector : public StmtVisitor {
 public:
  /*!
   * \brief Detect the insertion position of the cache stage, and write the position into the
   * CacheStageInfo
   * \param self The state of the schedule
   * \param block_sref The sref of the unique writer block of the buffer being applied cache_read or
   * cache_write \param scope_sref The sref of the scope block of the cached block \param info The
   * cache stage info.
   */
  static void Detect(const ScheduleState& self, const StmtSRef& block_sref,
                     const StmtSRef& scope_sref, CacheStageInfo* info) {
    std::vector<StmtSRef> related_blocks;
    for (const Dependency& def : self->GetBlockScope(scope_sref)->GetDepsBySrc(block_sref)) {
      if (def->kind == DepKind::kRAW) {
        related_blocks.push_back(def->dst);
      }
    }
    if (!related_blocks.empty()) {
      CacheLocDetector detector(self, block_sref, scope_sref, related_blocks);
      detector(GetRef<Stmt>(scope_sref->stmt));
      info->loc_sref = detector.loc_sref_;
      info->loc_pos = detector.loc_pos_;
    } else {
      info->loc_sref = scope_sref;
      const auto* body = scope_sref->StmtAs<BlockNode>()->body.as<SeqStmtNode>();
      info->loc_pos = body == nullptr ? 1 : body->size();
    }
  }

 private:
  /*!
   * \brief Constructor
   * \param self The state of the schedule
   * \param block_sref The sref of the unique writer block of the buffer being applied cache_read or
   * cache_write
   * \param scope_sref The sref of the scope block of the cached block
   * \param related_blocks Producer blocks for cache_write, or consumer blocks for cache_read
   */
  CacheLocDetector(const ScheduleState self, const StmtSRef& block_sref, const StmtSRef& scope_sref,
                   const std::vector<StmtSRef>& related_blocks)
      : self_(self),
        block_sref_(block_sref),
        scope_sref_(scope_sref),
        related_blocks_(related_blocks) {}

  void VisitStmt_(const SeqStmtNode* seq_stmt) final {
    bool previous_visited_block = visited_block_;
    bool previous_visited_related = visited_related_;
    visited_block_ = visited_related_ = false;

    int pos = -1;
    for (size_t i = 0; i < seq_stmt->size(); ++i) {
      if (loc_pos_ != -1) {
        break;
      }
      VisitStmt(seq_stmt->seq[i]);
      // `pos` can be assigned only once when we visited `block_sref`
      if (visited_block_ && visited_related_ && pos == -1) {
        // The offset of insert position from the block
        pos = i;
      }
    }
    visited_block_ = visited_block_ || previous_visited_block;
    visited_related_ = visited_related_ || previous_visited_related;
    // Only we visited the writing block and any one of the related blocks
    // That means that we have found the lowest ancestor
    // of the block and any one of the related ones
    if (visited_block_ && visited_related_ && loc_pos_ == -1) {
      loc_pos_ = pos;
    }
  }

  void VisitStmt_(const BlockNode* block) final {
    // Only visit the current scope under buffer writer's parent block
    if (block == scope_sref_->stmt) {
      // The block vistied is the current parent scope
      StmtVisitor::VisitStmt_(block);
      // Handling cache_read for input buffer
      if (visited_block_ && visited_related_ && !loc_sref_.defined()) {
        loc_sref_ = self_->stmt2ref.at(block);
        if (loc_pos_ == -1) {
          loc_pos_ = 1;
        }
      }
      return;
    }
    // Update `visited_block`
    if (block_sref_->stmt == block) {
      visited_block_ = true;
      return;
    }
    // Update `visited_related`
    for (const StmtSRef& related_block : related_blocks_) {
      if (related_block->stmt == block) {
        visited_related_ = true;
        return;
      }
    }
  }

  void VisitStmt_(const ForNode* loop) final {
    StmtVisitor::VisitStmt_(loop);
    if (visited_block_ && visited_related_ && !loc_sref_.defined() && loc_pos_ != -1) {
      loc_sref_ = self_->stmt2ref.at(loop);
    }
  }

 private:
  /*! \brief The schedule class */
  const ScheduleState self_;
  /*! \brief The dominate block which write the buffer */
  const StmtSRef& block_sref_;
  /*! \brief The parent scope of the dominate block */
  const StmtSRef& scope_sref_;
  /*! \brief Producer blocks for cache_write and consumer blocks for cache_read */
  const std::vector<StmtSRef>& related_blocks_;
  /*! \brief The flag whether we have visited the dominate block */
  bool visited_block_{false};
  /*! \brief The flag whether we have visited at least one related blocks */
  bool visited_related_{false};
  /*! \brief The AST node whose body is where the cache stage should be inserted */
  StmtSRef loc_sref_{nullptr};
  /*! \brief The index to insert the cache_read/cache_write stage */
  int loc_pos_{-1};
};

/*! \brief Mutator for CacheRead. */
class CacheReadRewriter : public StmtExprMutator {
 public:
  /*!
   * \brief Rewrite the AST and add a cache_read stage with the information provided
   * \param scope_sref The parent scope of this mutation
   * \param info The cache stage information
   * \return The new AST rooting at the original parent scope
   */
  static Stmt Rewrite(const StmtSRef& scope_sref, CacheStageInfo* info) {
    CacheReadRewriter rewriter(scope_sref, info);
    return rewriter(GetRef<Stmt>(scope_sref->stmt));
  }

 private:
  explicit CacheReadRewriter(const StmtSRef& scope_sref, CacheStageInfo* info)
      : scope_sref_(scope_sref), info_(info) {}

  Stmt VisitStmt_(const ForNode* loop) final {
    Stmt stmt = StmtMutator::VisitStmt_(loop);
    // Check the insertion point
    if (loop == info_->loc_sref->stmt) {
      // Insert cache stage into the loop if it is the right place
      ObjectPtr<ForNode> n = make_object<ForNode>(*stmt.as<ForNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Stmt(n);
    }
    return stmt;
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    Block old_stmt = GetRef<Block>(block);
    // We don't mutate the block which generates info->read_buffer
    if (block != scope_sref_->stmt &&
        GetBufferRegionFromBuffer(block->writes, info_->read_buffer).defined()) {
      return std::move(old_stmt);
    }
    // Mutate the body
    Block stmt = Downcast<Block>(StmtMutator::VisitStmt_(block));
    // Check the insertion point
    if (block == info_->loc_sref->stmt) {
      // Insert cache stage into the block if it is the right place
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Block(n);
    }
    // Check if it is the block corresponding to the parent scope
    if (block == scope_sref_->stmt) {
      // If so, put buffer allocation on the parent scope
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->alloc_buffers.push_back(info_->alloc);
      stmt = Block(n);
    } else {
      // Otherwise, update read regions and match_buffers
      Array<BufferRegion> reads =
          ReplaceBuffer(block->reads, info_->read_buffer, info_->write_buffer);
      Array<MatchBufferRegion> match_buffers =
          ReplaceBuffer(block->match_buffers, info_->read_buffer, info_->write_buffer);
      if (!reads.same_as(block->reads) || !match_buffers.same_as(block->match_buffers)) {
        ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
        n->reads = std::move(reads);
        n->match_buffers = std::move(match_buffers);
        stmt = Block(n);
      }
    }
    info_->block_reuse.Set(old_stmt, stmt);
    return std::move(stmt);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* load) final {
    if (load->buffer.same_as(info_->read_buffer)) {
      ObjectPtr<BufferLoadNode> n = make_object<BufferLoadNode>(*load);
      n->buffer = info_->write_buffer;
      return PrimExpr(n);
    }
    return ExprMutator::VisitExpr_(load);
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated LoadNode.  Please use BufferLoadNode instead.";
    return PrimExpr();
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    if (op == info_->read_buffer->data.get()) {
      return info_->write_buffer->data;
    }
    return GetRef<PrimExpr>(op);
  }

 private:
  /*! \brief The parent scope of the insertion */
  const StmtSRef& scope_sref_;
  /*! \brief The info for inserting cache stage */
  CacheStageInfo* info_;
};

/*! \brief Mutator for CacheWrite */
class CacheWriteRewriter : public StmtExprMutator {
 public:
  /*!
   * \brief Rewrite the AST and add a cache_write stage with the information provided.
   * \param scope_sref The parent scope of this mutation.
   * \param writer_block_sref The only writer block in the scope.
   * \param info The cache stage information.
   * \return The new AST rooting at the original parent scope.
   */
  static Stmt Rewrite(const StmtSRef& scope_sref, const StmtSRef& writer_block_sref,
                      CacheStageInfo* info) {
    CacheWriteRewriter rewriter(scope_sref, writer_block_sref, info);
    return rewriter(GetRef<Stmt>(scope_sref->stmt));
  }

 private:
  explicit CacheWriteRewriter(const StmtSRef& scope_sref, const StmtSRef& writer_block_sref,
                              CacheStageInfo* info)
      : scope_sref_(scope_sref), writer_block_sref_(writer_block_sref), info_(info) {}

  Stmt VisitStmt_(const ForNode* loop) final {
    Stmt stmt = StmtMutator::VisitStmt_(loop);
    // Check the insertion point
    if (loop == info_->loc_sref->stmt) {
      // Insert cache stage into the loop if it is the right place
      ObjectPtr<ForNode> n = make_object<ForNode>(*stmt.as<ForNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Stmt(n);
    }
    return stmt;
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    Block old_stmt = GetRef<Block>(block);
    // We only mutate the block which generates info->write_buffer
    if (block != writer_block_sref_->stmt && block != scope_sref_->stmt && !under_writer_block_) {
      return std::move(old_stmt);
    }

    // Mutate the body
    bool under_scope = under_writer_block_ || block == writer_block_sref_->stmt;
    std::swap(under_scope, under_writer_block_);
    Block stmt = Downcast<Block>(StmtMutator::VisitStmt_(block));
    std::swap(under_scope, under_writer_block_);

    // Find the insertion point
    if (block == info_->loc_sref->stmt) {
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Block(n);
    }
    // Put buffer allocation on the parent scope
    if (block == scope_sref_->stmt) {
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->alloc_buffers.push_back(info_->alloc);
      stmt = Block(n);
    } else {
      // Since cache_write changes the block, we need to update the buffer it writes
      auto writes = ReplaceBuffer(block->writes, info_->write_buffer, info_->read_buffer);
      auto reads = ReplaceBuffer(block->reads, info_->write_buffer, info_->read_buffer);
      auto match_buffers =
          ReplaceBuffer(block->match_buffers, info_->write_buffer, info_->read_buffer);
      if (!writes.same_as(block->writes) || !reads.same_as(block->reads) ||
          !match_buffers.same_as(block->match_buffers)) {
        ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
        n->writes = std::move(writes);
        n->reads = std::move(reads);
        n->match_buffers = std::move(match_buffers);
        stmt = Block(n);
      }
    }
    info_->block_reuse.Set(old_stmt, stmt);
    return std::move(stmt);
  }

  Stmt VisitStmt_(const BufferStoreNode* store) final {
    BufferStore stmt = Downcast<BufferStore>(StmtMutator::VisitStmt_(store));
    if (stmt->buffer.same_as(info_->write_buffer)) {
      auto n = CopyOnWrite(stmt.get());
      n->buffer = info_->read_buffer;
      return Stmt(n);
    } else {
      return std::move(stmt);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* load) final {
    if (load->buffer.same_as(info_->write_buffer)) {
      ObjectPtr<BufferLoadNode> n = make_object<BufferLoadNode>(*load);
      n->buffer = info_->read_buffer;
      return PrimExpr(n);
    }
    return ExprMutator::VisitExpr_(load);
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated LoadNode.  Please use BufferLoadNode instead.";
    return PrimExpr();
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated StoreNode.  Please use BufferStoreNode instead.";
    return Stmt();
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    if (op == info_->write_buffer->data.get()) {
      return info_->read_buffer->data;
    }
    return GetRef<PrimExpr>(op);
  }

 private:
  /*! \brief The parent scope of the insertion. */
  const StmtSRef& scope_sref_;
  /*! \brief The parent scope of the insertion. */
  const StmtSRef& writer_block_sref_;
  /*! \brief The info for inserting cache stage. */
  CacheStageInfo* info_;
  /*! \brief Whether the current node is under the given block. */
  bool under_writer_block_{false};
};

/*! \brief Collect the related Load/Store to reindex */
class ReIndexCollector : public StmtExprVisitor {
 public:
  static Array<PrimExpr> Collect(const Buffer& buffer, const Stmt& body) {
    ReIndexCollector collector(buffer);
    collector(body);
    ICHECK(collector.indices.defined());
    std::vector<PrimExpr> result;
    for (const PrimExpr& index : collector.indices.value()) {
      result.push_back(collector.analyzer_.Simplify(index));
    }
    return result;
  }

 private:
  explicit ReIndexCollector(const Buffer& buffer) : buffer_(buffer) {}

  bool EqualIndices(const Array<PrimExpr>& lhs, const Array<PrimExpr>& rhs) {
    ICHECK_EQ(lhs.size(), rhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (!analyzer_.CanProveEqual(lhs[i], rhs[i])) return false;
    }
    return true;
  }

  void VisitExpr_(const BufferLoadNode* load) final {
    StmtExprVisitor::VisitExpr_(load);
    if (load->buffer.same_as(buffer_)) {
      if (!indices.defined()) {
        indices = load->indices;
      } else {
        ICHECK(EqualIndices(indices.value(), load->indices));
      }
    }
  }

  void VisitStmt_(const BlockNode* block) final {
    // no sub-blocks under this block
    ICHECK(!scope_);
    scope_ = true;
    for (const IterVar& iter : block->iter_vars) {
      analyzer_.Bind(iter->var, iter->dom);
    }
    StmtExprVisitor::VisitStmt_(block);
  }

  void VisitStmt_(const BufferStoreNode* store) final {
    StmtExprVisitor::VisitStmt_(store);
    if (store->buffer.same_as(buffer_)) {
      if (!indices.defined()) {
        indices = store->indices;
      } else {
        ICHECK(EqualIndices(indices.value(), store->indices));
      }
    }
  }

  void VisitExpr_(const VarNode* var) final { ICHECK(var != buffer_->data.get()); }

  /*! \brief Indicate if we are in the scope block */
  bool scope_{false};
  /*! \brief The arith ananlyzer */
  arith::Analyzer analyzer_;
  /*! \brief The buffer to rewrite */
  Buffer buffer_;
  /*! \brief The indices of buffer acess to rewrite */
  Optional<Array<PrimExpr>> indices{NullOpt};
};

/*! \brief Mutator of ReIndex */
class ReIndexRewriter : public StmtExprMutator {
 public:
  static Stmt Rewrite(const StmtSRef& scope_sref, const StmtSRef& block_sref, CacheStageInfo* info,
                      const std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>& covered) {
    ReIndexRewriter rewriter(scope_sref, block_sref, info, covered);
    return rewriter(GetRef<Stmt>(scope_sref->stmt));
  }

 private:
  explicit ReIndexRewriter(const StmtSRef& scope_sref, const StmtSRef& block_sref,
                           CacheStageInfo* info,
                           const std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>& covered)
      : scope_sref_(scope_sref), block_sref_(block_sref), info_(info), covered_(covered) {
    new_buffer_ = info->alloc;
    old_buffer_ = info->read_buffer.same_as(new_buffer_) ? info->write_buffer : info->read_buffer;
  }

  Array<BufferRegion> ReplaceBuffer(Array<BufferRegion> regions, const Buffer& source,
                                    const Buffer& target, const Region& new_region) {
    regions.MutateByApply([&source, &target, &new_region](BufferRegion region) -> BufferRegion {
      if (region->buffer.same_as(source)) {
        ObjectPtr<BufferRegionNode> n = make_object<BufferRegionNode>(*region.get());
        n->buffer = target;
        n->region = new_region;
        return BufferRegion(n);
      }
      return region;
    });
    return regions;
  }

  Array<MatchBufferRegion> ReplaceBuffer(Array<MatchBufferRegion> match_buffers,
                                         const Buffer& source, const Buffer& target,
                                         const Region& new_region) {
    match_buffers.MutateByApply(
        [&source, &target, &new_region](MatchBufferRegion match_buffer) -> MatchBufferRegion {
          if (match_buffer->source->buffer.same_as(source)) {
            ObjectPtr<MatchBufferRegionNode> n =
                make_object<MatchBufferRegionNode>(*match_buffer.get());
            n->source = BufferRegion(target, new_region);
            return MatchBufferRegion(n);
          }
          return match_buffer;
        });
    return match_buffers;
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    Block old_stmt = GetRef<Block>(block);
    if (is_scope_) {
      is_scope_ = false;
      Block stmt = Downcast<Block>(StmtExprMutator::VisitStmt_(block));
      // Insert cache stage into the loop
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      n->alloc_buffers.push_back(info_->alloc);
      stmt = Block(n);
      info_->block_reuse.Set(old_stmt, stmt);
      return stmt;
    }
    if (block == block_sref_->stmt) {
      // Collect the updated indices and regions
      for (const IterVar& iter : block->iter_vars) {
        if (covered_.count(iter->var)) {
          indices_.push_back(iter->var);
          region_.push_back(Range::FromMinExtent(iter->var, 1));
        }
      }
      Block stmt = Downcast<Block>(StmtExprMutator::VisitStmt_(block));
      // Update block reads/writes
      auto writes = ReplaceBuffer(block->writes, old_buffer_, new_buffer_, region_);
      auto reads = ReplaceBuffer(block->reads, old_buffer_, new_buffer_, region_);
      auto match_buffers = ReplaceBuffer(block->match_buffers, old_buffer_, new_buffer_, region_);
      if (!writes.same_as(block->writes) || !reads.same_as(block->reads) ||
          !match_buffers.same_as(block->match_buffers)) {
        ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
        n->writes = std::move(writes);
        n->reads = std::move(reads);
        n->match_buffers = std::move(match_buffers);
        stmt = Block(n);
      }
      info_->block_reuse.Set(old_stmt, stmt);
      return stmt;
    }
    return old_stmt;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    if (op->buffer.same_as(old_buffer_)) {
      ObjectPtr<BufferStoreNode> n = make_object<BufferStoreNode>(*stmt.as<BufferStoreNode>());
      n->buffer = new_buffer_;
      n->indices = indices_;
      stmt = BufferStore(n);
    }
    return stmt;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    if (op->buffer.same_as(old_buffer_)) {
      ObjectPtr<BufferLoadNode> n = make_object<BufferLoadNode>(*expr.as<BufferLoadNode>());
      n->buffer = new_buffer_;
      n->indices = indices_;
      expr = BufferLoad(n);
    }
    return expr;
  }

 private:
  /*! \brief The parent scope of the insertion. */
  const StmtSRef& scope_sref_;
  /*! \brief The parent scope of the insertion. */
  const StmtSRef& block_sref_;
  /*! \brief The info for inserting reindex stage. */
  CacheStageInfo* info_;
  /*! \brief Whether old block var is covered in the indices */
  const std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>& covered_;
  /*! \brief Whether the current block is scope block */
  bool is_scope_{true};
  /*! \brief The  buffer to be replaced */
  Buffer old_buffer_;
  /*! \brief The reindex buffer */
  Buffer new_buffer_;
  /*! \brief The new indices */
  Array<PrimExpr> indices_;
  /*! \brief The new region */
  Region region_;
};

/******** Implementation ********/

StmtSRef CacheRead(ScheduleState self, const StmtSRef& block_sref, int read_buffer_index,
                   const String& storage_scope) {
  /*!
   * Check:
   *   - The index is in the array of block reading region
   *   - There is at most one block who write the buffer in the scope
   *
   * Mutate:
   *   - Allocate new cache buffer under the current scope.
   *   - Find the lowest ancestor of the block and ANY ONE of the consumers blocks.
   *   - Copy the buffer with the consumed region.
   */

  // Step 0. Check the input storage scope.
  CheckStorageScope(self, storage_scope);

  // Step 1. Check index, getting the target buffer and the parent scope
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  Buffer read_buffer =
      GetNthAccessBuffer(self, GetRef<Block>(block), read_buffer_index, /*is_write=*/false);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/true);
  const BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_block, scope_sref);

  // Step 2. Create CacheStageInfo
  CacheStageInfo info;
  info.read_buffer = read_buffer;
  // Create the corresponding buffer to be written, i.e. result of cache_read
  info.write_buffer = WithScope(read_buffer, storage_scope);
  // Create the corresponding buffer allocation
  info.alloc = info.write_buffer;

  // Step 3. Update cache stage info.
  BufferRegion cache_region{nullptr};
  if (Optional<StmtSRef> _write_block_sref = GetOnlyWriteBlock(self, scope_sref, read_buffer)) {
    // Case 1. The buffer is written inside the block.
    StmtSRef write_block_sref = _write_block_sref.value();
    const BlockNode* write_block = TVM_SREF_TO_BLOCK(write_block, write_block_sref);
    // Find the producing region
    BufferRegion region = GetBufferRegionFromBuffer(write_block->writes, read_buffer).value();
    StmtSRef parent_sref = GetRef<StmtSRef>(write_block_sref->parent);

    // Detect insert position
    CacheLocDetector::Detect(self, write_block_sref, scope_sref, &info);
    cache_region = RelaxBufferRegion(self, region, write_block_sref, parent_sref, info.loc_sref);
  } else {
    // Case 2. The buffer is the input block for the scope.
    info.loc_sref = scope_sref;
    info.loc_pos = 0;
    if (Optional<BufferRegion> region =
            GetBufferRegionFromBuffer(scope_block->reads, read_buffer)) {
      cache_region = region.value();
    } else {
      cache_region = BufferRegion::FullRegion(read_buffer);
    }
  }

  // Step 4. Making new cache stage block and rewrite readers.
  Block cache_read_stage = MakeCacheStage(/*cache_region=*/cache_region, /*info=*/&info,
                                          /*storage_scope=*/storage_scope);
  Stmt new_scope = CacheReadRewriter::Rewrite(/*scope_sref=*/scope_sref, /*info=*/&info);

  // Step 5. Replacing and updating flags.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_read_stage.get());
  BlockInfo& block_info = self->block_info[result_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info.region_cover = true;
  block_info.scope->stage_pipeline = true;
  return result_block_sref;
}

StmtSRef CacheWrite(ScheduleState self, const StmtSRef& block_sref, int write_buffer_index,
                    const String& storage_scope) {
  /*!
   * Check:
   *   - The index is in the array of block reading region
   *   - There is only one block who write the buffer in the scope
   *
   * Mutate:
   *   - Allocate new cache buffer under the current scope.
   *   - Find the lowest ancestor of the block and ANY ONE of the producer blocks.
   *   - Copy the buffer with the consumed region.
   */

  // Step 0. Check the input storage scope.
  CheckStorageScope(self, storage_scope);

  // Step 1. Checking index, getting the target buffer and the parent scope
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  Buffer write_buffer =
      GetNthAccessBuffer(self, GetRef<Block>(block), write_buffer_index, /*is_write=*/true);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/true);

  // Step 2. Creating CacheStageInfo
  CacheStageInfo info;
  info.read_buffer = WithScope(write_buffer, storage_scope);
  // Create the corresponding buffer to be written, i.e. result of cache_write
  info.write_buffer = write_buffer;
  // Create the corresponding buffer allocation
  info.alloc = info.read_buffer;

  // Step 3. Check the only writer block.
  ICHECK_EQ(block_sref.get(), GetOnlyWriteBlock(self, scope_sref, write_buffer).get());

  // Step 4. Find the producing region and insert position
  BufferRegion region = GetBufferRegionFromBuffer(block->writes, write_buffer).value();
  StmtSRef parent_sref = GetRef<StmtSRef>(block_sref->parent);
  // Detect insert position
  CacheLocDetector::Detect(self, block_sref, scope_sref, &info);
  BufferRegion cache_region =
      RelaxBufferRegion(self, region, block_sref, parent_sref, info.loc_sref);

  // Step 5. Making new cache stage block and rewrite readers.
  Block cache_write_stage = MakeCacheStage(/*cache_region=*/cache_region, /*info=*/&info,
                                           /*storage_scope=*/storage_scope);
  Stmt new_scope = CacheWriteRewriter::Rewrite(/*scope_sref=*/scope_sref,
                                               /*writer_block_sref=*/block_sref, /*info=*/&info);

  // Step 6. Replacing and updating flags.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_write_stage.get());
  BlockInfo& block_info = self->block_info[result_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info.region_cover = true;
  block_info.scope->stage_pipeline = true;
  return result_block_sref;
}

StmtSRef ReIndex(ScheduleState self, const StmtSRef& block_sref, int buffer_index,
                 bool is_write_index) {
  // TODO: handle the case where buffer occurs both in read & write
  // Step 0. Checking index, getting the target buffer and the parent scope
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  Buffer buffer =
      GetNthAccessBuffer(self, GetRef<Block>(block), buffer_index, /*is_write=*/is_write_index);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/true);

  // Step 1. Collect the original indices and check there's only single pattern of related
  // Load/Store and the buffer is not accessed opaquely
  Array<PrimExpr> original_indices = ReIndexCollector::Collect(buffer, GetRef<Block>(block));

  // collect info about block vars appearing in the original_indices
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> covered;
  for (const PrimExpr& index : original_indices) {
    PreOrderVisit(index, [&](const ObjectRef& obj) -> bool {
      if (const VarNode* var = obj.as<VarNode>()) {
        covered.insert(GetRef<Var>(var));
      }
      return true;
    });
  }

  // Step 2. Creating CacheStageInfo
  CacheStageInfo info;
  // Create the corresponding buffer to be read(write), i.e. the result of reindex read(write)
  if (is_write_index) {
    info.read_buffer = WithBlockIters(buffer, block->iter_vars, covered);
    info.write_buffer = buffer;
    info.alloc = info.read_buffer;
  } else {
    info.read_buffer = buffer;
    info.write_buffer = WithBlockIters(buffer, block->iter_vars, covered);
    info.alloc = info.write_buffer;
  }

  // Step 3. Check the block belongs to a chain loop nesting under the scope,
  //         and get the insert location
  const StmtSRefNode* loop;
  for (loop = block_sref->parent; loop->parent != scope_sref.get();) {
    const ForNode* outer = loop->parent->StmtAs<ForNode>();
    const ForNode* inner = loop->StmtAs<ForNode>();
    ICHECK(outer != nullptr && inner != nullptr);
    ICHECK(outer->body.get() == inner);
    loop = loop->parent;
  }
  info.loc_pos = loop->seq_index == -1 ? 0 : loop->seq_index;
  if (is_write_index) {
    info.loc_pos++;
  }

  // Step 4. Making new reindex stage block and rewrite
  Block reindex_stage =
      MakeReIndexStage(block, &info, covered, original_indices, buffer_index, is_write_index);
  Stmt new_scope = ReIndexRewriter::Rewrite(scope_sref, block_sref, &info, covered);

  // Step 5. Replacing and updating flags
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(reindex_stage.get());
  BlockInfo& block_info = self->block_info[result_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info.region_cover = true;
  block_info.scope->stage_pipeline = true;
  return result_block_sref;
}

/******** Instruction Registration ********/

struct CacheReadTraits : public UnpackedInstTraits<CacheReadTraits> {
  static constexpr const char* kName = "CacheRead";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, BlockRV block, Integer read_buffer_index,
                                         String storage_scope) {
    return sch->CacheRead(block, read_buffer_index->value, storage_scope);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, Integer read_buffer_index,
                                 String storage_scope) {
    PythonAPICall py("cache_read");
    py.Input("block", block);
    py.Input("read_buffer_index", read_buffer_index->value);
    py.Input("storage_scope", storage_scope);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct CacheWriteTraits : public UnpackedInstTraits<CacheWriteTraits> {
  static constexpr const char* kName = "CacheWrite";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, BlockRV block, Integer write_buffer_index,
                                         String storage_scope) {
    return sch->CacheWrite(block, write_buffer_index->value, storage_scope);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, Integer write_buffer_index,
                                 String storage_scope) {
    PythonAPICall py("cache_write");
    py.Input("block", block);
    py.Input("write_buffer_index", write_buffer_index->value);
    py.Input("storage_scope", storage_scope);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct ReIndexTraits : public UnpackedInstTraits<ReIndexTraits> {
  static constexpr const char* kName = "ReIndex";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, BlockRV block, Integer buffer_index,
                                         Bool is_write_index) {
    return sch->ReIndex(block, buffer_index, is_write_index);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, Integer buffer_index,
                                 Bool is_write_index) {
    PythonAPICall py("reindex");
    py.Input("block", block);
    py.Input("buffer_index", buffer_index);
    py.Input("is_write_index", is_write_index.operator bool());
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(CacheReadTraits);
TVM_REGISTER_INST_KIND_TRAITS(CacheWriteTraits);
TVM_REGISTER_INST_KIND_TRAITS(ReIndexTraits);

}  // namespace tir
}  // namespace tvm
