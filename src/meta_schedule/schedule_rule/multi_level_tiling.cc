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
#include "./multi_level_tiling.h"

#include <tvm/meta_schedule/schedule_rule.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "../utils.h"
#include "analysis.h"

namespace tvm {
namespace meta_schedule {

using tir::BlockRV;
using tir::IterVarType;
using tir::LoopRV;
using tir::Schedule;

// Do nothing; Inherited from ScheduleRuleNode
void MultiLevelTilingNode::InitializeWithTuneContext(const TuneContext& context) {
  if (Optional<Integer> v = context->target.value()->GetAttr<Integer>("max_threads_per_block")) {
    this->max_threads_per_block_ = v.value()->value;
    if (Optional<Integer> v = context->target.value()->GetAttr<Integer>("thread_warp_size")) {
      this->thread_warp_size_ = v.value()->value;
    } else {
      LOG(INFO) << "'thread_warp_size' is not defined in the target";
    }
  }
}

// Entry of the mega rule; Inherited from ScheduleRuleNode
Array<Schedule> MultiLevelTilingNode::Apply(const Schedule& sch, const BlockRV& block_rv) {
  if (!NeedsMultiLevelTiling(sch->state(), sch->GetSRef(block_rv))) {
    return {sch};
  }
  sch->Annotate(block_rv, tir::attr::meta_schedule_tiling_structure, structure);

  Array<Schedule> results;
  for (auto&& state : ApplySubRules({State(sch, block_rv)})) {
    results.push_back(std::move(state.sch));
  }
  return results;
}

std::vector<State> MultiLevelTilingNode::ApplySubRules(std::vector<State> states) {
  states = SubRule(std::move(states), [&](State state) { return DetectTensorCore(state); });
  states = SubRule(std::move(states), [&](State state) { return AddWriteReuse(state); });
  states = SubRule(std::move(states), [&](State state) { return TileLoopNest(state); });
  states = SubRule(std::move(states), [&](State state) { return AddReadReuse(state); });
  states = SubRule(std::move(states), [&](State state) { return FuseWriteReuse(state); });
  return states;
}

std::vector<State> MultiLevelTilingNode::AddWriteReuse(State state) const {
  const ReuseConfig& config = this->reuse_write_;
  if (config.req == ReuseType::kNoReuse) {
    if (state.tensor_core_is_used) state = TensorCoreStore(state);
    return {std::move(state)};
  }
  std::vector<int> levels = config.levels;
  ReuseType req = config.req;
  if (Optional<Array<Integer>> ann = tir::GetAnn<Array<Integer>>(
          state.sch->GetSRef(state.block_rv), "meta_schedule.write_cache_level")) {
    req = ReuseType::kMustReuse;
    levels = std::vector<int>(ann.value().begin(), ann.value().end());
  }
  std::vector<State> results;
  if (req == ReuseType::kMayReuse) {
    // Case 1. If the write cache is already there, we don't need to add another.
    Array<BlockRV> consumer_rvs = state.sch->GetConsumers(state.block_rv);
    if (consumer_rvs.size() == 1 && IsWriteCache(state.sch->GetSRef(consumer_rvs[0]))) {
      for (int level : levels) {
        State new_state = state;
        new_state.sch = state.sch->Copy();
        new_state.sch->Seed(state.sch->ForkSeed());
        const LoopRV& loop_rv = new_state.tiles[level - 1].back();
        new_state.sch->ReverseComputeAt(consumer_rvs[0], loop_rv, true);
        results.push_back(std::move(new_state));
      }
      results.push_back(state);
      return results;
    } else {
      // Case 2. No write cache is added
      State new_state(/*sch=*/state.sch->Copy(), /*block_rv=*/state.block_rv);
      new_state.sch->Seed(state.sch->ForkSeed());
      results.emplace_back(std::move(new_state));
    }
  }
  //
  // Case 3. Add one write cache
  BlockRV write_cache = state.sch->CacheWrite(/*block_rv=*/state.block_rv, /*read_buffer_index=*/0,
                                              /*storage_scope=*/config.scope);
  for (int level : levels) {
    State new_state = state;
    new_state.sch = state.sch->Copy();
    new_state.sch->Seed(state.sch->ForkSeed());
    const LoopRV& loop_rv = new_state.tiles[level - 1].back();
    new_state.sch->ReverseComputeAt(write_cache, loop_rv, true);
    results.push_back(std::move(new_state));
  }
  state.write_cache_is_added = true;
  if (state.tensor_core_is_used) state = TensorCoreStore(state);
  results.emplace_back(std::move(state));
  return results;
}

std::vector<State> MultiLevelTilingNode::TileLoopNest(State state) const {
  Schedule& sch = state.sch;
  const BlockRV& block_rv = state.block_rv;
  // Step 1. Assuming trivial binding, pair the loops and their iter-var-types
  Array<LoopRV> loops = sch->GetLoops(block_rv);
  std::vector<IterVarType> iter_types = GetBlockVarTypes(sch->GetSRef(state.block_rv));
  ICHECK_EQ(loops.size(), iter_types.size());
  // Step 2. For each loop axis, tile it
  int64_t spatial_loop_product = 1;
  std::vector<Array<LoopRV>> tiles(s_indices_.size() + r_indices_.size());
  for (int i = 0, n = loops.size(); i < n; ++i) {
    LoopRV loop = loops[i];
    const std::vector<int>* idx = nullptr;
    if (iter_types[i] == IterVarType::kDataPar) {
      idx = &s_indices_;
      if (spatial_loop_product != -1) {
        if (const int64_t* extent = tir::GetLoopIntExtent(sch->Get(loop).get())) {
          spatial_loop_product *= *extent;
        } else {
          spatial_loop_product = -1;
        }
      }
    } else if (iter_types[i] == IterVarType::kCommReduce) {
      idx = &r_indices_;
    } else {
      continue;
    }
    // Do the split
    int n_tiles = idx->size();
    Array<tir::ExprRV> factors = sch->SamplePerfectTile(
        /*loop=*/loop,
        /*n=*/n_tiles,
        /*max_innermost_factor=*/max_innermost_factor);
    Array<tir::LoopRV> splits = sch->Split(/*loop=*/loop,
                                           /*factors=*/{factors.begin(), factors.end()});
    // Put every tile to its slot
    for (int j = 0; j < n_tiles; ++j) {
      tiles[idx->at(j)].push_back(splits[j]);
    }
  }
  // Step 3. Reorder to organize the tiles
  sch->Reorder(support::ConcatArrayList<LoopRV>(tiles.begin(), tiles.end()));
  // Step 4. Bind the tiles to threads
  int n_binds = std::min(tile_binds.size(), tiles.size());
  for (int i = 0; i < n_binds; ++i) {
    LoopRV fused = sch->Fuse(tiles[i]);
    sch->Bind(fused, tile_binds[i]);
    tiles[i] = {fused};
  }
  state.tiles = Array<Array<LoopRV>>{tiles.begin(), tiles.end()};
  if (this->thread_warp_size_ != -1) {
    int64_t low_inclusive = 1;
    int64_t high_inclusive = this->max_threads_per_block_;
    if (spatial_loop_product > 2 * this->thread_warp_size_) {
      low_inclusive = this->thread_warp_size_;
    }
    sch->Annotate(block_rv, tir::attr::meta_schedule_thread_extent_low_inclusive,
                  Integer(low_inclusive));
    sch->Annotate(block_rv, tir::attr::meta_schedule_thread_extent_high_inclusive,
                  Integer(high_inclusive));
  }
  return {state};
}

std::vector<State> MultiLevelTilingNode::AddReadReuse(State state) const {
  const ReuseConfig& config = this->reuse_read_;
  if (config.req == ReuseType::kNoReuse) {
    if (state.tensor_core_is_used) state = TensorCoreLoad(state);
    return {std::move(state)};
  }
  ICHECK(config.req != ReuseType::kMayReuse);
  const BlockRV& block_rv = state.block_rv;
  std::vector<State> results;
  results.reserve(config.levels.size());
  for (int level : config.levels) {
    Schedule sch = state.sch->Copy();
    sch->Seed(state.sch->ForkSeed());
    const LoopRV& loop_rv = state.tiles[level - 1].back();
    // Enumerate all buffers that are read but not written
    std::vector<int> read_buffer_ndims = tir::GetReadBufferNDims(sch->GetSRef(block_rv));
    for (int i = 0, n_reads = read_buffer_ndims.size(); i < n_reads; ++i) {
      int buffer_ndim = read_buffer_ndims[i];
      if (buffer_ndim == -1) {
        continue;
      }
      // Do cache_read
      BlockRV cache_read_block = sch->CacheRead(block_rv, i, config.scope);
      {
        tir::Annotate(sch->state(), sch->GetSRef(cache_read_block),  //
                      tir::attr::meta_schedule_cache_type,
                      Integer(tir::attr::meta_schedule_cache_type_read));
      }
      // Insert cache_read block to the proper place
      sch->ComputeAt(cache_read_block, loop_rv, true);
      // Fuse the iterators of the cache_read
      Array<LoopRV> buffer_loops = sch->GetLoops(cache_read_block);
      LoopRV fused = sch->Fuse(Array<LoopRV>{buffer_loops.end() - buffer_ndim,  //
                                             buffer_loops.end()});
      // Annotate cooperative fetching
      if (!vector_load_lens.empty()) {
        int n = vector_load_lens.size();
        double prob = 1.0 / n;
        tir::ExprRV vector_load_len =
            sch->SampleCategorical(support::AsArray<int, Integer>(vector_load_lens),
                                   Array<FloatImm>(n, FloatImm(DataType::Float(64), prob)));
        sch->Annotate(cache_read_block, tir::attr::meta_schedule_cooperative_fetch,
                      vector_load_len);
      }
    }
    State new_state = state;
    new_state.sch = sch;
    if (new_state.tensor_core_is_used) new_state = TensorCoreLoad(new_state);
    results.push_back(std::move(new_state));
  }
  return results;
}

inline std::vector<State> MultiLevelTilingNode::FuseWriteReuse(State state) const {
  const ReuseConfig& config = this->reuse_write_;
  if (config.req == ReuseType::kNoReuse) {
    if (state.tensor_core_is_used) state = TensorCoreStoreFusion(state, r_indices_.front() - 1);
    return {std::move(state)};
  }
  // If the only-consumer does not exist, or is not elementwise, then do not do fusion
  if (!state.write_cache.defined()) {
    if (state.tensor_core_is_used) state = TensorCoreStoreFusion(state, r_indices_.front() - 1);
    return {std::move(state)};
  }
  std::vector<State> results;
  // Special case.
  //    Stages added by `cache_write` must be fused at some level, otherwise it has no benefit.
  //    On the other hand, If the consumer stage is not added by  `cache_write`,
  //    we may choose not to fuse by setting `must_cache_write = False`
  if (!state.write_cache_is_added && config.req != ReuseType::kMustReuse) {
    results.push_back(state);
  }
  BlockRV consumer = state.write_cache.value();
  // Enumerate the level of tile to be fused at
  for (int level : config.levels) {
    State new_state = state;
    new_state.sch = state.sch->Copy();
    new_state.sch->Seed(state.sch->ForkSeed());
    const LoopRV& loop_rv = new_state.tiles[level - 1].back();
    if (new_state.tensor_core_is_used) new_state = TensorCoreStoreFusion(new_state, level - 1);
    new_state.sch->ReverseComputeAt(consumer, loop_rv, true);
    results.push_back(std::move(new_state));
  }
  return results;
}

// Constructor

ScheduleRule ScheduleRule::MultiLevelTiling(String structure, Optional<Array<String>> tile_binds,
                                            bool use_tensor_core,
                                            Optional<Integer> max_innermost_factor,
                                            Optional<Array<Integer>> vector_load_lens,
                                            Optional<Map<String, ObjectRef>> reuse_read,
                                            Optional<Map<String, ObjectRef>> reuse_write) {
  auto node = MultiLevelTilingInitCommon<MultiLevelTilingNode>(
      structure, tile_binds, max_innermost_factor, vector_load_lens, reuse_read, reuse_write);
  node->use_tensor_core = use_tensor_core;
  if (use_tensor_core) {
    // Check whether corresponding wmma intrinsics are registered
    tir::TensorIntrin::Get("wmma_sync");
    tir::TensorIntrin::Get("wmma_load_a");
    tir::TensorIntrin::Get("wmma_load_b");
    tir::TensorIntrin::Get("wmma_store");
    tir::TensorIntrin::Get("wmma_fill");
  }
  return ScheduleRule(node);
}

State MultiLevelTilingNode::TensorCoreLoad(State state);
  // Add the cache read stage for Tensor Core
  state.tensor_core_load_A = state.sch->CacheRead(state.block_rv, 1, "wmma.matrix_a");
  state.tensor_core_load_B = state.sch->CacheRead(state.block_rv, 2, "wmma.matrix_b");
  const Array<LoopRV>& r_tiles = state.tiles[r_indices_.back()];
  // Insert cache_read block to the proper place
  ICHECK(!r_tiles.empty()) << "ValueError: Cannot find any reduction loop in the block";
  state.sch->ComputeAt(state.tensor_core_load_A.value(), r_tiles.back(), true);
  state.sch->ComputeAt(state.tensor_core_load_B.value(), r_tiles.back(), true);
  // Annotate the block
  state.sch->Annotate(state.tensor_core_load_A.value(), tir::attr::meta_schedule_auto_tensorize,
                      String("wmma_load_a"));
  state.sch->Annotate(state.tensor_core_load_B.value(), tir::attr::meta_schedule_auto_tensorize,
                      String("wmma_load_b"));
  return state;
}

State MultiLevelTilingNode::TensorCoreStore(State state);
  // Add the cache read stage for Tensor Core
  state.tensor_core_store = state.sch->CacheWrite(state.block_rv, 0, "wmma.accumulator");
  // Annotate the block
  state.sch->Annotate(state.tensor_core_store.value(), tir::attr::meta_schedule_auto_tensorize,
                      String("wmma_store"));
  return state;
}

State MultiLevelTilingNode::TensorCoreStoreFusion(State state, int level);
  const LoopRV& loop = state.tiles[level].back();
  state.sch->ReverseComputeAt(state.tensor_core_store.value(), loop, true);
  return state;
}

BlockRV MultiLevelTilingNode::GetRootBlockRV(const Schedule& sch, BlockRV block_rv) const {
  const tir::StmtSRefNode* block = sch->GetSRef(block_rv).get();
  for (; block->parent != nullptr; block = block->parent)
    ;
  for (const auto& kv : sch->mod()->functions) {
    const GlobalVar& gv = kv.first;
    const BaseFunc& base_func = kv.second;
    if (const auto* func = base_func.as<tir::PrimFuncNode>()) {
      const tir::BlockNode* root = func->body.as<tir::BlockRealizeNode>()->block.get();
      if (root == block->StmtAs<tir::BlockNode>()) {
        BlockRV root_rv = sch->GetBlock(root->name_hint, gv->name_hint);
        return root_rv;
      }
    }
  }
  ICHECK(false) << "Ill schedule data structure";
  throw;
}


TVM_REGISTER_NODE_TYPE(MultiLevelTilingNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleMultiLevelTiling")
    .set_body_typed(ScheduleRule::MultiLevelTiling);

}  // namespace meta_schedule
}  // namespace tvm
