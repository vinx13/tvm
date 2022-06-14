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

void Padding(ScheduleState self, const StmtSRef& block_sref, const Array<IntImm>& padding) {
  const BlockNode* block = block_sref->StmtAs<BlockNode>();
  ICHECK(IsTrivialBinding(self, block_sref));
  
  const BufferStoreNode* store = block->body.as<BufferStoreNode>();
  ICHECK(store != nullptr);

  StmtSRef scope_root_sref = GetScopeRoot(self, block_sref,
                                          /*require_stage_pipeline=*/true);
  ICHECK(IsCompleteBlock(self, block_sref, scope_root_sref));
  
  Array<StmtSRef> loops = GetLoops(block_sref);

  std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual> block_var_no;
  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    block_var_no[block->iter_vars[i]->var] = i;
  }

  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> indices;
}

/******** Instruction Registration ********/

struct PaddingTraits : public UnpackedInstTraits<PaddingTraits> {
  static constexpr const char* kName = "Padding";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block, Array<IntImm> padding) {
    sch->Padding(block, padding);
  }

  static String UnpackedAsPython(Array<String> outputs, BlockRV block, Array<IntImm> padding) {
    PythonAPICall py("padding");
    py.Input("block", block);
    py.Input("padding", padding);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(PaddingTraits);

}  // namespace tir
}  // namespace tvm