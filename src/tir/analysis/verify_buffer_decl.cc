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
 * \brief Verify buffer declaration. This pass checks if the buffer is explicitly declared via
 *  `DeclBuffer` before it is used.
 * \file verify_buffer_decl.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/expr_functor.h>
#include "tvm/tir/stmt.h"

namespace tvm {
namespace tir {

class BufferDeclVerifier : public StmtExprVisitor {
 public:
  static void Verify(const Array<Buffer> param_buffers, const Stmt& body) {
    BufferDeclVerifier verifier;
    for (const auto& buffer : param_buffers) {
      verifier.buffer_in_scope_.insert(buffer);
    }
    verifier(body);
  }

 private:
  void VisitStmt_(const DeclBufferNode* op) {
    buffer_in_scope_.insert(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
    buffer_in_scope_.erase(op->buffer);
  }

  void CheckBufferInScope(const Buffer& buffer) {
    if (buffer_in_scope_.count(buffer) == 0) {
      LOG(FATAL) << "Buffer " << buffer << " is not in scope";
    }
  }

  void VisitExpr_(const BufferLoadNode* op) {
    CheckBufferInScope(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) {
    CheckBufferInScope(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BlockNode* op) {
    for (const Buffer& buffer : op->alloc_buffers) {
      buffer_in_scope_.insert(buffer);
    }
    for (const MatchBufferRegion& match_buffer : op->match_buffers) {
      buffer_in_scope_.insert(match_buffer->buffer);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_in_scope_;
};

void VerifyBufferDecl(const PrimFunc& prim_func) {
  Array<Buffer> params;
  for (const auto& kv : prim_func->buffer_map) {
    params.push_back(kv.second);
  }
  for (const auto& kv : prim_func->preflattened_buffer_map) {
    params.push_back(kv.second);
  }
  BufferDeclVerifier::Verify(params, prim_func->body);
}

TVM_REGISTER_GLOBAL("tir.analysis.VerifyBufferDecl").set_body_typed(VerifyBufferDecl);

namespace transform {

Pass VerifyBufferDecl() {
  auto pass_func = [=](IRModule mod, PassContext ctx) {
    for (auto kv : mod->functions) {
      if (auto* n = kv.second.as<PrimFuncNode>()) {
        auto func = GetRef<PrimFunc>(n);
        VerifyBufferDecl(func);
      }
    }
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tir.VerifyBufferDecl", {});
}

}  // namespace transform

}  // namespace tir
}  // namespace tvm
