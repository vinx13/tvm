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
#include <tvm/ir/ir_builder.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace ir_builder {

void IRBuilderFrameNode::EnterWithScope() {
  IRBuilder::Current()->frames.push_back(GetRef<IRBuilderFrame>(this));
}

void IRBuilderFrameNode::ExitWithScope() {
  for (auto it = callbacks.rbegin(); it != callbacks.rend(); ++it) {
    (*it)();
  }
  this->callbacks.clear();
  IRBuilder::Current()->frames.pop_back();
}

void IRBuilderFrameNode::AddCallback(runtime::TypedPackedFunc<void()> callback) {
  if (IRBuilder::Current()->frames.empty()) {
    LOG(FATAL) << "ValueError: No frames in Builder to add callback";
  }
  IRBuilder::Current()->frames.back()->callbacks.push_back(callback);
}

IRBuilder::IRBuilder() {
  ObjectPtr<IRBuilderNode> n = make_object<IRBuilderNode>();
  n->frames.clear();
  n->result = NullOpt;
  data_ = n;
}

std::vector<IRBuilder>* ThreadLocalBuilderStack() {
  thread_local std::vector<IRBuilder> stack;
  return &stack;
}

void IRBuilder::EnterWithScope() {
  IRBuilderNode* n = this->get();
  CHECK(n->frames.empty()) << "ValueError: There are frame(s) left in the builder: "
                           << n->frames.size()
                           << ". Please use a fresh new builder every time building IRs";
  n->result = NullOpt;
  std::vector<IRBuilder>* stack = ThreadLocalBuilderStack();
  stack->push_back(*this);
}

void IRBuilder::ExitWithScope() {
  std::vector<IRBuilder>* stack = ThreadLocalBuilderStack();
  ICHECK(!stack->empty());
  stack->pop_back();
}

IRBuilder IRBuilder::Current() {
  std::vector<IRBuilder>* stack = ThreadLocalBuilderStack();
  CHECK(!stack->empty()) << "ValueError: No builder in current scope";
  return stack->back();
}

IRModuleFrame IRModule() {
  ObjectPtr<IRModuleFrameNode> n = make_object<IRModuleFrameNode>();
  n->global_vars.clear();
  n->functions.clear();
  return IRModuleFrame(n);
}

void IRModuleFrameNode::ExitWithScope() {
  ICHECK_EQ(functions.size(), global_vars.size());
  int n = functions.size();
  Map<GlobalVar, BaseFunc> func_map;
  for (int i = 0; i < n; ++i) {
    func_map.Set(global_vars[i], functions[i]);
  }
  IRBuilder builder = IRBuilder::Current();
  ICHECK(!builder->result.defined()) << "ValueError: Builder.result has already been set";
  builder->result = tvm::IRModule(func_map);
}

namespace details {

Namer::FType& Namer::vtable() {
  static FType inst;
  return inst;
}

void Namer::Name(ObjectRef node, String name) {
  static const FType& f = vtable();
  CHECK(node.defined()) << "ValueError: Cannot name nullptr with: " << name;
  CHECK(f.can_dispatch(node)) << "ValueError: Do not know how to name type \""
                              << node->GetTypeKey();
  f(node, name);
}

}  // namespace details

TVM_REGISTER_NODE_TYPE(IRBuilderFrameNode);
TVM_REGISTER_NODE_TYPE(IRBuilderNode);
TVM_REGISTER_NODE_TYPE(IRModuleFrameNode);
TVM_REGISTER_GLOBAL("ir_builder.IRBuilderFrameEnter")
    .set_body_method<IRBuilderFrame>(&IRBuilderFrameNode::EnterWithScope);
TVM_REGISTER_GLOBAL("ir_builder.IRBuilderFrameExit")
    .set_body_method<IRBuilderFrame>(&IRBuilderFrameNode::ExitWithScope);
TVM_REGISTER_GLOBAL("ir_builder.IRBuilderFrameAddCallback")
    .set_body_method<IRBuilderFrame>(&IRBuilderFrameNode::AddCallback);
TVM_REGISTER_GLOBAL("ir_builder.IRBuilder").set_body_typed([]() { return IRBuilder(); });
TVM_REGISTER_GLOBAL("ir_builder.IRBuilderEnter").set_body_method(&IRBuilder::EnterWithScope);
TVM_REGISTER_GLOBAL("ir_builder.IRBuilderExit").set_body_method(&IRBuilder::ExitWithScope);
TVM_REGISTER_GLOBAL("ir_builder.IRBuilderCurrent").set_body_typed(IRBuilder::Current);
TVM_REGISTER_GLOBAL("ir_builder.IRBuilderGet")
    .set_body_method<IRBuilder>(&IRBuilderNode::Get<ObjectRef>);
TVM_REGISTER_GLOBAL("ir_builder.IRBuilderName").set_body_typed(IRBuilder::Name<ObjectRef>);
TVM_REGISTER_GLOBAL("ir_builder.IRModule").set_body_typed(IRModule);

}  // namespace ir_builder
}  // namespace tvm
