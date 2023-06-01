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
 *
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file src/relax/transform/allocate_workspace.cc
 * \brief Allocate a workspace and append it to the arguments of external functions, to
 * satisfy their temporary storage requirement.
 */

#include <tvm/ir/name_supply.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>

#include <cstddef>
#include <unordered_map>
#include <utility>

#include "../op/op_common.h"

namespace tvm {
namespace relax {

class ExternFunctionRewriter : ExprMutator {
 public:
  using ExprMutator::VisitExpr_;

  ExternFunctionRewriter(
      IRModule mod, const std::unordered_map<const FunctionNode*, size_t>& workspace_param_sizes)
      : ExprMutator(mod), name_sup_(""), workspace_param_sizes_(workspace_param_sizes) {}

  std::unordered_map<const GlobalVarNode*, Function> Run() {
    std::unordered_map<const GlobalVarNode*, Function> ret;
    for (const auto& [gvar, f] : builder_->GetContextIRModule()->functions) {
      if (f->GetAttr<Integer>(attr::kWorkspaceSize)) {
        ret[gvar.get()] = Downcast<Function>(VisitExpr(f));
      }
    }
    return ret;
  }

  Expr VisitExpr_(const FunctionNode* func_node) override {
    if (!func_node->GetAttr<String>(attr::kCodegen) &&
        !func_node->GetAttr<String>(attr::kComposite)) {
      return ExprMutator::VisitExpr_(func_node);
    }
    if (auto workspace = func_node->GetAttr<Integer>(attr::kWorkspaceSize)) {
      // Append the workspace parameter to this function.
      Array<Var> new_params = func_node->params;

      auto add_workspace_param = [this, &new_params](size_t workspace_param_size) {
        auto sinfo =
            TensorStructInfo(ShapeExpr({Integer(workspace_param_size)}), DataType::UInt(8));
        Var workspace_param(name_sup_->FreshName("workspace"), sinfo);
        new_params.push_back(workspace_param);
      };

      if (func_node->GetAttr<String>(attr::kCodegen)) {
        // Update the signature of the outer function (called by main).
        workspace_param_size_ = workspace_param_sizes_.at(func_node);
        add_workspace_param(workspace_param_size_);
        workspace_var_param_ = new_params.back();
      } else {
        // Update the signature of the inner (composite) function.
        add_workspace_param(workspace_param_size_);
      }

      return Function(new_params, VisitExpr(func_node->body), func_node->ret_struct_info,
                      func_node->is_pure, func_node->attrs);
    }
    return ExprMutator::VisitExpr_(func_node);
  }

  Expr VisitExpr_(const CallNode* call_node) override {
    auto new_op = VisitExpr(call_node->op);
    if (auto var = new_op.as<Var>()) {
      if (auto callee = builder_->LookupBinding(var.value());
          callee && callee->IsInstance<FunctionNode>() &&
          Downcast<Function>(callee.value())->GetAttr<String>(attr::kComposite)) {
        // Append the workspace argument to this call. The callee should have been updated to accept
        // a workspace as the last parameter.
        auto new_args = call_node->args;
        ICHECK(workspace_var_param_.defined());
        new_args.push_back(workspace_var_param_);
        return Call(new_op, new_args, call_node->attrs, call_node->sinfo_args, call_node->span);
      }
    }
    return ExprMutator::VisitExpr_(call_node);
  }

 private:
  NameSupply name_sup_;
  /*! \brief A variable that represents the workspace parameter passed from main. */
  Var workspace_var_param_;
  size_t workspace_param_size_ = 0;
  const std::unordered_map<const FunctionNode*, size_t>& workspace_param_sizes_;
};

class WorkspaceProvider : ExprMutator {
 public:
  explicit WorkspaceProvider(IRModule mod) : ExprMutator(mod), mod_(mod) {}
  using ExprMutator::VisitBindingBlock_;
  using ExprMutator::VisitExpr_;

  IRModule Run() {
    auto get_max_workpace_size_and_callees = [](const FunctionNode* f, IRModule mod) {
      size_t max_workspace_size = 0;
      std::vector<Function> callees;

      PostOrderVisit(GetRef<Function>(f), [=, &max_workspace_size, &callees](Expr e) {
        if (e->IsInstance<CallNode>()) {
          auto callee = Downcast<Call>(e)->op;
          if (callee->IsInstance<GlobalVarNode>()) {
            auto callee_gvar = Downcast<GlobalVar>(callee);
            auto callee_func = mod->Lookup(callee_gvar);
            if (auto workspace = callee_func->GetAttr<Integer>(relax::attr::kWorkspaceSize)) {
              auto ws_size = workspace.value()->value;
              max_workspace_size = std::max<size_t>(max_workspace_size, ws_size);
              ICHECK(callee_func->IsInstance<FunctionNode>());
              callees.push_back(Downcast<Function>(callee_func));
            }
          }
        }
      });

      return std::make_pair(max_workspace_size, callees);
    };

    std::unordered_map<const GlobalVarNode*, std::vector<Function>> all_callees;

    size_t max_ws_size = 0;
    for (const auto& [gvar, base_f] : mod_->functions) {
      if (auto f = base_f.as<FunctionNode>()) {
        auto [max_workspace_size, callees] = get_max_workpace_size_and_callees(f, mod_);
        if (!callees.empty()) {
          max_ws_size = std::max(max_ws_size, max_workspace_size);
          all_callees[gvar.get()] = callees;
        }
      }
    }

    if (all_callees.empty()) {
      return mod_;
    }

    std::unordered_map<const FunctionNode*, size_t> workspace_param_sizes;

    for (const auto& [gv, callees] : all_callees) {
      for (const auto& callee : callees) {
        workspace_param_sizes[callee.get()] = max_ws_size;
      }
    }

    auto new_funcs = relax::ExternFunctionRewriter(mod_, workspace_param_sizes).Run();

    for (const auto& [gvar, f] : new_funcs) {
      auto new_gvar = builder_->AddFunction(f, gvar->name_hint);
      // This is only required since the well-formed check requires kGlobalSymbol to be the same
      // as the actual name of the global variable.
      builder_->UpdateFunction(new_gvar,
                               WithAttr(f, tvm::attr::kGlobalSymbol, new_gvar->name_hint));
      gvar_map_[gvar] = new_gvar;
      builder_->GetContextIRModule()->Remove(GetRef<GlobalVar>(gvar));
    }

    for (const auto& [gvar_ptr, _] : all_callees) {
      auto gvar = GetRef<GlobalVar>(gvar_ptr);
      auto func = Downcast<Function>(mod_->Lookup(gvar));
      max_workspace_size_ = max_ws_size;
      auto new_func = Function(func->params, VisitExpr(func->body), func->ret_struct_info,
                               func->is_pure, func->attrs);
      builder_->UpdateFunction(gvar, new_func);
      workspace_var_main_ = Var();
    }

    return builder_->GetContextIRModule();
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block_node) final {
    builder_->BeginDataflowBlock();
    if (!workspace_var_main_.defined()) {
      auto shape = ShapeExpr({Integer(max_workspace_size_)});
      auto ty = DataTypeImm(DataType::UInt(8));
      auto storage = MakeVMAllocStorage(shape, PrimValue::Int64(0), ty);
      auto workspace = MakeVMAllocTensor(storage, PrimValue::Int64(0), shape, ty);
      workspace_var_main_ = builder_->Emit(workspace, "workspace_main");
    }
    for (const auto& binding : block_node->bindings) {
      this->VisitBinding(binding);
    }
    return builder_->EndBlock();
  }

  Expr VisitExpr_(const GlobalVarNode* gvar_node) override {
    if (gvar_map_.count(gvar_node)) {
      return gvar_map_[gvar_node];
    }
    return ExprMutator::VisitExpr_(gvar_node);
  }

  Expr VisitExpr_(const CallNode* call_node) override {
    auto new_op = VisitExpr(call_node->op);

    if (auto gv = new_op.as<GlobalVar>()) {
      auto callee = builder_->GetContextIRModule()->Lookup(gv.value());
      if (callee->HasNonzeroAttr(attr::kWorkspaceSize)) {
        auto new_args = call_node->args;
        ICHECK(workspace_var_main_.defined());
        new_args.push_back(workspace_var_main_);
        return Call(new_op, new_args, call_node->attrs, call_node->sinfo_args, call_node->span);
      }
    }

    return ExprMutator::VisitExpr_(call_node);
  }

 private:
  IRModule mod_;
  /*! \brief A variable that represents the workspace created at the beginning of main. */
  Var workspace_var_main_;
  size_t max_workspace_size_ = 0;
  /*! \brief A map from old global variables representing a function with workspace requirement to
   * the new ones that are transformed to take an additional workspace parameter. This is only
   * needed since the struct info of the global variables changes between transformation. */
  std::unordered_map<const GlobalVarNode*, GlobalVar> gvar_map_;
};

}  // namespace relax

namespace transform {

Pass AllocateWorkspace() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relax::WorkspaceProvider(m).Run(); };

  return CreateModulePass(pass_func, 0, "AllocateWorkspace", {});
}

TVM_REGISTER_GLOBAL("relax.transform.AllocateWorkspace").set_body_typed(AllocateWorkspace);

}  // namespace transform
}  // namespace tvm
