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

#include <tvm/driver/driver_api.h>
#include <tvm/ir/function.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>
#include <tvm/relax/struct_info.h>
#include <utility>

namespace tvm {
namespace relax {

class ExprBinder : public ExprMutator {
 public:
  explicit ExprBinder(const tvm::Map<tir::Var, PrimExpr>& symbolic_var_map)
      : symbolic_var_map_(symbolic_var_map) {}

 private:
  Expr VisitExpr_(const FunctionNode* op) final {
    tvm::Array<Var> params;
    bool all_params_unchanged = true;
    for (const Var& param : op->params) {
      Var new_param = this->VisitVarDef(param);
      params.push_back(new_param);
      if (!param.same_as(new_param)) {
        this->var_remap_[param->vid] = new_param;
        all_params_unchanged = false;
      }
    }

    Expr body = this->VisitWithNewScope(op->body, params);

    // FuncStructInfo does not depend on Expr
    if (all_params_unchanged && body.same_as(op->body)) {
      return GetRef<Expr>(op);
    } else {
      // purity won't be affected, no need to update annotation
      return Function(params, body, VisitExprDepStructInfoField(op->ret_struct_info), op->is_pure,
                      op->attrs);
    }
  }

  PrimExpr VisitPrimExpr(const PrimExpr& expr) final {
    if (const tir::VarNode* var = expr.as<tir::VarNode>()) {
      auto it = symbolic_var_map_.find(GetRef<tir::Var>(var));
      if (it != symbolic_var_map_.end()) {
        return (*it).second;
      }
    }
    return ExprMutator::VisitPrimExpr(expr);
  }


 private:
  const tvm::Map<tir::Var, PrimExpr>& symbolic_var_map_;
};

/*!
 * \brief Bind params to a specific function in a module
 * \param m The module
 * \param func_name The name of the specific function
 * \param param The param dict
 * \return The module after binding params.
 */
IRModule BindSymVars(IRModule m, String func_name, Map<String, Integer> sym_val_map) {
  Function func = Downcast<Function>(m->Lookup(func_name));
  Map<tir::Var, PrimExpr> smap;
  for(auto param:func->params){
    if(const auto* tsinfo = GetStructInfoAs<TensorStructInfoNode>(param)){
      const auto* shape = tsinfo->shape.as<ShapeExprNode>();
      ICHECK(shape != nullptr) << "Shape should be defined.";
      for(auto val:shape->values){
        if(const auto* v = val.as<tir::VarNode>()){
          if(sym_val_map.find(v->name_hint)!=sym_val_map.end())
            smap.Set(GetRef<tir::Var>(v), sym_val_map[v->name_hint]);
        }
      }
    }
    // TODO: ShapeExpr
  }
  GlobalVar gv = m->GetGlobalVar(func_name);
  Expr bound_expr= ExprBinder(smap).VisitExpr(func);
  Function bound_func = Downcast<Function>(bound_expr);
  m->Update(gv, bound_func);
  return m;
}

namespace transform {

Pass BindSymVars(String func_name, Map<String, Integer> sym_val_map) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return relax::BindSymVars(std::move(mod), func_name, sym_val_map); 
      };
  return CreateModulePass(pass_func, 0, "BindSymVars", {});
}

TVM_REGISTER_GLOBAL("relax.transform.BindSymVars").set_body_typed(BindSymVars);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
