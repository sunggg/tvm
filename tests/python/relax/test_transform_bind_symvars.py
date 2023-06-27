# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import numpy as np
import tvm
import tvm.script
import tvm.testing
from tvm import relax
from tvm.script import relax as R
from tvm.script import tir as T
from tvm import ir
from tvm.ir.transform import PassContext
from tvm.ir.module import IRModule
from tvm.relax import PyExprMutator


@ir.transform.module_pass(opt_level=0)
def MockModulePass(mod: IRModule, ctx: PassContext) -> IRModule:
    @relax.expr_functor.mutator
    class SymVarBinder(PyExprMutator):
        def visit_prim_expr_(self, op: relax.expr.PrimExpr):
            print(op)
            return op

    SymVarBinder(mod).visit_expr(mod["main"])
    return mod


def test_bind_symbolic_vars():
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(("batch", "m"), dtype="float32"),
            w0: R.Tensor(("n", "m"), dtype="float32"),
            b0: R.Tensor(("n",), dtype="float32"),
            w1: R.Tensor(("k", "n"), dtype="float32"),
            b1: R.Tensor(("k",), dtype="float32"),
        ) -> R.Tensor(("batch", "k"), dtype="float32"):
            batch = T.Var("batch", "int64")
            k = T.Var("k", "int64")
            m = T.Var("m", "int64")
            n = T.Var("n", "int64")
            with R.dataflow():
                lv0 = R.call_dps_packed(
                    "linear0", (x, w0, b0), out_sinfo=R.Tensor((batch, n), dtype="float32")
                )
                out = R.call_dps_packed(
                    "linear1", (lv0, w1, b1), out_sinfo=R.Tensor((batch, k), dtype="float32")
                )
                R.output(out)
            return out

    Before.show()
    mod = MockModulePass(Before)
    # mod = relax.transform.BindParams("main", params_dict)(Before)
    # mod.show()
    """
    # Since it contains ConstantNode, it's hard to check with structural equality.
    func = mod["main"]
    assert len(func.params) == 1
    batch = func.params[0].struct_info.shape[0]
    tvm.ir.assert_structural_equal(
        func.params[0].struct_info, relax.TensorStructInfo((batch, 4), "float32")
    )
    tvm.ir.assert_structural_equal(
        func.ret_struct_info, relax.TensorStructInfo((batch, 8), "float32")
    )
    bindings = func.body.blocks[0].bindings
    tvm.ir.assert_structural_equal(
        bindings[0].var.struct_info, relax.TensorStructInfo((batch, 6), "float32")
    )
    tvm.ir.assert_structural_equal(
        bindings[1].var.struct_info, relax.TensorStructInfo((batch, 8), "float32")
    )
    """


if __name__ == "__main__":
    test_bind_symbolic_vars()
    # tvm.testing.main()
