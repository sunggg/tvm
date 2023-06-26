import tvm
import numpy as np
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
import tvm.relax.backend.contrib.coreml
from test_codegen_coreml import verify

"""
@I.ir_module
class Module:
    @R.function
    def fused_relax_add1(
        x: R.Tensor((10, 10), dtype="float32"), y: R.Tensor((10, 10), dtype="float32")
    ) -> R.Tensor((10, 10), dtype="float32"):
        R.func_attr({"Codegen": "coreml", "Primitive": 1, "global_symbol": "fused_relax_add1"})
        with R.dataflow():
            # from tvm.script import relax as R

            @R.function
            def lv_1(
                x_1: R.Tensor((10, 10), dtype="float32"),
                y_1: R.Tensor((10, 10), dtype="float32"),
            ) -> R.Tensor((10, 10), dtype="float32"):
                R.func_attr({"Composite": "coreml.add", "Primitive": 1})
                with R.dataflow():
                    gv: R.Tensor((10, 10), dtype="float32") = R.add(x_1, y_1)
                    R.output(gv)
                return gv

            lv_1: R.Tensor((10, 10), dtype="float32") = lv(x, y)
            gv: R.Tensor((10, 10), dtype="float32") = lv_1
            R.output(gv)
        return gv

    @R.function
    def main(
        x: R.Tensor((10, 10), dtype="float32"), y: R.Tensor((10, 10), dtype="float32")
    ) -> R.Tensor((10, 10), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv: R.Tensor((10, 10), dtype="float32") = cls.fused_relax_add1(x, y)
            R.output(gv)
        return gv
"""
"""
@I.ir_module
class Module:
    @R.function
    def fused_relax_add1(
        x: R.Tensor((10, 10), dtype="float32"), y: R.Tensor((10, 10), dtype="float32")
    ) -> R.Tensor((10, 10), dtype="float32"):
        R.func_attr({"Codegen": "coreml", "global_symbol": "fused_relax_add1"})

        @R.function
        def lv_2(
            x_1: R.Tensor((10, 10), dtype="float32"),
            y_1: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tensor((10, 10), dtype="float32"):
            R.func_attr({"Composite": "coreml.add"})
            with R.dataflow():
                gv: R.Tensor((10, 10), dtype="float32") = R.add(x_1, y_1)
                R.output(gv)
            return gv

        lv_1: R.Tensor((10, 10), dtype="float32") = lv_2(x, y)
        return lv_1

    @R.function
    def main(
        x: R.Tensor((10, 10), dtype="float32"), y: R.Tensor((10, 10), dtype="float32")
    ) -> R.Tensor((10, 10), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv: R.Tensor((10, 10), dtype="float32") = cls.fused_relax_add1(x, y)
            R.output(gv)
        return gv
"""
"""
# Issues:
# M1. dataflow block in the outer func
# m1. remove primitive
# m2. naming convention
# m3. does not collect all inputs. only collect input of the last callnode
# m4. last assignment is unnecessary

# issue1. CanonicalizeBinding does not work with R.output
# This might be too strong constraint
# e.g.,
#    with Dataflow():
#      y = ...
#      x = y
#      R.output(x)

@I.ir_module
class Module:
    @R.function
    def fused_relax_multiply_relax_nn_softmax(
        x: R.Tensor((10, 10), dtype="float32"), y: R.Tensor((10, 10), dtype="float32")
    ) -> R.Tensor((10, 10), dtype="float32"):
        R.func_attr(
            {
                "Codegen": "coreml",
                "Primitive": 1,
                "global_symbol": "fused_relax_multiply_relax_nn_softmax",
            }
        )

        # from tvm.script import relax as R
        @R.function
        def lv__(
            x_1: R.Tensor((10, 10), dtype="float32"),
            y_1: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tensor((10, 10), dtype="float32"):
            R.func_attr({"Composite": "coreml.multiply", "Primitive": 1})
            with R.dataflow():
                gv: R.Tensor((10, 10), dtype="float32") = R.multiply(x_1, y_1)
                R.output(gv)
            return gv

        lv_1: R.Tensor((10, 10), dtype="float32") = lv__(x, y)
        # from tvm.script import relax as R

        @R.function
        def lv1__(lv_2: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((10, 10), dtype="float32"):
            R.func_attr({"Composite": "coreml.nn.softmax", "Primitive": 1})
            with R.dataflow():
                gv: R.Tensor((10, 10), dtype="float32") = R.nn.softmax(lv_2, axis=-1)
                R.output(gv)
            return gv

        gv: R.Tensor((10, 10), dtype="float32") = lv1__(lv_1)
        # gv: R.Tensor((10, 10), dtype="float32") = lv_2

        return gv

    @R.function
    def main(
        x: R.Tensor((10, 10), dtype="float32"), y: R.Tensor((10, 10), dtype="float32")
    ) -> R.Tensor((10, 10), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv: R.Tensor((10, 10), dtype="float32") = cls.fused_relax_multiply_relax_nn_softmax(
                x, y
            )
            R.output(gv)
        return gv


mod = Module
assert tvm.relax.analysis.well_formed(mod)
mod = tvm.relax.transform.RunCodegen()(mod)
assert tvm.relax.analysis.well_formed(mod)

# mod.show()


target, dev = "llvm", tvm.cpu(0)
x_data = tvm.nd.array(np.random.rand(10, 10).astype("float32"), dev)
y_data = tvm.nd.array(np.random.rand(10, 10).astype("float32"), dev)

ex1 = relax.build(mod, target=target)
vm1 = relax.VirtualMachine(ex1, dev)
out1 = vm1["main"](x_data, y_data)

"""
# from tvm.script import ir as I
# from tvm.script import relax as R


@I.ir_module
class Module:
    @R.function
    def fused_relax_nn_softmax(
        x: R.Tensor((10, 10), dtype="float32")
    ) -> R.Tensor((10, 10), dtype="float32"):
        R.func_attr({"Composite": "coreml.nn.softmax", "Primitive": 1})
        gv: R.Tensor((10, 10), dtype="float32") = R.nn.softmax(x, axis=-1)
        return gv

    @R.function
    def main(x: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((10, 10), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((10, 10), dtype="float32") = cls.fused_relax_nn_softmax(x)
            gv: R.Tensor((10, 10), dtype="float32") = lv
            R.output(gv)
        return gv


mod = Module
# mod.show()
mod = relax.transform.MergeCompositeFunctions()(mod)
# mod.show()
mod = tvm.relax.transform.RunCodegen()(mod)
