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
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel
"""Pattern table and codegen for CoreML"""

import os
import shutil

import tvm._ffi
from ...expr_functor import PyExprVisitor, visitor
from tvm.contrib import xcode, coreml_runtime
from tvm.contrib.xcode import compile_coreml
from typing import List
import tvm
from tvm.relax import transform
from tvm.relax.struct_info import TensorStructInfo, PrimStructInfo
from tvm.relax.expr import (
    Binding,
    BindingBlock,
    Expr,
    Call,
    Function,
    PrimValue,
    SeqExpr,
    Var,
    VarBinding,
    Constant,
)
from tvm.relax.dpl.pattern import is_op, wildcard

from ..pattern_registry import get_patterns_with_prefix, register_patterns

register_patterns(
    [
        ("coreml.add", is_op("relax.add")(wildcard(), wildcard())),
        ("coreml.multiply", is_op("relax.multiply")(wildcard(), wildcard())),
        ("coreml.clip", is_op("relax.clip")(wildcard(), wildcard(), wildcard())),
        ("coreml.expand_dims", is_op("relax.expand_dims")(wildcard())),
        ("coreml.nn.relu", is_op("relax.nn.relu")(wildcard())),
        # TODO(@tvm-team): enable when it is implemented
        # ("coreml.nn.batch_flatten", is_op("relax.nn.batch_flatten")(wildcard())),
        ("coreml.nn.softmax", is_op("relax.nn.softmax")(wildcard())),
        ("coreml.nn.avg_pool2d", is_op("relax.nn.avg_pool2d")(wildcard())),
        ("coreml.nn.conv2d", is_op("relax.nn.conv2d")(wildcard(), wildcard())),
    ]
)


def partition_for_coreml(mod):
    """
    Partition the input module into coreml-supported subgraphs.

    Parameters
    ----------
    mod: tvm.IRModule
        The IRModule to be partitioned.

    Returns
    -------
    mod: tvm.IRModule
        The resulting IRModule, containing partitioned subgraphs to be
        offloaded to the coreml backend.
    """

    patterns = get_patterns_with_prefix("coreml")
    mod = transform.FuseOpsByPattern(patterns, bind_constants=True, annotate_codegen=True)(mod)
    return mod


# Codegen for coreml


def _convert_add(builder, name, inputs, outputs, args, attrs):
    builder.add_elementwise(name=name, input_names=inputs, output_name=outputs[0], mode="ADD")


def _convert_multiply(builder, name, inputs, outputs, args, attrs):
    builder.add_elementwise(name=name, input_names=inputs, output_name=outputs[0], mode="MULTIPLY")


def _convert_clip(builder, name, inputs, outputs, args, attrs):
    builder.add_clip(
        name=name,
        input_name=inputs[0],
        output_name=outputs[0],
        min_value=inputs[1].value.value,
        max_value=inputs[2].value.value,
    )


def _convert_batch_flatten(builder, name, inputs, outputs, args, attrs):
    builder.add_flatten_to_2d(name=name, input_name=inputs[0], output_name=outputs[0])


def _convert_expand_dims(builder, name, inputs, outputs, args, attrs):
    axes = [int(v) for v in attrs["axis"]]
    builder.add_expand_dims(name=name, input_name=inputs[0], output_name=outputs[0], axes=axes)


def _convert_relu(builder, name, inputs, outputs, args, attrs):
    builder.add_activation(
        name=name, non_linearity="RELU", input_name=inputs[0], output_name=outputs[0]
    )


def _convert_softmax(builder, name, inputs, outputs, args, attrs):
    builder.add_softmax_nd(
        name=name, input_name=inputs[0], output_name=outputs[0], axis=int(attrs["axis"])
    )


def _convert_conv2d(builder, name, inputs, outputs, args, attrs):
    weight = inputs[1].data.numpy()
    if attrs["kernel_layout"] == "OIHW":
        # convert to 'HWIO'
        weight = weight.transpose([2, 3, 1, 0])
    kh, kw, kc, oc = weight.shape

    builder.add_convolution(
        name=name,
        kernel_channels=kc,
        output_channels=oc,
        height=kh,
        width=kw,
        stride_height=int(attrs["strides"][0]),
        stride_width=int(attrs["strides"][0]),
        border_mode="valid",
        groups=int(attrs["groups"]),
        W=weight,
        b=None,
        has_bias=False,
        input_name=inputs[0],
        output_name=outputs[0],
        dilation_factors=[int(v) for v in attrs["dilation"]],
        padding_top=int(attrs["padding"][0]),
        padding_bottom=int(attrs["padding"][2]),
        padding_left=int(attrs["padding"][1]),
        padding_right=int(attrs["padding"][3]),
    )


def _convert_avg_pool2d(builder, name, inputs, outputs, args, attrs):
    builder.add_pooling(
        name=name,
        height=1,
        width=1,
        stride_height=1,
        stride_width=1,
        layer_type="AVERAGE",
        padding_type="VALID",
        input_name=inputs[0],
        output_name=outputs[0],
        # is_global=True,
    )


_convert_map = {
    "add": _convert_add,
    "multiply": _convert_multiply,
    "clip": _convert_clip,
    "expand_dims": _convert_expand_dims,
    "nn.relu": _convert_relu,
    "nn.batch_flatten": _convert_batch_flatten,
    "nn.softmax": _convert_softmax,
    "nn.conv2d": _convert_conv2d,
    "nn.avg_pool2d": _convert_avg_pool2d,
}


@visitor
class CodegenCoreML(PyExprVisitor):
    """
    A visitor to traverse subgraphs and build Core ML models.
    """

    def __init__(self, model_name, function):
        import coremltools
        from coremltools.models.neural_network import NeuralNetworkBuilder

        self.model_name = model_name
        self.function = function
        self.out_map = {}
        self.model_inputs_ = []
        self.buf_idx_ = 0

        getter = tvm.get_global_func("relax.analysis.get_var2val")
        assert getter, "Cannot find `relax.analysis.get_var2val` function."

        self.var2val = getter(function)
        self.cur_binding_var = None

        # Update inputs and outputs after we visit all the nodes.
        # Set dummy values for now.
        # TODO: support multiple outputs
        inputs = [
            (
                "",
                coremltools.models.datatypes.Array(
                    1,
                ),
            )
            for _ in self.function.params
        ]
        outputs = [
            (
                "",
                coremltools.models.datatypes.Array(
                    1,
                ),
            )
        ]
        self.builder = NeuralNetworkBuilder(inputs, outputs, disable_rank5_shape_mapping=True)

    def visit_constant_(self, const):
        output = "buf_" + str(self.buf_idx_)
        self.builder.add_load_constant_nd(
            name=output,
            output_name=output,
            constant_value=const.data.numpy(),
            shape=const.data.shape,
        )
        self.buf_idx_ = self.buf_idx_ + 1
        self.out_map[const] = [output]

    def visit_var_(self, var):
        name = var.name_hint
        sinfo = var.struct_info
        if isinstance(sinfo, TensorStructInfo):
            shape = [int(v) for v in list(sinfo.shape)]
        elif isinstance(sinfo, PrimStructInfo):
            shape = []
        else:
            raise Exception("Currently not supported: ", type(sinfo))

        dtype = sinfo.dtype
        self.model_inputs_.append((name, shape, dtype))
        self.out_map[var] = [name]

    def visit_call_(self, call: Call) -> None:
        primvals = []
        attrs = []
        consts = []

        @visitor
        class Collector(PyExprVisitor):
            def visit_call_(self, call: Call) -> None:
                attrs.append(call.attrs)
                for arg in call.args:
                    if isinstance(arg, PrimValue):
                        primvals.append(arg)
                    if isinstance(arg, Constant):
                        consts.append(arg)

        assert isinstance(call.op, Var)
        assert call.op in self.var2val
        func = self.var2val[call.op]
        assert "Composite" in func.attrs, "Only composite functions are supported."
        composite_name = func.attrs["Composite"]

        Collector().visit_expr(func.body)

        inputs = []
        for arg in call.args:
            super().visit_expr(arg)
            for out in self.out_map[arg]:
                inputs.append(out)

        for arg in primvals:
            inputs.append(arg)

        for arg in consts:
            inputs.append(arg)

        outputs = ["buf_" + str(self.buf_idx_)]
        # Get the op name and remove "relax." prefix.
        op_name = composite_name[7:]
        layer_name = op_name + "_" + str(self.buf_idx_)

        print(layer_name, call.attrs, inputs, call.args)
        assert op_name in _convert_map, "{} is not supported".format(op_name)
        _convert_map[op_name](self.builder, layer_name, inputs, outputs, call.args, attrs[0])

        self.buf_idx_ = self.buf_idx_ + 1
        self.out_map[self.cur_binding_var] = outputs

    def visit_var_binding_(self, binding: VarBinding) -> None:
        self.cur_binding_var = binding.var
        self.visit_expr(binding.value)
        self.cur_binding_var = None

    def visit_binding_block_(self, block: BindingBlock) -> None:
        # We only visit the last VarBinding to retrieve
        # target composite function
        self.visit_binding(block.bindings[-1])

    def visit_seq_expr_(self, op: SeqExpr) -> None:
        for bb in op.blocks:
            self.visit_binding_block_(bb)

    def serialize(self, func: Function):
        # TODO:handle params
        # handle func body
        self.visit_expr(func.body)

    def compile(self, out_dir):
        """
        Build a Core ML model and compile it with Xcode toolchain.
        """
        import coremltools
        from coremltools.proto.Model_pb2 import ArrayFeatureType

        FEATURE_TYPE_MAP = {
            "float32": ArrayFeatureType.FLOAT32,
            "float64": ArrayFeatureType.DOUBLE,
            "int32": ArrayFeatureType.INT32,
        }
        input_names, input_dims, input_dtypes = zip(*self.model_inputs_)
        self.builder.set_input(input_names, input_dims)

        for i, dtype in enumerate(input_dtypes):
            assert dtype in FEATURE_TYPE_MAP
            input_desc = self.builder.spec.description.input
            input_desc[i].type.multiArrayType.dataType = FEATURE_TYPE_MAP[dtype]

        output_dim = [int(n) for n in self.function.struct_info.ret.shape]

        # assert self.function.body in self.out_map
        self.builder.set_output(["buf_0"], [output_dim])
        # self.builder.set_output(self.out_map[self.function.body], [output_dim])

        for i, dtype in enumerate([self.function.struct_info.ret.dtype]):
            assert dtype in FEATURE_TYPE_MAP
            output_desc = self.builder.spec.description.output
            output_desc[i].type.multiArrayType.dataType = FEATURE_TYPE_MAP[dtype]

        model = coremltools.models.MLModel(self.builder.spec)
        compile_coreml(model, self.model_name, out_dir)


@tvm._ffi.register_func("relax.ext.coreml")
def coreml_compiler(funcs, options, constant_names):
    """
    Create a CoreML runtime from a Relax module.
    """
    compiled_funcs = []
    for func in funcs:
        assert isinstance(func, tvm.relax.Function)
        model_dir = os.getcwd()
        name = str(func.attrs.global_symbol)
        builder = CodegenCoreML(name, func)
        builder.serialize(func)

        mlmodelc_path = "{}/{}.mlmodelc".format(model_dir, name)

        if os.path.exists(mlmodelc_path):
            shutil.rmtree(mlmodelc_path)

        builder.compile(model_dir)

        dev = tvm.cpu(0)
        coreml_runtime.create(name, mlmodelc_path, dev).module

        creator = tvm.get_global_func("tvm.coreml_runtime.create")
        assert creator, "Cannot find `tvm.coreml_runtime.create` function."
        compiled_funcs.append(creator(name, mlmodelc_path))
    return compiled_funcs
