import argparse
import numpy as np
import os
from numpy.lib import ufunclike
import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_runtime
import onnx
from tvm.relay import transform

from tvm.contrib.utils import tempdir
from onnxruntime.quantization import (
    quantize_static,
    quantize_dynamic,
    CalibrationDataReader,
    quantize,
)
from onnxruntime.quantization.quant_utils import find_by_name
from tvm.relay.frontend.common import infer_shape
from google.protobuf.json_format import MessageToDict
from tvm.meta_schedule.integration import extract_task_from_relay

batch_size = 1
seq_len = 128
shape_dict = {
    "input_ids": (batch_size, seq_len),
    "segment_ids": (batch_size, seq_len),
    "input_mask": (batch_size, seq_len),
}
data = dict()
for key, shape in shape_dict.items():
    data[key] = np.random.uniform(size=shape).astype("int64")

"""
def onnx_quantization():
    def get_quantized_model(fp32_name, quant_name, static):
        if static:

            class PsuedoData(CalibrationDataReader):
                def __init__(self, size):
                    super().__init__()
                    self.size = size
                    self.it = 0

                def get_next(self):
                    self.it += 1
                    if self.it == self.size:
                        return None
                    else:
                        return {
                            "input_ids": data1,
                            "attention_mask": data2,
                            "token_type_ids": data3,
                        }

            return quantize_static(fp32_name, quant_name, PsuedoData(size=1000))
        else:
            return quantize_dynamic(fp32_name, quant_name)

    def quantize_static(model) -> onnx.ModelProto:
        quantization_params = {}
        for node in model.graph.node:
            # if node.op_type == "MatMul":
            #    inp0 = find_by_name(node.input[0], model.initializer())
            #    inp1 = find_by_name(node.input[1], model.initializer())
            #    if inp0 is not None:
            #        print(f"inp0: {inp0.dims}")
            #    if inp1 is not None:
            #        print(f"inp1: {inp1.dims}")

            for inp in node.input:
                quantization_params[inp] = [0.0, 0.5]
            for output in node.output:
                quantization_params[output] = [0.0, 0.5]

        return quantize(model, static=True, quantization_params=quantization_params)

    # name = "models/bert_large_v1_1_fake_quant.onnx"
    fp32_name = "models/bert-small.onnx"
    quant_name = "models/bert-small-static-quant.onnx"
    # get_quantized_model(fp32_name, quant_name)
    # quant_model = onnx.load(quant_name)

    model = onnx.load(fp32_name)
    # mod0, params0 = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)

    quant_model = quantize_static(model)
    mod, params = relay.frontend.from_onnx(quant_model, shape_dict, freeze_params=True)
"""


model_name = "models/bert-base-qat.onnx"
quant_model_tag = "models/bert-base-int8"


def import_onnx_qat_and_save_relay(model_name, quant_model_tag):
    quant_model = onnx.load(model_name)
    mod, params = relay.frontend.from_onnx(quant_model, shape_dict, freeze_params=True)
    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.FoldConstant(),
            transform.SimplifyInference(),
            transform.FoldScaleAxis(),
        ]
    )

    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    mod = tvm.relay.transform.FakeQuantizationToInteger(use_qat=True)(mod)

    with open(f"{quant_model_tag}.json", "w") as fo:
        fo.write(tvm.ir.save_json(mod))

    with open(f"{quant_model_tag}.params", "wb") as fo:
        fo.write(relay.save_param_dict(params))


def read_relay(quant_model_tag):
    with open(f"{quant_model_tag}.json", "r") as fi:
        mod = tvm.ir.load_json(fi.read())
    with open(f"{quant_model_tag}.params", "rb") as fi:
        params = relay.load_param_dict(fi.read())

    return mod, params


# quant_model = onnx.load(model_name)
# mod, params = relay.frontend.from_onnx(quant_model, shape_dict, freeze_params=True)


import_onnx_qat_and_save_relay(model_name, quant_model_tag)
mod, params = read_relay(quant_model_tag)

target, dev = tvm.target.Target("cuda"), tvm.cuda()

print(mod)
print(f"Target: {target}, Device: {dev}")
extracted_tasks = extract_task_from_relay(mod, target)
for i, tsk in enumerate(extracted_tasks):
    print(f"[{i}] {tsk.task_name}, {tsk.mod}")


assert 0
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)


module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
for key in shape_dict.keys():
    module.set_input(key, tvm.nd.array(data[key]))

# evaluate
print("Evaluate inference time cost...")
print(module.benchmark(dev, number=20, repeat=20))
