import numpy as np
import torch
import tvm
from tvm import relay
from torch.ao.quantization.stubs import QuantStub, DeQuantStub
from torch import nn

import torch.nn.functional as F
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import torch.nn.intrinsic.quantized.dynamic as nniqd
import torch.nn.intrinsic.qat as nniqat
import torch.nn.quantized as nnq
import torch.nn.quantized._reference as nnqr
import torch.nn.quantized.dynamic as nnqd
import torch.nn.qat as nnqat
import torch.nn.qat.dynamic as nnqatd

from tvm.meta_schedule.integration import extract_task_from_relay

# Select model
model_name = "bert-base-uncased"

# Setup input
batch_size = 1
seq_len = 128
shape_dict = {
    "input_ids": (batch_size, seq_len),
    "attention_mask": (batch_size, seq_len),
    "token_type_ids": (batch_size, seq_len),
}
data1 = np.random.uniform(size=shape_dict["input_ids"]).astype("long")
data2 = np.random.uniform(size=shape_dict["attention_mask"]).astype("long")
data3 = np.random.uniform(size=shape_dict["token_type_ids"]).astype("long")

# Setup target
target = "cuda"
dev = tvm.device(target, 0)


def quantize_model(model, inp, static):
    if static:
        # model.fuse_model()
        # TODO: Need to fix this
        model.qconfig = torch.quantization.get_default_qat_qconfig()
        torch.quantization.prepare(model, inplace=True)
        # Dummy calibration
        model(*inp)
        torch.quantization.convert(model, inplace=True)
        return model
    else:
        # dynamic (i.e. weights-only) quantized model
        return torch.quantization.quantize_dynamic(model, dtype=torch.qint8)


model = torch.hub.load("huggingface/pytorch-transformers", "model", model_name, return_dict=False)
pt_data1 = torch.from_numpy(data1)
pt_data2 = torch.from_numpy(data2)
pt_data3 = torch.from_numpy(data3)
pt_input = (pt_data1, pt_data2, pt_data3)

quantized_model = quantize_model(model, pt_input, static=False)

script_module = torch.jit.trace(quantized_model, pt_input).eval()
shape_tuple = [(a, b) for (a, b) in shape_dict.items()]
mod, params = relay.frontend.from_pytorch(script_module, shape_tuple)

# print(mod)
print(f"Target: {target}, Device: {dev}")
extracted_tasks = extract_task_from_relay(mod, target)
for i, tsk in enumerate(extracted_tasks):
    print(f"[{i}] {tsk.task_name}, {tsk.mod}")

"""
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
rt_mod.set_input("input_ids", data1)
rt_mod.set_input("attention_mask", data2)
rt_mod.set_input("token_type_ids", data3)

print(rt_mod.benchmark(dev, number=1, repeat=20))
"""
