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
"""Integration test for CUDA with Tensor Core"""
# pylint: disable=missing-function-docstring
import pytest
import tempfile
import tvm
from tvm.script import tir as T
from tvm import te, tir
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing import te_workload
import tvm.testing
import numpy as np
import os
from tvm.contrib import nvcc
import sys
from tvm import relay
from tvm.meta_schedule.integration import extract_task_from_relay, ApplyHistoryBest
from tvm.meta_schedule.tune import tune_relay
from tvm.meta_schedule import EvolutionarySearchConfig, ReplayTraceConfig, ReplayFuncConfig
from tvm.contrib import graph_executor
from tvm.meta_schedule.database import JSONDatabase
import os.path as osp

from tvm.meta_schedule.arg_info import ArgInfo
from tvm.meta_schedule.arg_info import TensorInfo
from tvm.meta_schedule.builder import BuilderInput, LocalBuilder
from tvm.meta_schedule.runner import (
    EvaluatorConfig,
    LocalRunner,
    PyRunner,
    RPCConfig,
    RPCRunner,
    RunnerFuture,
    RunnerInput,
)
from tvm.tir import FloatImm

TARGET = tvm.target.Target("nvidia/geforce-rtx-3070")


def get_huggingface_model(model_name: str):
    import torch
    import transformers

    if model_name == "gpt2":
        configuration = transformers.GPT2Config(torchscript=True)
        model = transformers.GPT2Model(configuration)
        model.eval()
    elif model_name == "bert":
        configuration = transformers.BertConfig(torchscript=True)
        model = transformers.BertModel(configuration)
        model.eval()
    else:
        raise Exception("Not supported model")

    input_shape = [16, 512]
    input_data = np.random.uniform(0, 100, size=input_shape).astype("long")
    torch_input_data = torch.from_numpy(input_data)
    scripted_model = torch.jit.trace(model.cpu(), torch_input_data).eval()

    shape_dict = {"input0": input_shape}
    mod, params = relay.frontend.from_pytorch(scripted_model, list(shape_dict.items()))
    return mod, params, input_data


def _create_json_database(tmpdir: str) -> JSONDatabase:
    path_workload = osp.join(tmpdir, "workloads.json")
    path_tuning_record = osp.join(tmpdir, "tuning_records.json")
    return JSONDatabase(path_workload, path_tuning_record)


def test_integration_hugginface_model(model_name: str):
    mod, params, input_data = get_huggingface_model(model_name)
    mod = relay.transform.InferType()(mod)
    print(mod)
    assert 0
    target = tvm.target.Target("nvidia/geforce-rtx-3070", host="llvm")
    dev = tvm.device("cuda", 0)

    def get_output(data, lib):
        gmod = graph_executor.GraphModule(lib["default"](dev))
        gmod.set_input("input0", data)
        gmod.run()
        return gmod.get_output(0).numpy()

    # Measure performance w/o tuning
    with tvm.transform.PassContext(opt_level=3):
        lib0: tvm.module = tvm.relay.build(mod, target=target, params=params)
        gmod0 = graph_executor.GraphModule(lib0["default"](dev))
        ftimer = gmod0.module.time_evaluator("run", dev, number=10, repeat=20)
        perfs = np.array(ftimer().results) * 1000
        perf0 = np.mean(perfs)

    # Measure performance w/ tuning
    with tempfile.TemporaryDirectory() as work_dir:
        db = _create_json_database("tuning")
        lib1: tvm.module = tune_relay(
            mod=mod,
            params=params,
            target=target,
            config=ReplayTraceConfig(
                num_trials_per_iter=20,
                num_trials_total=200,
            ),
            # config=EvolutionarySearchConfig(
            #    num_trials_per_iter=100,
            #    num_trials_total=2000,
            #    population_size=100,
            # ),
            # config=ReplayFuncConfig(
            #    num_trials_per_iter=20,
            #    num_trials_total=2000,
            # ),
            work_dir=work_dir,
            # database=db,
        )
        gmod1 = graph_executor.GraphModule(lib1["default"](dev))
        ftimer = gmod1.module.time_evaluator("run", dev, number=10, repeat=20)
        perfs = np.array(ftimer().results) * 1000
        perf1 = np.mean(perfs)

    # Check correctness
    actual_output = get_output(input_data, lib0)
    expected_output = get_output(input_data, lib1)
    assert np.allclose(actual_output, expected_output, rtol=1e-4, atol=2e-4)

    # Print speedup from tuning
    print(f"speedup: {perf0:.4f} (ms)/{perf1:.4f} (ms)={perf0/perf1:.4f}x")

    # extracted_tasks = extract_task_from_relay(mod, TARGET, params)
    # print(extracted_tasks)


if __name__ == "__main__":
    test_integration_hugginface_model("bert")
    # test_integration_matmul()
