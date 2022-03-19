import tvm
from tvm import relay
from tvm.meta_schedule.builder import BuilderInput, LocalBuilder
from tvm.meta_schedule.utils import get_global_func_with_default_on_worker
from tvm.meta_schedule.runner import (
    EvaluatorConfig,
    LocalRunner,
    RunnerInput,
)
import numpy as np
from tvm.runtime import Module
from typing import List
import itertools
import pandas as pd


def pf_dense(inputs, attrs, dim, stride):
    new_workload = []
    if dim == 0:
        for ii in range(1, inputs[0][0][dim], stride):
            new_input = ((ii, inputs[0][0][1]), inputs[0][1])
            new_workload.append([[new_input, inputs[1]], attrs])

    elif dim == 1:
        for ii in range(1, inputs[0][0][dim], stride):
            new_input1 = ((inputs[0][0][0], ii), inputs[0][1])
            new_input2 = ((inputs[1][0][0], ii), inputs[1][1])
            new_workload.append([[new_input1, new_input2], attrs])
    else:
        raise Exception("Not implemented yet")
    return new_workload


def pf_conv2d(inputs, attrs, dim, stride):
    new_workload = []
    if dim == 1:
        for ii in range(1, inputs[0][0][dim], stride):
            new_input1 = list(inputs[0])
            new_input1[0][dim] = ii
            new_input2 = list(inputs[1])
            new_input2[0][dim] = ii
            new_workload.append([[new_input1, new_input2], attrs])
    else:
        raise Exception("Not implemented yet")
    return new_workload


def pf_grouped_conv2d(inputs, attrs, dim, stride):
    new_workload = []
    if dim == 1:
        for ii in range(1, inputs[0][0][dim], stride):
            new_input1 = list(inputs[0])
            new_input1[0][dim] = ii
            new_input2 = list(inputs[1])
            new_input2[0][dim] = ii

            new_attrs = dict(attrs)
            new_attrs["groups"] = ii
            new_workload.append([[new_input1, new_input2], new_attrs])

    else:
        raise Exception("Not implemented yet")

    return new_workload


OP_SPECS = {
    "bert-dense2d": {
        "impl": relay.nn.dense,
        "inputs": [[[8192, 768], "float32"], [[768, 768], "float32"]],
        # "output_shape": [[8192, 768], "float32"],
        "attrs": None,
        "target": "cublas",
        "partition_func": pf_dense,
    },
    "resnet50-conv2d": {
        "impl": relay.nn.conv2d,
        "inputs": [
            [[1, 256, 56, 56], "float32"],
            [[128, 256, 1, 1], "float32"],
        ],
        # "output_shape": [[8, 128, 28, 28], "float32"],
        "attrs": {
            "strides": (2, 2),
            "padding": (0, 0, 0, 0),
            "channels": 128,
            "kernel_size": (1, 1),
        },
        "target": "cudnn",
        "partition_func": pf_conv2d,
        "dim": 1,
        "stride": 1,
    },
    "mobilenet-conv2d": {
        "impl": relay.nn.conv2d,
        "inputs": [
            [[1, 256, 28, 28], "float32"],
            [[256, 1, 3, 3], "float32"],
        ],
        # "output_shape": [[1, 256, 28, 28], "float32"],
        "attrs": {
            "padding": (1, 1, 1, 1),
            "groups": 256,
            "channels": 128,
            "kernel_size": (3, 3),
        },
        "target": "cudnn",
        "partition_func": pf_grouped_conv2d,
        "dim": 1,
        "stride": 1,
    },
    "squeezenet-conv2d": {
        "impl": relay.nn.conv2d,
        "inputs": [
            [[1, 256, 28, 28], "float32"],
            [[256, 256, 1, 1], "float32"],
        ],
        # "output_shape": [[1, 256, 28, 28], "float32"],
        "attrs": {
            "padding": (0, 0, 0, 0),
            "channels": 256,
            "kernel_size": (1, 1),
        },
        "target": "cudnn",
        "partition_func": pf_conv2d,
        "dim": 1,
        "stride": 1,
    },
}


def gen_workload(impl, inputs, attrs):
    inps = []
    for idx, [shape, dtype] in enumerate(inputs):
        inps.append(relay.var(f"data{idx}", relay.TensorType(shape, dtype)))
    op = impl(*inps, **attrs)
    inputs = relay.analysis.free_vars(op)
    f = relay.Function(inputs, op)
    return tvm.IRModule.from_expr(f)


def generate_experiments(spec):
    exps = []
    impl = spec["impl"]
    orig_inputs = spec["inputs"]

    for pp in spec["partition_func"](orig_inputs, spec["attrs"], spec["dim"], spec["stride"]):
        inp, attrs = pp[0], pp[1]
        w = gen_workload(impl, inp, attrs)
        exps.append([w, spec["target"]])

    return exps


def measure(mod, target_str, target_host="llvm", device_id=0):
    if target_str == "cudnn" or "cublas":
        target_str = "cuda -libs=" + target_str
    elif target_str == "cuda":
        pass
    else:
        raise Exception("Unsupported target")

    target = tvm.target.Target(target_str)
    # Evaluation
    def relay_build(
        mod: Module,
        target: tvm.target.Target,
        params: dict = {},
    ):
        return tvm.relay.build_module._build_module_no_factory(mod, target, target_host, params)

    # Build candidate
    builder = LocalBuilder(f_build=relay_build)
    (builder_result,) = builder.build([BuilderInput(mod, target)])
    # print(builder_result.error_msg)
    assert builder_result.artifact_path is not None
    assert builder_result.error_msg is None

    runner_input = RunnerInput(
        builder_result.artifact_path,
        target_str,
        [],  # ArgInfo
    )

    evaluator_config = EvaluatorConfig(
        number=20,
        repeat=20,
        min_repeat_ms=100,
        enable_cpu_cache_flush=False,
    )

    # Wrap with a executor and evaluator configs
    # Evaluation function for Relay
    def eval_func(rt_mod, device, evaluator_config, repeated_args):
        rt_mod = tvm.contrib.graph_executor.GraphModule(rt_mod["default"](device))

        eval = rt_mod.module.time_evaluator(
            func_name="run",
            dev=device,
            number=evaluator_config.number,
            repeat=evaluator_config.repeat,
            min_repeat_ms=evaluator_config.min_repeat_ms,
            f_preproc="cache_flush_cpu_non_first_arg"
            if evaluator_config.enable_cpu_cache_flush
            else "",
        )
        repeated_costs: List[List[float]] = []
        for args in repeated_args:
            profile_result = eval(*args)
            repeated_costs.append(profile_result.results)

        costs = [float(cost) for cost in itertools.chain.from_iterable(repeated_costs)]
        return costs

    runner = LocalRunner(
        timeout_sec=100,
        evaluator_config=evaluator_config,
        f_run_evaluator=eval_func,
    )

    (runner_future,) = runner.run([runner_input])
    runner_result = runner_future.result()
    # print(runner_result.error_msg)
    assert runner_result.error_msg is None
    perfs = []
    for result in runner_result.run_secs:
        if isinstance(result, tvm.tir.FloatImm):
            result = result.value
        assert isinstance(result, float)
        assert result >= 0.0
        perfs.append(result)

    def _clean_build(artifact_path: str) -> None:
        f_clean_build = get_global_func_with_default_on_worker(
            "meta_schedule.remove_build_dir", None
        )
        if f_clean_build is not None:
            f_clean_build(artifact_path)
        else:
            raise RuntimeError("Unable to find remove_build_dir function.")

    _clean_build(builder_result.artifact_path)

    return tuple([np.mean(perfs), np.std(perfs)])


if __name__ == "__main__":
    device = "rtx3070"
    name = "resnet50-conv2d"
    spec = OP_SPECS[name]
    exps = generate_experiments(spec)

    data = []
    for (workload, target) in exps:
        perf = measure(workload, target)
        data.append(perf)

    pd.DataFrame(data).to_csv(
        f"results/{device}_{name}_{spec['target']}_d{spec['dim']}s{spec['stride']}.csv"
    )
