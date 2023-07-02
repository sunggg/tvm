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
import pytest

import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relax, TVMError
from tvm.relax.backend.contrib.cublas import partition_for_cublas
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
from tvm.relax.testing import get_relax_matmul_module


@pytest.fixture(autouse=True)
def reset_seed():
    np.random.seed(0)

has_cublas = tvm.get_global_func("relax.ext.cublas", True)
has_cutlass = tvm.get_global_func("relax.ext.cutlass", True)

cublas_enabled = pytest.mark.skipif(
    not has_cublas,
    reason="CUBLAS not enabled.",
)
cutlass_enabled = pytest.mark.skipif(
    not has_cutlass,
    reason="CUTLASS not enabled.",
)

pytestmark = [cublas_enabled, cutlass_enabled]


def build_and_run(mod, inputs_np, target, legalize=False):
    if legalize:
        mod = relax.transform.LegalizeOps()(mod)

    dev = tvm.device(target, 0)
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, dev)
    f = vm["main"]
    inputs = [tvm.nd.array(inp, dev) for inp in inputs_np]
    return f(*inputs).numpy()


def get_test_mod():
    x_shape = (8, 8)
    y_shape= (8, 8)
    dtype = "float16"
   
    x = np.random.randn(*x_shape).astype(dtype)
    y = np.random.randn(*y_shape).astype(dtype)
    inps = (x,y)    

    return get_relax_matmul_module(
        x_shape,
        y_shape,
        dtype,
    ), inps

def verify_roundtrip(mod, inps, target):
    out1 = build_and_run(mod, inps, target)
    # Roundtrip
    new_mod = tvm.ir.load_json(tvm.ir.save_json(mod))
    out2 = build_and_run(new_mod, inps, target)
    tvm.testing.assert_allclose(out1, out2)


def test_cutlass():
    mod, inps = get_test_mod()
    mod = partition_for_cutlass(mod)
    mod = relax.transform.RunCodegen()(mod)
    # cutlass BYOC is based on source module that is not 
    # binary serializable. Thus, it would raise an error. 
    with pytest.raises(TVMError):
        verify_roundtrip(mod, inps, "cuda")

def test_cublas():
    mod, inps = get_test_mod()
    mod = partition_for_cublas(mod)
    mod = relax.transform.RunCodegen()(mod)
    # cublas BYOC is based on Json runtime that is binary serializable. 
    # thus, it is roundtrip-able. 
    verify_roundtrip(mod, inps, "cuda")

if __name__ == "__main__":
    tvm.testing.main()
