import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose

from nanoserve.engine.layers import RMSNorm, SwiGLU
from nanoserve.reference.llama.layers import (
    rms_norm as numpy_rms_norm,
)
from nanoserve.reference.llama.layers import (
    swiglu as numpy_swiglu,
)

RTOL = 1e-5
ATOL = 1e-6


def test_rms_norm_matches_numpy_reference() -> None:
    inputs = np.array(
        [[[1.0, 2.0, 3.0, 4.0]]],
        dtype=np.float64,
    )
    weight = np.array([0.5, 1.0, 1.5, 2.0])

    module = RMSNorm(hidden_size=4, epsilon=1e-6)

    with torch.no_grad():
        module.weight.copy_(torch.tensor(weight, dtype=torch.float32))
        torch_result = module(torch.tensor(inputs, dtype=torch.float32))

    numpy_result = numpy_rms_norm(
        inputs,
        weight,
        epsilon=1e-6,
    )

    assert torch_result.shape == (1, 1, 4)
    assert_allclose(
        torch_result.numpy(),
        numpy_result,
        rtol=RTOL,
        atol=ATOL,
    )


def test_rms_norm_preserves_dtype_and_device() -> None:
    module = RMSNorm(hidden_size=4)
    inputs = torch.ones(2, 3, 4, dtype=torch.float32)

    output = module(inputs)

    assert output.dtype == inputs.dtype
    assert output.device == inputs.device


def test_swiglu_matches_numpy_reference() -> None:
    inputs = np.array(
        [[[1.0, 2.0, 3.0, 4.0]]],
        dtype=np.float64,
    )
    gate_weight = np.arange(32, dtype=np.float64).reshape(8, 4) / 32.0
    up_weight = np.arange(32, dtype=np.float64).reshape(8, 4) / 64.0
    down_weight = np.arange(32, dtype=np.float64).reshape(4, 8) / 32.0

    module = SwiGLU(
        hidden_size=4,
        intermediate_size=8,
    )

    with torch.no_grad():
        module.gate_proj.weight.copy_(torch.tensor(gate_weight, dtype=torch.float32))
        module.up_proj.weight.copy_(torch.tensor(up_weight, dtype=torch.float32))
        module.down_proj.weight.copy_(torch.tensor(down_weight, dtype=torch.float32))
        torch_result = module(torch.tensor(inputs, dtype=torch.float32))

    numpy_result = numpy_swiglu(
        inputs,
        gate_weight,
        up_weight,
        down_weight,
    )

    assert torch_result.shape == inputs.shape
    assert_allclose(
        torch_result.numpy(),
        numpy_result,
        rtol=RTOL,
        atol=ATOL,
    )


def test_swiglu_registers_all_projection_parameters() -> None:
    module = SwiGLU(
        hidden_size=4,
        intermediate_size=8,
    )

    parameter_names = dict(module.named_parameters())

    assert set(parameter_names) == {
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    }


def test_swiglu_preserves_device() -> None:
    module = SwiGLU(
        hidden_size=4,
        intermediate_size=8,
    )
    inputs = torch.ones(2, 3, 4)

    output = module(inputs)

    assert output.device == inputs.device


def test_rms_norm_rejects_wrong_hidden_size() -> None:
    module = RMSNorm(hidden_size=4)

    with pytest.raises(ValueError, match="configured hidden_size"):
        module(torch.ones(2, 3))
