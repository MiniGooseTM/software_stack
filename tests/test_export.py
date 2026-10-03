import re
from pathlib import Path

import numpy as np
import pytest

from self_balancing_robot.envs.balance_env import ACTOR_FRAME_SIZE
from self_balancing_robot.export import ACTIVATION_RELU, export, frame_fields_crc, read_policy_file

MODEL = Path(__file__).resolve().parents[1] / "models" / "ppo_balance.zip"


@pytest.mark.skipif(not MODEL.exists(), reason="no trained model")
def test_export_writes_every_layer(tmp_path):
    # `export` itself checks its forward pass against the SB3 policy.
    output = tmp_path / "policy_weights.h"
    export(MODEL, output)
    header = output.read_text()
    layers = int(re.search(r"#define POLICY_NUM_LAYERS (\d+)", header).group(1))
    for i in range(layers):
        assert f"POLICY_W{i}[" in header and f"POLICY_B{i}[" in header
    assert "#define POLICY_INPUT_SIZE 88" in header


@pytest.mark.skipif(not MODEL.exists(), reason="no trained model")
def test_policy_file_reproduces_its_test_vectors(tmp_path):
    export(MODEL, tmp_path / "policy_weights.h")
    policy = read_policy_file((tmp_path / "policy.bin").read_bytes())
    assert policy["frame_size"] == ACTOR_FRAME_SIZE
    assert policy["frame_fields_crc"] == frame_fields_crc()
    assert policy["control_period_s"] == pytest.approx(0.01)
    assert policy["name"].endswith("ppo_balance.zip")

    # The forward pass the firmware does, on the stored inputs, must give SB3's stored actions.
    x = policy["test_inputs"].astype(np.float64)
    for weight, bias, activation in policy["layers"]:
        x = x @ weight.T + bias
        if activation == ACTIVATION_RELU:
            x = np.maximum(x, 0.0)
    np.testing.assert_allclose(np.clip(x, -1.0, 1.0), policy["test_outputs"], atol=1e-5)
    # Real observations, not all zeros, and actions that are not all saturated
    assert np.abs(policy["test_inputs"]).max() > 0.1
    assert np.abs(policy["test_outputs"]).min() < 1.0


@pytest.mark.skipif(not MODEL.exists(), reason="no trained model")
def test_read_policy_file_rejects_corruption(tmp_path):
    export(MODEL, tmp_path / "policy_weights.h")
    data = bytearray((tmp_path / "policy.bin").read_bytes())
    data[200] ^= 0xFF
    with pytest.raises(ValueError, match="CRC"):
        read_policy_file(bytes(data))
