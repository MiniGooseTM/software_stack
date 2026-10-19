import re

import numpy as np
import pytest
from conftest import initial_params

from self_balancing_robot.actor import Actor, save_params
from self_balancing_robot.envs.balance_env import ACTOR_FRAME_SIZE
from self_balancing_robot.export import ACTIVATION_RELU, export, frame_fields_crc, read_policy_file


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory, env):
    """A training run directory with freshly initialized (untrained) weights."""
    (normalizer, policy, value), _ = initial_params(env, seed=3)
    # Scale the (deliberately tiny) initial action layer up so the test actions vary.
    mean = policy["params"]["Dense_0"]
    policy["params"]["Dense_0"] = {**mean, "kernel": mean["kernel"] * 30.0}
    path = tmp_path_factory.mktemp("run")
    save_params(path, (normalizer, policy, value), {"stage": 1})
    return path


@pytest.fixture(scope="module")
def exported(run_dir, tmp_path_factory):
    # `export` itself checks its forward pass against Brax's policy network.
    output = tmp_path_factory.mktemp("export") / "policy_weights.h"
    export(run_dir, output)
    return output


def test_export_writes_every_layer(exported):
    header = exported.read_text()
    layers = int(re.search(r"#define POLICY_NUM_LAYERS (\d+)", header).group(1))
    assert layers == 4
    for i in range(layers):
        assert f"POLICY_W{i}[" in header and f"POLICY_B{i}[" in header
    assert "#define POLICY_INPUT_SIZE 88" in header
    assert "#define POLICY_CONTROL_PERIOD_S 0.010000f" in header


def test_policy_file_reproduces_its_test_vectors(exported, run_dir):
    policy = read_policy_file((exported.parent / "policy.bin").read_bytes())
    assert policy["frame_size"] == ACTOR_FRAME_SIZE
    assert policy["frame_fields_crc"] == frame_fields_crc()
    assert policy["control_period_s"] == pytest.approx(0.01)
    assert policy["name"] == run_dir.as_posix()[-43:]

    # The forward pass the firmware does, on the stored inputs, must give the stored actions.
    x = policy["test_inputs"].astype(np.float64)
    for weight, bias, activation in policy["layers"]:
        x = x @ weight.T + bias
        if activation == ACTIVATION_RELU:
            x = np.maximum(x, 0.0)
    np.testing.assert_allclose(np.clip(x, -1.0, 1.0), policy["test_outputs"], atol=1e-5)
    # Real observations, not all zeros, and actions that are not all saturated
    assert np.abs(policy["test_inputs"]).max() > 0.1
    assert np.abs(policy["test_outputs"]).min() < 1.0


def test_policy_file_loads_as_the_same_actor(exported, run_dir):
    from_run = Actor.load(run_dir)
    from_file = Actor.load(exported.parent / "policy.bin")
    policy = read_policy_file((exported.parent / "policy.bin").read_bytes())
    np.testing.assert_allclose(
        from_file.predict(policy["test_inputs"]), from_run.predict(policy["test_inputs"])
    )


def test_read_policy_file_rejects_corruption(exported):
    data = bytearray((exported.parent / "policy.bin").read_bytes())
    data[200] ^= 0xFF
    with pytest.raises(ValueError, match="CRC"):
        read_policy_file(bytes(data))
