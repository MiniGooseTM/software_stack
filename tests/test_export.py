import re
from pathlib import Path

import pytest

from self_balancing_robot.export import export

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
