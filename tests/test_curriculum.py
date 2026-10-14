from dataclasses import replace

import numpy as np
import pytest

from self_balancing_robot.config import RewardConfig
from self_balancing_robot.curriculum import (
    BALANCE,
    DOMAIN_RANDOMIZATION,
    FULL_COMMAND,
    Stage,
    stage_index,
    stage_table,
    validate,
    without_randomization,
)
from self_balancing_robot.curriculum import CURRICULUM as TRAINING_CURRICULUM
from self_balancing_robot.curriculum import RANDOMIZED_CURRICULUM as CURRICULUM


def test_default_curriculum_keeps_the_tuned_progression():
    """Balance, commands in quarters of the full range (sensor noise and latency growing
    alongside), then slope, push, CoM shift and wheel kicks in sixths of their maxima."""
    names = [stage.name for stage in CURRICULUM]
    assert names[0] == "balance" and len(names) == 11
    commands = [stage.command_max for stage in CURRICULUM]
    np.testing.assert_allclose(
        commands[1:5], [np.multiply(FULL_COMMAND, k / 4) for k in range(1, 5)]
    )
    assert all(command == FULL_COMMAND for command in commands[5:])
    randomized = CURRICULUM[5:]
    for k, stage in enumerate(randomized, start=1):
        assert np.isclose(stage.slope_max, k / 6 * 0.3)
        assert np.isclose(stage.push_max, k / 6 * 0.5)
        assert np.isclose(stage.com_offset_max, k / 6 * 0.008)
        assert np.isclose(stage.wheel_kick_max, k / 6 * 0.02)
        assert (stage.mass_fraction, stage.friction_fraction) == (0.10, 0.40)
        assert stage.battery_voltage == (10.5, 12.5)
        assert stage.max_latency_steps == 2 and stage.imu_angle_noise > 0
        assert (stage.min_episodes, stage.success_rate, stage.speed_ratio) == (5_000, 0.90, 0.90)
        assert stage.still_speed_limit == 0.05 and stage.reversal_success_rate == 0.85
    full_noise = CURRICULUM[-1].imu_angle_noise, CURRICULUM[-1].encoder_noise
    for k, stage in enumerate(CURRICULUM[1:5], start=1):
        assert (stage.min_episodes, stage.success_rate, stage.speed_ratio) == (2_000, 0.95, 0.90)
        assert stage.still_speed_limit == 0.02 and stage.reversal_probability == 0.2
        assert stage.stop_probability == 0.2
        assert stage.slope_max == 0 and stage.mass_fraction == 0 and stage.push_max == 0
        assert stage.wheel_kick_max == 0
        assert stage.max_latency_steps == k // 2
        noise = stage.imu_angle_noise, stage.encoder_noise
        np.testing.assert_allclose(noise, np.multiply(full_noise, k / 4))
    assert BALANCE.imu_angle_noise == 0 and BALANCE.max_latency_steps == 0
    assert BALANCE.speed_ratio is None and BALANCE.command_max == (0.0, 0.0)


def test_band_floors_come_from_the_previous_stage():
    table = stage_table(CURRICULUM)
    # Command steps sample their new third; once the range stops growing, 0 or uniform.
    floors = [(0, 0), (0, 0), (0.1, 0.625), (0.2, 1.25), (0.3, 1.875), (0, 0)]
    np.testing.assert_allclose(table["command_floor"][:6], floors, atol=1e-6)
    assert np.all(table["command_floor"][5:] == 0)
    np.testing.assert_allclose(table["slope_floor"][5:], np.arange(6) / 6 * 0.3, atol=1e-6)
    np.testing.assert_allclose(table["push_floor"][5:], np.arange(6) / 6 * 0.5, atol=1e-6)
    np.testing.assert_allclose(table["wheel_kick_floor"][5:], np.arange(6) / 6 * 0.02, atol=1e-6)


def test_reward_settings_get_a_column_per_stage():
    table = stage_table(CURRICULUM)
    weights = [stage.rewards.smoothness_weight for stage in CURRICULUM]
    np.testing.assert_allclose(table["smoothness_weight"], weights)
    assert "rewards" not in table and table["alive_weight"].shape == (len(CURRICULUM),)


def test_unchecked_criteria_always_pass_in_the_table():
    table = stage_table(CURRICULUM)
    assert table["speed_ratio"][0] == 0.0
    assert table["still_speed_limit"][0] == np.inf
    assert table["reversal_success_rate"][0] == 0.0
    assert table["min_episodes"].dtype == np.int32 and table["max_latency_steps"].dtype == np.int32


def test_stages_are_found_by_name_or_index():
    assert stage_index(CURRICULUM, "balance") == 0
    assert stage_index(CURRICULUM, "commands 0.4 m/s") == 4
    assert stage_index(CURRICULUM, 2) == 2
    assert stage_index(CURRICULUM, "-1") == len(CURRICULUM) - 1
    with pytest.raises(ValueError, match="no stage"):
        stage_index(CURRICULUM, "flying")
    with pytest.raises(ValueError, match="out of range"):
        stage_index(CURRICULUM, len(CURRICULUM))


@pytest.mark.parametrize(
    ("curriculum", "message"),
    [
        ((), "no stages"),
        ((BALANCE, BALANCE), "unique"),
        ((replace(BALANCE, slope_max=-0.1),), "negative"),
        ((replace(BALANCE, zero_probability=0.7, frontier_probability=0.5),), "above 1"),
        ((replace(BALANCE, command_max=(FULL_COMMAND[0] + 0.1, 1.0)),), "FULL_COMMAND"),
        ((replace(BALANCE, push_interval_s=(4.0, 2.0)),), "push_interval_s"),
        ((replace(BALANCE, wheel_kick_interval_s=(3.0, 1.0)),), "wheel_kick_interval_s"),
        ((replace(BALANCE, battery_voltage=(12.5, 10.5)),), "battery_voltage"),
        ((replace(BALANCE, rewards=RewardConfig(smoothness_weight=-1.0)),), "smoothness"),
    ],
)
def test_invalid_curricula_are_rejected(curriculum, message):
    with pytest.raises(ValueError, match=message):
        validate(curriculum)


def test_a_new_stage_needs_only_a_list_entry():
    halfway = replace(CURRICULUM[2], name="commands 0.25 m/s", command_max=(0.25, 1.5))
    curriculum = (*CURRICULUM[:3], halfway, *CURRICULUM[3:])
    table = stage_table(curriculum)
    assert len(table["success_rate"]) == len(CURRICULUM) + 1
    np.testing.assert_allclose(table["command_floor"][3], (0.2, 1.25))
    np.testing.assert_allclose(table["command_floor"][4], (0.25, 1.5))
    assert isinstance(curriculum[3], Stage)


def test_training_curriculum_follows_the_randomization_switch():
    """Without randomization: the same stages, commands, rewards and tests, nothing random."""
    assert len(TRAINING_CURRICULUM) == len(CURRICULUM)
    if DOMAIN_RANDOMIZATION:
        assert TRAINING_CURRICULUM == CURRICULUM
        return
    for plain, randomized in zip(TRAINING_CURRICULUM, CURRICULUM):
        assert plain == without_randomization(randomized)
        assert (plain.name, plain.command_max, plain.rewards) == (
            randomized.name,
            randomized.command_max,
            randomized.rewards,
        )
        assert (plain.success_rate, plain.speed_ratio) == (
            randomized.success_rate,
            randomized.speed_ratio,
        )
    table = stage_table(TRAINING_CURRICULUM)
    for column in (
        "slope_max", "push_max", "wheel_kick_max", "com_offset_max", "mass_fraction",
        "friction_fraction", "max_latency_steps", "imu_angle_noise", "imu_rate_noise",
        "encoder_noise",
    ):
        assert np.all(table[column] == 0), column
    assert np.allclose(table["battery_voltage"], 11.1)
