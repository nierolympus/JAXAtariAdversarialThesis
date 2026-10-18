"""Reusable game-specific wrapper definitions for ACCEL training."""

from __future__ import annotations

from functools import partial
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.wrappers import (
    AccelDeltaModWrapper,
    AtariWrapper,
    GenericModSpec,
    JaxatariWrapper,
    LevelStateModSpec,
    _get_core_env_state,
    _has_wrapper_in_chain,
    _set_core_env_state,
    _unwrap_env_chain,
)


class GameAccelModWrapper(AccelDeltaModWrapper):
    """Game-neutral ACCEL metadata used by training metrics and videos.

    The physics/state operations still come from ``AccelDeltaModWrapper``.
    This layer lets each game provide declarative specs, bins, metric names,
    and compact video labels without adding game-specific lists to a runner.
    """

    def __init__(
        self,
        env,
        *,
        game_name: str,
        parameter_names: Sequence[str],
        short_labels: Sequence[str],
        mod_specs: Sequence[GenericModSpec] = (),
        pre_step_specs: Sequence[LevelStateModSpec] = (),
        level_encoding: str = "mask",
        parameter_values: Sequence[Sequence[float]] | None = None,
    ):
        self._accel_game_name = str(game_name)
        self._accel_parameter_names = tuple(str(name) for name in parameter_names)
        self._accel_short_labels = tuple(str(label) for label in short_labels)
        super().__init__(
            env,
            mod_specs=mod_specs,
            pre_step_specs=pre_step_specs,
            level_encoding=level_encoding,
            parameter_values=parameter_values,
        )
        expected = len(self._mod_specs)
        if len(self._accel_parameter_names) != expected:
            raise ValueError(
                f"{self._accel_game_name} has {expected} ACCEL dimensions but "
                f"{len(self._accel_parameter_names)} parameter names."
            )
        if len(self._accel_short_labels) != expected:
            raise ValueError(
                f"{self._accel_game_name} has {expected} ACCEL dimensions but "
                f"{len(self._accel_short_labels)} short labels."
            )

    @property
    def accel_game_name(self) -> str:
        return self._accel_game_name

    @property
    def accel_parameter_names(self) -> tuple[str, ...]:
        return self._accel_parameter_names

    @property
    def accel_short_labels(self) -> tuple[str, ...]:
        return self._accel_short_labels


class PongAccelModWrapper(GameAccelModWrapper):
    """Pong ACCEL with mask or discrete parameter-vector levels.

    The first seven specs amplify a physics-caused state delta every step.  In
    particular, ``ball_x`` and ``ball_y`` make the ball visibly travel faster;
    their amplified movement is capped at the game's physical ball-speed
    limit. Parameter levels then set paddle height, paddle width, and enemy
    tracking behavior directly; legacy masks expose shrink-only paddle
    morphology bits.
    """

    def __init__(
        self,
        env,
        # Kept for source compatibility with older experiment configs.  Paddle
        # growth is deliberately no longer an ACCEL action: morphology levels
        # may only retain or reduce the original paddle dimensions.
        player_paddle_height_multiplier: float = 1.0,
        player_paddle_shrink_multiplier: float = 0.5,
        player_paddle_width_multiplier: float = 0.5,
        enemy_step_multiplier: float = 2.0,
        enemy_move_interval: int = 4,
        enemy_match_ball_speed: bool = False,
        randomize_serve_direction: bool = False,
        max_ball_trajectory_step: float = 4.0,
        level_encoding: str = "mask",
        parameter_delta_multipliers: Sequence[float] = (
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5,
        ),
        parameter_paddle_heights: Sequence[float] = (8, 10, 12, 14, 16),
        parameter_paddle_widths: Sequence[float] = (2, 3, 4),
        parameter_enemy_step_sizes: Sequence[float] = (1, 2, 3, 4),
        parameter_enemy_tracking_modes: Sequence[float] = (0, 1),
        parameter_enemy_move_intervals: Sequence[float] = (0, 2, 4),
        enable_lazy_enemy: bool = True,
        enable_enemy_move_interval: bool = True,
    ):
        # Validate the legacy morphology multipliers at construction time.
        player_paddle_height_multiplier = float(player_paddle_height_multiplier)
        player_paddle_shrink_multiplier = float(player_paddle_shrink_multiplier)
        player_paddle_width_multiplier = float(player_paddle_width_multiplier)
        enemy_move_interval = float(enemy_move_interval)
        if player_paddle_height_multiplier > 1.0:
            raise ValueError(
                "ACCEL_PLAYER_PADDLE_HEIGHT_MULTIPLIER may not exceed 1.0; "
                "ACCEL paddle morphology can only shrink from its original size."
            )
        if player_paddle_shrink_multiplier > 1.0 or player_paddle_width_multiplier > 1.0:
            raise ValueError(
                "ACCEL paddle size multipliers may not exceed 1.0; paddle "
                "morphology can only shrink from its original size."
            )
        if (
            not enemy_move_interval.is_integer()
            or not 0 <= enemy_move_interval <= 8
        ):
            raise ValueError(
                "ACCEL_ENEMY_MOVE_INTERVAL must be an integer from 0 through 8."
            )
        enemy_move_interval = int(enemy_move_interval)
        enable_lazy_enemy = bool(enable_lazy_enemy)
        self._max_ball_trajectory_step = float(max_ball_trajectory_step)

        parameter_paddle_heights = tuple(float(v) for v in parameter_paddle_heights)
        parameter_paddle_widths = tuple(float(v) for v in parameter_paddle_widths)
        parameter_enemy_tracking_modes = tuple(
            float(v) for v in parameter_enemy_tracking_modes
        )
        parameter_enemy_move_intervals = tuple(
            float(v) for v in parameter_enemy_move_intervals
        )
        if not parameter_paddle_heights:
            raise ValueError(
                "ACCEL_PARAMETER_PADDLE_HEIGHTS must be non-empty."
            )
        if not parameter_paddle_widths or any(
            value > 4 for value in parameter_paddle_widths
        ):
            raise ValueError(
                "ACCEL_PARAMETER_PADDLE_WIDTHS must be non-empty and no larger "
                "than the original 4 px paddle width."
            )
        if enable_lazy_enemy and (
            not parameter_enemy_tracking_modes
            or any(value not in (0.0, 1.0) for value in parameter_enemy_tracking_modes)
        ):
            raise ValueError(
                "ACCEL_PARAMETER_ENEMY_TRACKING_MODES must contain only 0 "
                "(normal) and/or 1 (lazy_enemy)."
            )
        if enable_enemy_move_interval and (
            not parameter_enemy_move_intervals
            or any(
                not value.is_integer() or not 0 <= value <= 8
                for value in parameter_enemy_move_intervals
            )
        ):
            raise ValueError(
                "ACCEL_PARAMETER_ENEMY_MOVE_INTERVALS must contain integer "
                "cadences from 0 through 8."
            )
        delta_specs = [
            # BX: amplify the horizontal ball displacement each raw Pong step.
            # The cap prevents ACCEL from moving the ball faster than 4 px/step.
            GenericModSpec(
                target="ball_x",
                min_value=0,
                max_value=160,
                max_abs_delta=max_ball_trajectory_step,
                dtype=jnp.int32,
            ),
            # BY: amplify vertical ball displacement, including wall-bounce
            # travel, with the same physical 4 px/step cap.
            GenericModSpec(
                target="ball_y",
                min_value=24,
                max_value=194,
                max_abs_delta=max_ball_trajectory_step,
                dtype=jnp.int32,
            ),
            # VX: amplify changes to horizontal ball velocity caused by paddle
            # hits/boosts; the velocity itself remains in [-4, +4].
            GenericModSpec(
                target="ball_vel_x",
                op="mul",
                value=10,
                min_const="MIN_BALL_SPEED",
                max_const="MAX_BALL_SPEED",
                preserve_sign=True,
                dtype=jnp.float32,
            ),
            # VY: amplify vertical deflection changes caused by a paddle or
            # wall bounce; its magnitude is also bounded by the Pong limit.
            GenericModSpec(
                target="ball_vel_y",
                op="mul",
                value=1,
                min_const="MIN_BALL_SPEED",
                max_const="MAX_BALL_SPEED",
                preserve_sign=True,
                dtype=jnp.float32,
            ),
            # PY: amplify the player's physics-generated vertical displacement
            # while keeping the paddle inside the court.
            GenericModSpec(
                target="player_y",
                min_const="PADDLE_MIN_Y",
                max_value=190,
                max_abs_delta=5.75,
                dtype=jnp.float32,
            ),
            # EY: amplify the enemy AI paddle's vertical tracking movement,
            # capped at four pixels per raw step.
            GenericModSpec(
                target="enemy_y",
                min_value=24,
                max_value=190,
                max_abs_delta=4,
                dtype=jnp.int32,
            ),
            # PS: amplify the player's analog velocity change. This changes
            # acceleration/deceleration, not the paddle's collision geometry.
            GenericModSpec(
                target="player_speed",
                min_value=-5.75,
                max_value=5.75,
                dtype=jnp.float32,
            ),
        ]
        morphology_specs = [
            # SH: before every raw Pong physics step, replace the normal 16 px
            # player height with 16 * 0.5 = 8 px. ACCEL never enlarges the
            # paddle, so an active morphology level is never easier than base
            # Pong for this dimension.
            LevelStateModSpec(
                target="player_paddle_height",
                label="player_paddle_height_shrink",
                baseline_const="PLAYER_PADDLE_BASE_HEIGHT",
                op="mul",
                value=player_paddle_shrink_multiplier,
                min_const="PLAYER_PADDLE_MIN_HEIGHT",
                max_const="PLAYER_PADDLE_MAX_HEIGHT",
                dtype=jnp.int32,
            ),
            # PW: before every raw Pong physics step, replace the normal 4 px
            # paddle width with at most its original size. This changes the rendered
            # width, object observation width, and horizontal hit interval.
            LevelStateModSpec(
                target="player_paddle_width",
                baseline_const="PLAYER_PADDLE_BASE_WIDTH",
                op="mul",
                value=player_paddle_width_multiplier,
                min_const="PLAYER_PADDLE_MIN_WIDTH",
                max_const="PLAYER_PADDLE_MAX_WIDTH",
                dtype=jnp.int32,
            ),
            # ES: before each raw Pong step, replace the enemy's normal 2 px
            # AI tracking increment with 2 * 2 = 4 px. The enemy still moves
            # on its normal schedule (seven of every eight raw frames); each
            # scheduled movement is simply larger.
            LevelStateModSpec(
                target="enemy_step_size",
                baseline_const="ENEMY_STEP_SIZE",
                op="mul",
                value=enemy_step_multiplier,
                min_const="ENEMY_STEP_MIN_SIZE",
                max_const="ENEMY_STEP_MAX_SIZE",
                dtype=jnp.int32,
            ),
        ]
        if enable_lazy_enemy:
            # LE: the existing Pong lazy_enemy behavior.  While the ball moves
            # away from the enemy, it stays still; it only tracks the inbound
            # ball.  This is a sampled ACCEL adversarial action, not a global
            # training-environment setting.
            morphology_specs.append(
                LevelStateModSpec(
                    target="enemy_tracking_mode",
                    label="lazy_enemy",
                    baseline_const="ENEMY_TRACKING_MODE",
                    op="set",
                    value=1,
                    min_const="ENEMY_TRACKING_MODE_MIN",
                    max_const="ENEMY_TRACKING_MODE_MAX",
                    dtype=jnp.int32,
                )
            )
        # EI: make the enemy track on a different cadence. Zero is the native
        # seven-of-eight-frame schedule; a positive value moves only on every
        # Nth raw frame. Parameter levels set this value directly.
        if enable_enemy_move_interval:
            morphology_specs.append(
                LevelStateModSpec(
                    target="enemy_move_interval",
                    baseline_const="ENEMY_MOVE_INTERVAL",
                    op="set",
                    value=enemy_move_interval,
                    min_const="ENEMY_MOVE_INTERVAL_MIN",
                    max_const="ENEMY_MOVE_INTERVAL_MAX",
                    dtype=jnp.int32,
                )
            )
        parameter_values = [
            *([parameter_delta_multipliers] * len(delta_specs)),
            parameter_paddle_heights,
            parameter_paddle_widths,
            parameter_enemy_step_sizes,
        ]
        if enable_lazy_enemy:
            parameter_values.append(parameter_enemy_tracking_modes)
        if enable_enemy_move_interval:
            parameter_values.append(parameter_enemy_move_intervals)

        parameter_names = [
            "ball_x_multiplier",
            "ball_y_multiplier",
            "ball_vel_x_multiplier",
            "ball_vel_y_multiplier",
            "player_y_multiplier",
            "enemy_y_multiplier",
            "player_speed_multiplier",
            "player_paddle_height",
            "player_paddle_width",
            "enemy_step_size",
        ]
        short_labels = ["BX", "BY", "VX", "VY", "PY", "EY", "PS", "PH", "PW", "ES"]
        if enable_lazy_enemy:
            parameter_names.append("enemy_tracking_mode")
            short_labels.append("LE")
        if enable_enemy_move_interval:
            parameter_names.append("enemy_move_interval")
            short_labels.append("EI")
        super().__init__(
            env,
            game_name="pong",
            parameter_names=parameter_names,
            short_labels=short_labels,
            mod_specs=delta_specs,
            pre_step_specs=morphology_specs,
            level_encoding=level_encoding,
            # The seven delta dimensions independently scale the raw
            # physics-caused deltas. The remaining pre-step dimensions set
            # absolute values before physics: paddle geometry, enemy step,
            # optional lazy-enemy mode, and move cadence.
            parameter_values=parameter_values,
        )
        # Apply wrapper-only gameplay options before the first JITted step.
        # Their defaults keep normal Pong bit-for-bit intact.
        self._core_env.consts = self._core_env.consts.replace(
            ENEMY_MATCH_BALL_SPEED=bool(enemy_match_ball_speed),
            RANDOMIZE_SERVE_DIRECTION=bool(randomize_serve_direction),
        )

    def _set_enemy_response_trajectory(self, state, ball_x_multiplier, ball_y_multiplier):
        """Expose ACCEL's effective ball movement to the core enemy AI.

        The normal delta modifiers run after raw Pong physics, so modifying
        ``ball_x`` or ``ball_y`` does not update ``ball_vel_*``.  Record the
        same rounded, capped delta that the wrapper will apply, allowing
        ``ENEMY_MATCH_BALL_SPEED`` to override a slow enemy whenever the
        trajectory is genuinely faster without changing ball physics.
        """
        core_state = _get_core_env_state(state)
        max_step = jnp.asarray(self._max_ball_trajectory_step, dtype=jnp.float32)

        def accelerated_delta(position, velocity, multiplier):
            # This mirrors AccelDeltaModWrapper's integer position update for
            # normal ball motion.  A point/reset jump is intentionally left
            # unscaled, matching _compile_delta_spec's max_abs_delta guard.
            raw_delta = (
                (position + velocity).astype(jnp.int32) - position
            ).astype(jnp.float32)
            scaled_delta = jnp.clip(
                jnp.rint(raw_delta * multiplier),
                -max_step,
                max_step,
            )
            return jnp.where(jnp.abs(raw_delta) <= max_step, scaled_delta, raw_delta)

        response_ball_vel_x = accelerated_delta(
            core_state.ball_x,
            core_state.ball_vel_x,
            ball_x_multiplier,
        )
        response_ball_vel_y = accelerated_delta(
            core_state.ball_y,
            core_state.ball_vel_y,
            ball_y_multiplier,
        )
        uses_accel_trajectory = jnp.logical_or(
            jnp.not_equal(ball_x_multiplier, 1.0),
            jnp.not_equal(ball_y_multiplier, 1.0),
        )
        core_state = core_state.replace(
            enemy_response_ball_vel_x=response_ball_vel_x,
            enemy_response_ball_vel_y=response_ball_vel_y,
            enemy_response_uses_accel_trajectory=uses_accel_trajectory,
        )
        return _set_core_env_state(state, core_state)

    @partial(jax.jit, static_argnums=(0,))
    def _apply_pre_step_mods(self, state, level_mask, strength=1.0):
        state = super()._apply_pre_step_mods(state, level_mask, strength)
        ball_x_multiplier = jnp.where(level_mask[0], strength, 1.0)
        ball_y_multiplier = jnp.where(level_mask[1], strength, 1.0)
        return self._set_enemy_response_trajectory(
            state,
            ball_x_multiplier,
            ball_y_multiplier,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _apply_pre_step_parameters(self, state, parameter_values):
        state = super()._apply_pre_step_parameters(state, parameter_values)
        return self._set_enemy_response_trajectory(
            state,
            parameter_values[0],
            parameter_values[1],
        )


class FreewayAccelModWrapper(GameAccelModWrapper):
    """Freeway ACCEL levels for global car speed and collision geometry.

    All three dimensions are installed before raw Freeway physics. Speed is a
    phase-advance multiplier consumed by the lane cadence logic; width and
    height are used consistently by wrapping, collisions, observations, and
    rendering. Their neutral values reproduce unmodified Freeway.
    """

    def __init__(
        self,
        env,
        level_encoding: str = "mask",
        parameter_car_speed_multipliers: Sequence[float] = (1.0, 1.5, 2.0),
        parameter_car_widths: Sequence[float] = (8, 10, 12),
        parameter_car_heights: Sequence[float] = (10, 12, 14),
    ):
        speed_values = tuple(float(value) for value in parameter_car_speed_multipliers)
        width_values = tuple(float(value) for value in parameter_car_widths)
        height_values = tuple(float(value) for value in parameter_car_heights)
        parameter_groups = (
            (
                "ACCEL_PARAMETER_CAR_SPEED_MULTIPLIERS",
                speed_values,
                1.0,
                0.5,
                2.0,
                False,
            ),
            (
                "ACCEL_PARAMETER_CAR_WIDTHS",
                width_values,
                8.0,
                4.0,
                16.0,
                True,
            ),
            (
                "ACCEL_PARAMETER_CAR_HEIGHTS",
                height_values,
                10.0,
                6.0,
                14.0,
                True,
            ),
        )
        for config_name, values, baseline, minimum, maximum, integer_only in parameter_groups:
            if not values:
                raise ValueError(f"{config_name} must contain at least one value.")
            if integer_only and any(not value.is_integer() for value in values):
                raise ValueError(f"{config_name} must contain integer pixel sizes.")
            if any(not minimum <= value <= maximum for value in values):
                raise ValueError(
                    f"{config_name} values must be in [{minimum}, {maximum}]."
                )
            if baseline not in values:
                raise ValueError(
                    f"{config_name} must include neutral Freeway value {baseline:g}."
                )

        dynamics_specs = [
            LevelStateModSpec(
                target="car_speed_multiplier",
                baseline_const="car_speed_multiplier",
                op="set",
                value=2.0,
                min_const="car_speed_multiplier_min",
                max_const="car_speed_multiplier_max",
                dtype=jnp.float32,
            ),
            LevelStateModSpec(
                target="car_width",
                baseline_const="car_width",
                op="set",
                value=12,
                min_const="car_width_min",
                max_const="car_width_max",
                dtype=jnp.int32,
            ),
            LevelStateModSpec(
                target="car_height",
                baseline_const="car_height",
                op="set",
                value=14,
                min_const="car_height_min",
                max_const="car_height_max",
                dtype=jnp.int32,
            ),
        ]
        super().__init__(
            env,
            game_name="freeway",
            parameter_names=(
                "car_speed_multiplier",
                "car_width",
                "car_height",
            ),
            short_labels=("CS", "CW", "CH"),
            pre_step_specs=dynamics_specs,
            level_encoding=level_encoding,
            parameter_values=(speed_values, width_values, height_values),
        )


class FrostbiteAccelModWrapper(GameAccelModWrapper):
    """Frostbite ACCEL levels expressed as native-speed multipliers.

    Frostbite stores movement as whole pixels plus sixteenths of a pixel and
    recomputes those values whenever the game advances a level.  Replacing the
    raw state fields with fixed values would therefore destroy normal level
    scaling and make the nominal baseline diverge from the base game.  This
    adapter instead asks the core game for the current level's native speeds on
    every raw frame, scales those fixed-point values, and writes the resulting
    whole/fraction pairs before physics runs.

    The three ``GenericModSpec`` instances below are level-coordinate
    descriptors only.  Their normal post-step delta path is disabled because
    the corresponding dynamics must be installed before Frostbite physics.
    """

    def __init__(
        self,
        env,
        level_encoding: str = "parameter_vector",
        parameter_ice_speed_multipliers: Sequence[float] = (1.0, 1.5, 2.0),
        parameter_obstacle_speed_multipliers: Sequence[float] = (1.0, 1.5, 2.0),
        parameter_player_speed_multipliers: Sequence[float] = (0.75, 1.0),
    ):
        parameter_groups = (
            (
                "ACCEL_PARAMETER_ICE_SPEED_MULTIPLIERS",
                tuple(float(value) for value in parameter_ice_speed_multipliers),
                0.0,
                3.0,
            ),
            (
                "ACCEL_PARAMETER_OBSTACLE_SPEED_MULTIPLIERS",
                tuple(
                    float(value)
                    for value in parameter_obstacle_speed_multipliers
                ),
                0.5,
                3.0,
            ),
            (
                "ACCEL_PARAMETER_PLAYER_SPEED_MULTIPLIERS",
                tuple(float(value) for value in parameter_player_speed_multipliers),
                0.5,
                1.5,
            ),
        )
        for config_name, values, minimum, maximum in parameter_groups:
            if not values:
                raise ValueError(f"{config_name} must contain at least one value.")
            if any(not minimum <= value <= maximum for value in values):
                raise ValueError(
                    f"{config_name} values must be in [{minimum}, {maximum}]."
                )
            if 1.0 not in values:
                raise ValueError(
                    f"{config_name} must include neutral Frostbite value 1.0."
                )

        speed_specs = (
            GenericModSpec(target="ice_speed_frac", dtype=jnp.int32),
            GenericModSpec(target="obstacle_speed_frac", dtype=jnp.int32),
            GenericModSpec(target="bailey_speed_frac", dtype=jnp.int32),
        )
        super().__init__(
            env,
            game_name="frostbite",
            parameter_names=(
                "ice_speed_multiplier",
                "obstacle_speed_multiplier",
                "player_speed_multiplier",
            ),
            short_labels=("IS", "OS", "PS"),
            mod_specs=speed_specs,
            level_encoding=level_encoding,
            parameter_values=[group[1] for group in parameter_groups],
        )

    @staticmethod
    def _scale_fixed_point_speed(whole, fraction, multiplier):
        """Scale a non-negative 4-bit fixed-point speed and split it again."""
        raw_speed = (
            jnp.asarray(whole, dtype=jnp.int32) * 16
            + jnp.asarray(fraction, dtype=jnp.int32)
        )
        scaled_speed = jnp.maximum(
            0,
            jnp.rint(raw_speed.astype(jnp.float32) * multiplier).astype(jnp.int32),
        )
        return scaled_speed // 16, scaled_speed % 16

    def _apply_speed_multipliers(
        self,
        state,
        ice_multiplier,
        obstacle_multiplier,
        player_multiplier,
    ):
        core_state = _get_core_env_state(state)
        native = self._core_env._calculate_speeds_for_level(core_state.level)
        ice_whole, ice_frac = self._scale_fixed_point_speed(
            native["ice_speed_whole"],
            native["ice_speed_frac"],
            ice_multiplier,
        )
        obstacle_whole, obstacle_frac = self._scale_fixed_point_speed(
            native["obstacle_speed_whole"],
            native["obstacle_speed_frac"],
            obstacle_multiplier,
        )
        jump_whole, jump_frac = self._scale_fixed_point_speed(
            native["bailey_jump_speed_whole"],
            native["bailey_jump_speed_frac"],
            player_multiplier,
        )
        walk_whole, walk_frac = self._scale_fixed_point_speed(
            native["bailey_walk_speed_whole"],
            native["bailey_walk_speed_frac"],
            player_multiplier,
        )
        core_state = core_state.replace(
            ice_speed_whole=ice_whole,
            ice_speed_frac=ice_frac,
            obstacle_speed_whole=obstacle_whole,
            obstacle_speed_frac=obstacle_frac,
            bailey_jump_speed_whole=jump_whole,
            bailey_jump_speed_frac=jump_frac,
            bailey_speed_whole=walk_whole,
            bailey_speed_frac=walk_frac,
        )
        return _set_core_env_state(state, core_state)

    @partial(jax.jit, static_argnums=(0,))
    def _apply_pre_step_mods(self, state, level_mask, strength=1.0):
        multipliers = jnp.where(
            jnp.asarray(level_mask, dtype=bool),
            jnp.asarray(strength, dtype=jnp.float32),
            1.0,
        )
        return self._apply_speed_multipliers(state, *multipliers)

    @partial(jax.jit, static_argnums=(0,))
    def _apply_pre_step_parameters(self, state, parameter_values):
        return self._apply_speed_multipliers(state, *parameter_values)

    @partial(jax.jit, static_argnums=(0,))
    def _apply_level_deltas(self, pre_state, post_state, level_mask, strength):
        del pre_state, level_mask, strength
        return post_state

    @partial(jax.jit, static_argnums=(0,))
    def _apply_parameter_deltas(self, pre_state, post_state, parameter_values):
        del pre_state, parameter_values
        return post_state


class StructuredLevelAccelWrapper(JaxatariWrapper):
    """Reusable ACCEL lifecycle for first-class structured level PyTrees.

    A compatible core game exposes ``get_structured_level()`` and
    ``apply_structured_level(state, level)``.  The wrapper owns reset/step
    integration, while a small game adapter only has to define how levels are
    sampled and mutated.  Existing mask and parameter-vector wrappers do not
    use this path.
    """

    uses_structured_levels = True
    _level_encoding = "structured"

    def __init__(self, env, *, game_name: str):
        super().__init__(env)
        self._action_space = self._env.action_space()
        self._core_env = _unwrap_env_chain(self._env)
        if not _has_wrapper_in_chain(self._env, AtariWrapper):
            raise ValueError(
                "StructuredLevelAccelWrapper must directly wrap a chain "
                "containing AtariWrapper."
            )
        for method_name in ("get_structured_level", "apply_structured_level"):
            if not hasattr(self._core_env, method_name):
                raise TypeError(
                    f"{type(self._core_env).__name__} does not implement the "
                    f"structured-level hook {method_name}()."
                )
        self._accel_game_name = str(game_name)
        self._baseline_level = self._core_env.get_structured_level()

    def action_space(self):
        return self._action_space

    @property
    def accel_game_name(self) -> str:
        return self._accel_game_name

    @property
    def baseline_level(self):
        return self._baseline_level

    def _apply_level(self, state: Any, level: Any):
        core_state = _get_core_env_state(state)
        if core_state is None:
            raise TypeError("Could not locate AtariState in the wrapped state tree.")
        core_state = self._core_env.apply_structured_level(core_state, level)
        return _set_core_env_state(state, core_state)

    def _observation(self, state: Any):
        return self._core_env._get_observation(_get_core_env_state(state))

    def reset(self, key, level=None, strength=1.0):
        """Reset the game and install ``level`` before returning its observation."""
        del strength
        _, state = self._env.reset(key)
        selected_level = self._baseline_level if level is None else level
        state = self._apply_level(state, selected_level)
        return self._observation(state), state

    def step(self, state, action, level, strength=1.0):
        """Keep the selected layout installed across deaths and stage changes."""
        del strength
        state = self._apply_level(state, level)
        _, state, reward, terminated, truncated, info = self._env.step(state, action)
        state = self._apply_level(state, level)
        return self._observation(state), state, reward, terminated, truncated, info

    def sample_random_level(self, rng):
        raise NotImplementedError

    def mutate_level(self, rng, level):
        raise NotImplementedError

    def level_metrics(self, levels) -> dict[str, jax.Array]:
        del levels
        return {}

    def describe_level(self, level) -> str:
        del level
        return f"{self._accel_game_name} structured level"


class KangarooAccelWrapper(StructuredLevelAccelWrapper):
    """Kangaroo curriculum that moves stable collectible positions.

    An edit shifts one fruit or bell horizontally in one of the three stages.
    Vertical positions and collision geometry stay canonical, and moving
    entities remain ordinary runtime state. This changes the object layout
    without making the platform/ladder route itself invalid.
    """

    def __init__(
        self,
        env,
        *,
        initial_edits: int = 3,
        mutation_edits: int = 1,
        object_step: int = 8,
    ):
        super().__init__(env, game_name="kangaroo")
        self._initial_edits = int(initial_edits)
        self._mutation_edits = int(mutation_edits)
        self._object_step = int(object_step)
        self._object_min_x = int(self._core_env.consts.LEFT_CLIP)
        self._fruit_max_x = int(
            self._core_env.consts.RIGHT_CLIP - self._core_env.consts.FRUIT_WIDTH
        )
        self._bell_max_x = int(
            self._core_env.consts.RIGHT_CLIP - self._core_env.consts.BELL_WIDTH
        )
        if self._initial_edits < 0:
            raise ValueError("ACCEL_KANGAROO_INITIAL_EDITS must be non-negative.")
        if self._mutation_edits < 1:
            raise ValueError("ACCEL_KANGAROO_MUTATION_EDITS must be at least 1.")
        if not 0 < self._object_step <= self._fruit_max_x - self._object_min_x:
            raise ValueError(
                "ACCEL_KANGAROO_OBJECT_STEP must be positive and fit inside "
                "the playable horizontal range."
            )

    def _mutate_once(self, rng, level):
        selection_key, direction_key = jax.random.split(rng)
        # Three fruits plus one bell in each of the three stages.
        selection = jax.random.randint(selection_key, (), 0, 3 * 4)
        stage_index = selection // 4
        object_index = selection % 4
        is_bell = object_index == 3
        fruit_index = jnp.minimum(object_index, 2)

        old_fruit_x = level.fruit_positions[stage_index, fruit_index, 0]
        old_bell_x = level.bell_positions[stage_index, 0]
        old_x = jnp.where(is_bell, old_bell_x, old_fruit_x)
        max_x = jnp.where(is_bell, self._bell_max_x, self._fruit_max_x)

        direction = jnp.where(jax.random.bernoulli(direction_key), 1, -1)
        candidate = old_x + direction * self._object_step
        candidate = jnp.where(
            candidate < self._object_min_x,
            old_x + self._object_step,
            candidate,
        )
        candidate = jnp.where(
            candidate > max_x,
            old_x - self._object_step,
            candidate,
        )
        candidate = jnp.clip(candidate, self._object_min_x, max_x)

        fruit_positions = level.fruit_positions.at[
            stage_index, fruit_index, 0
        ].set(jnp.where(is_bell, old_fruit_x, candidate))
        bell_positions = level.bell_positions.at[stage_index, 0].set(
            jnp.where(is_bell, candidate, old_bell_x)
        )
        return level._replace(
            fruit_positions=fruit_positions,
            bell_positions=bell_positions,
        )

    def _apply_edits(self, rng, level, num_edits):
        def edit_once(_, carry):
            key, current_level = carry
            key, edit_key = jax.random.split(key)
            return key, self._mutate_once(edit_key, current_level)

        return jax.lax.fori_loop(0, num_edits, edit_once, (rng, level))[1]

    def sample_random_level(self, rng):
        count_key, edit_key = jax.random.split(rng)
        num_edits = jax.random.randint(
            count_key, (), 0, self._initial_edits + 1
        )
        return self._apply_edits(edit_key, self._baseline_level, num_edits)

    def mutate_level(self, rng, level):
        return self._apply_edits(rng, level, self._mutation_edits)

    def level_metrics(self, levels) -> dict[str, jax.Array]:
        fruit_distances = jnp.abs(
            levels.fruit_positions[..., 0]
            - self._baseline_level.fruit_positions[None, ..., 0]
        )
        bell_distances = jnp.abs(
            levels.bell_positions[..., 0]
            - self._baseline_level.bell_positions[None, ..., 0]
        )
        stage_distances = jnp.concatenate(
            [fruit_distances, bell_distances[..., None]], axis=-1
        )
        metrics = {
            "kangaroo_layout/changed_objects_mean": jnp.mean(
                jnp.sum(stage_distances != 0, axis=(-2, -1))
            ),
            "kangaroo_layout/object_x_shift_mean": jnp.mean(stage_distances),
            "kangaroo_layout/object_x_shift_max": jnp.max(stage_distances),
            "kangaroo_layout/fruit_x_shift_mean": jnp.mean(fruit_distances),
            "kangaroo_layout/bell_x_shift_mean": jnp.mean(bell_distances),
        }
        metrics.update(
            {
                f"kangaroo_layout/stage_{stage_index + 1}_x_shift_mean": jnp.mean(
                    stage_distances[:, stage_index]
                )
                for stage_index in range(3)
            }
        )
        return metrics

    def describe_level(self, level) -> str:
        fruit_x = np.asarray(jax.device_get(level.fruit_positions[..., 0]))
        bell_x = np.asarray(jax.device_get(level.bell_positions[..., 0]))
        baseline_fruit_x = np.asarray(
            jax.device_get(self._baseline_level.fruit_positions[..., 0])
        )
        baseline_bell_x = np.asarray(
            jax.device_get(self._baseline_level.bell_positions[..., 0])
        )
        return (
            "Kangaroo object layout | fruit x by stage="
            f"{fruit_x.tolist()} | bell x={bell_x.tolist()} | "
            f"base fruit={baseline_fruit_x.tolist()} | "
            f"base bell={baseline_bell_x.tolist()}"
        )


class SeaquestAccelModWrapper(GameAccelModWrapper):
    """Seaquest ACCEL levels that set velocity before core game physics.

    Seaquest stores several entities in packed ``[x, y, direction]`` arrays,
    so post-step whole-array delta scaling would also corrupt direction flags.
    These four dimensions instead set explicit scalar dynamics fields consumed
    by the game before movement and collision handling. Their neutral values
    reproduce the untouched ``seaquest_default`` implementation.
    """

    def __init__(
        self,
        env,
        level_encoding: str = "mask",
        parameter_player_move_speeds: Sequence[float] = (1, 2),
        parameter_enemy_speed_multipliers: Sequence[float] = (1, 2),
        parameter_player_missile_speeds: Sequence[float] = (3, 4, 5, 6),
        parameter_enemy_missile_speed_multipliers: Sequence[float] = (1, 2),
    ):
        parameter_groups = (
            (
                "ACCEL_PARAMETER_PLAYER_MOVE_SPEEDS",
                tuple(float(value) for value in parameter_player_move_speeds),
                1,
                1,
                3,
            ),
            (
                "ACCEL_PARAMETER_ENEMY_SPEED_MULTIPLIERS",
                tuple(float(value) for value in parameter_enemy_speed_multipliers),
                1,
                1,
                3,
            ),
            (
                "ACCEL_PARAMETER_PLAYER_MISSILE_SPEEDS",
                tuple(float(value) for value in parameter_player_missile_speeds),
                5,
                1,
                8,
            ),
            (
                "ACCEL_PARAMETER_ENEMY_MISSILE_SPEED_MULTIPLIERS",
                tuple(
                    float(value)
                    for value in parameter_enemy_missile_speed_multipliers
                ),
                1,
                1,
                3,
            ),
        )
        for config_name, values, baseline, minimum, maximum in parameter_groups:
            if not values:
                raise ValueError(f"{config_name} must contain at least one value.")
            if any(not value.is_integer() for value in values):
                raise ValueError(f"{config_name} must contain integer pixel speeds.")
            if any(not minimum <= value <= maximum for value in values):
                raise ValueError(
                    f"{config_name} values must be in [{minimum}, {maximum}]."
                )
            if float(baseline) not in values:
                raise ValueError(
                    f"{config_name} must include neutral Seaquest value {baseline}."
                )

        velocity_specs = [
            LevelStateModSpec(
                target="player_move_speed",
                baseline_const="PLAYER_MOVE_SPEED",
                op="set",
                value=2,
                min_const="PLAYER_MOVE_SPEED_MIN",
                max_const="PLAYER_MOVE_SPEED_MAX",
                dtype=jnp.int32,
            ),
            LevelStateModSpec(
                target="enemy_speed_multiplier",
                baseline_const="ENEMY_SPEED_MULTIPLIER",
                op="set",
                value=2,
                min_const="ENEMY_SPEED_MULTIPLIER_MIN",
                max_const="ENEMY_SPEED_MULTIPLIER_MAX",
                dtype=jnp.int32,
            ),
            LevelStateModSpec(
                target="player_missile_speed",
                baseline_const="PLAYER_MISSILE_SPEED",
                op="set",
                value=3,
                min_const="PLAYER_MISSILE_SPEED_MIN",
                max_const="PLAYER_MISSILE_SPEED_MAX",
                dtype=jnp.int32,
            ),
            LevelStateModSpec(
                target="enemy_missile_speed_multiplier",
                baseline_const="ENEMY_MISSILE_SPEED_MULTIPLIER",
                op="set",
                value=2,
                min_const="ENEMY_MISSILE_SPEED_MULTIPLIER_MIN",
                max_const="ENEMY_MISSILE_SPEED_MULTIPLIER_MAX",
                dtype=jnp.int32,
            ),
        ]
        super().__init__(
            env,
            game_name="seaquest",
            parameter_names=(
                "player_move_speed",
                "enemy_speed_multiplier",
                "player_missile_speed",
                "enemy_missile_speed_multiplier",
            ),
            short_labels=("PM", "ES", "PT", "ET"),
            pre_step_specs=velocity_specs,
            level_encoding=level_encoding,
            parameter_values=[group[1] for group in parameter_groups],
        )
