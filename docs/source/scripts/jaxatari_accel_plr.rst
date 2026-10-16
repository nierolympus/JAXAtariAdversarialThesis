JAXAtari ACCEL / PLR with JAXUED
=================================

``scripts/benchmarks/jaxatari_accel_plr.py`` adapts JAXUED's PLR/ACCEL
curriculum algorithm to JAXAtari Pong and Seaquest. It is not a maze generator: in this
runner, a *level* is a fixed-length JAX array that selects adversarial
environment dynamics for a rollout. The default experiment uses a quantized
integer parameter vector; the former Boolean-mask experiment remains
available for direct comparison.

The normal experiment configuration is
``scripts/benchmarks/config/alg/pqn_jaxatari_object_jaxued_accel.yaml``.
The matching Seaquest configuration is
``scripts/benchmarks/config/alg/pqn_jaxatari_object_jaxued_accel_seaquest.yaml``.

Architecture
------------

.. code-block:: text

   JAXUED LevelSampler                JAXAtari Pong wrapper stack
   -------------------                ---------------------------
   stores/prioritizes mask levels --> PongAccelModWrapper
   chooses DR/replay/mutation          -> LogWrapper
                                        -> NormalizeObservationWrapper
                                        -> object or pixel observation wrapper
                                        -> AtariWrapper -> JaxPong

   protagonist rollout: policy action + selected level mask + strength
   -------------------------------------------------------------------->

The protagonist always takes normal Pong actions. The curriculum does not
choose those actions. Instead, it chooses the mask passed to
``PongAccelModWrapper.step(state, action, level_mask, strength)`` on every
step of that rollout.

For Seaquest, ``SeaquestAccelModWrapper`` uses the same protocol with four
pre-physics parameters: player movement speed, enemy speed multiplier, player
torpedo speed, and enemy torpedo speed multiplier. Shared parameter names and
video labels live in ``GameAccelModWrapper`` rather than in the Pong runner.

What comes from JAXUED
----------------------

The implementation intentionally reuses the parts of
``jaxued/examples/maze_plr.py`` that do not depend on a maze representation.

* ``jaxued.level_sampler.LevelSampler`` is the actual level buffer. It stores
  masks, score priorities, timestamps, duplicate handling, and replay
  sampling.
* The ACCEL branch schedule matches Maze PLR:

  * **DR/new**: sample random masks, roll out, score them, and insert/update
    the sampler.
  * **replay**: sample masks from ``LevelSampler`` according to its priority
    and staleness weights, roll out, update their scores, and train the
    protagonist.
  * **repeat**: optionally continue the exact same replay levels and Pong
    states for another rollout chunk, giving the protagonist more than one
    update before mutation.
  * **mutation**: mutate the last replay masks, roll out, score, and insert
    them. This branch occurs after a replay in ACCEL mode.

  Selecting ``JAXUED_ALG: plr`` removes the mutation branch and keeps the
  standard new/replay schedule.
* The runner supports several level scores:

  * dense truncated Q-PVL (``q_pvl``) by default for PQN. It averages the
    positive difference between reward-derived Q(lambda) returns and the
    protagonist's greedy state-value estimate. This is the PQN analogue of
    PLR's positive-value-loss regret proxy, but it does not require a complete
    episode inside the rollout (important for 21-point Pong matches);
  * generic reward learning progress (``env_learning_progress``) as an
    optional environment-only fallback. It stores an exponential moving
    average of raw return for every exact level and prioritizes normalized
    absolute return change;
  * raw environment-return difficulty (``env_return``) as a baseline. The
    summed raw reward is negated so lower protagonist return has higher replay
    priority;
  * ``positive_value_loss`` for ``JAXUED_SCORE_FUNCTION: pvl``;
  * ``compute_max_returns`` and ``max_mc`` for ``max_mc``;
  * ``q_pvl_all_steps`` and ``dense_pvl`` as backward-compatible aliases for
    ``q_pvl``.
* With ``PROTAGONIST_ALG: jaxued_ppo``, the optional recurrent protagonist
  uses JAXUED's ``ResetRNN`` and the same PPO update structure. ``distrax``
  provides its categorical action distribution.
* ``train_state_to_log_dict`` follows the lightweight JAXUED logging idea: it
  copies only sampler summaries to the host rather than the entire level
  buffer.

What is deliberately different from Maze PLR
--------------------------------------------

JAXUED's Maze example operates on ``Maze`` / ``Level`` objects and calls
``reset_to_level``. JAXAtari Pong has neither of those concepts. Its state is
created by the normal environment ``reset``; the level is forwarded through
the observation wrappers to ACCEL inside frame-skip and applied every raw
frame.

Consequently, this runner does **not** use these maze-specific JAXUED pieces:

* ``UnderspecifiedEnv``, ``EnvParams``, ``EnvState``, or Maze observation
  types;
* ``Maze``, ``MazeRenderer``, ``Level``, the maze level generator, or maze
  mutators;
* ``AutoReplayWrapper``;
* Maze rendering/evaluation/video code; JAXAtari's renderer and W&B video
  code are used instead;
* the Maze example's Orbax checkpoint-manager flow. This runner currently
  uses its local optional ``SAVE_PATH`` safetensors flow.

The PQN default is also local: it uses the existing
``pqn_agent_adv_random.QNetwork`` and its lambda-return Q-learning update.
This is intentional so the agent remains compatible with the established
JAXAtari benchmark workflow. The sampler and curriculum schedule are still
JAXUED's algorithm.

Training flow
-------------

#. Build the JAXAtari wrapper stack, then put ``PongAccelModWrapper`` last.
#. Compute the number of mask dimensions and initialise ``LevelSampler`` with
   a zero-mask placeholder.
#. At each update choose DR, replay, or mutation using the JAXUED schedule.
#. Reset before every newly generated, newly sampled replay, or mutated level
   batch. A repeat branch alone carries the same Pong state forward, so one
   trajectory is never scored under multiple ACCEL levels.
#. For every policy step, call the wrapper with that same mask and
   ``ACCEL_STRENGTH``.
#. Compute the configured per-level score. With the PQN default, this is dense
   positive Q(lambda) value error, a truncated PLR regret proxy. Insert or
   update the levels in the sampler, then log branch and sampler statistics.

The runner has ``EXPLORATORY_GRAD_UPDATES: true`` in the regular config. This
is important for sparse-reward Pong: it trains on DR, replay, and mutation
rollouts. Setting it to ``false`` restores the Maze-PLR score-only exploratory
option, where only replay rollouts update the protagonist.

Parameter-vector semantics (default)
------------------------------------

With the default lazy-enemy setting, the parameter-vector encoding stores an
integer vector with twelve dimensions. Disabling ``ACCEL_ENABLE_LAZY_ENEMY``
removes that coordinate and leaves eleven dimensions. Each integer indexes a
finite list of bins, so JAXUED duplicate checking and replay remain exact while
the search space is far larger than a Boolean mask.

The first seven dimensions select independent multipliers for the existing
physics-caused delta targets: ball X/Y position, ball X/Y velocity, player
Y, enemy Y, and player speed. Their configured bins range from 1.0 to 1.25 and
include 1.0 as normal Pong. With ``ACCEL_ENABLE_LAZY_ENEMY: true`` (the
default), the final five dimensions set player paddle height, paddle width,
enemy tracking step, lazy-enemy mode, and enemy move interval before raw Pong
physics. The lazy-enemy dimension reuses the game's existing ``lazy_enemy``
behavior: the enemy stays still while the ball travels away, then tracks it on
the inbound path. Set ``ACCEL_ENABLE_LAZY_ENEMY: false`` to remove this
adversarial action from training levels.

Default bins are height [14, 15, 16], width [2, 3, 4], enemy step
[1, 2, 3, 4], lazy-enemy mode [0, 1], and enemy move interval [0, 2, 4].
Mode 0 is continuous tracking;
mode 1 is the existing ``lazy_enemy`` behavior. Paddle dimensions are
deliberately capped at their base-Pong
sizes (16 px high and 4 px wide), so ACCEL may only retain or shrink them.
The exact base-Pong
vector uses 1.0 multipliers, height 16, width 4, enemy step 2, tracking mode
0, and move interval 0. Interval 0 keeps Pong's native seven-of-eight-frame
enemy schedule; a positive interval moves only on every Nth raw frame.
``ACCEL_ENEMY_MATCH_BALL_SPEED`` can be enabled to align the enemy with the
next position of an incoming ball faster than the base speed, including a ball
whose ACCEL trajectory modifier makes it move faster while retaining its raw
velocity. This preserves a collision-capable response for slow-cadence or
slow-step-size levels.
Random generation includes that base vector with
ACCEL_PARAMETER_BASELINE_PROB. Mutation moves one parameter by one bin, with
an optional second one-bin mutation. The default grid contains
6^7 * 3 * 3 * 4 * 2 * 3 = 60,466,176 exact levels.

The default regular config uses this encoding with a 1,024-entry sampler.
It logs decoded values under accel_parameters/*. In training video overlays,
highlighted panels are dimensions that differ from base Pong; the W&B caption
contains all exact decoded values.

Legacy mask semantics and the twelve Pong actions
--------------------------------------------------

There are twelve mask bits. Their order is stable and is used by
``accel_actions/spec_XX_*`` metrics and the video banner.

.. list-table::
   :header-rows: 1
   :widths: 8 10 24 58

   * - Index
     - Video label
     - Target
     - Effect when active
   * - 0
     - BX
     - ``ball_x``
     - Amplifies the horizontal position delta of the ball. The resulting raw
       Pong movement is capped at ``ACCEL_MAX_BALL_TRAJECTORY_STEP`` (4 by
       default); score/reset jumps are not amplified.
   * - 1
     - BY
     - ``ball_y``
     - Amplifies the vertical ball position delta, including travel after a
       wall bounce, with the same cap.
   * - 2
     - VX
     - ``ball_vel_x``
     - Amplifies physics-caused horizontal velocity changes at paddle hits or
       boosts. The value is clamped to the core Pong range [-4, +4].
   * - 3
     - VY
     - ``ball_vel_y``
     - Amplifies physics-caused vertical deflection changes at paddles and
       walls. The value is clamped to [-4, +4].
   * - 4
     - PY
     - ``player_y``
     - Amplifies player paddle vertical displacement while retaining court
       bounds.
   * - 5
     - EY
     - ``enemy_y``
     - Amplifies the vertical displacement generated by enemy AI tracking.
   * - 6
     - PS
     - ``player_speed``
     - Amplifies analog player acceleration/deceleration, bounded by the
       normal 5.75 px/step paddle-speed limit.
   * - 7
     - SH
     - ``player_paddle_height_shrink``
     - Sets the paddle to 0.5 times its baseline height (16 -> 8 px by
       default). Rendering, observation, movement bounds, and the collision
       hitbox all use this state value.
   * - 8
     - PW
     - ``player_paddle_width``
     - Sets the paddle to 0.5 times its baseline width (4 -> 2 px by default).
       Rendering, object observation, and the horizontal collision interval
       use this value.
   * - 9
     - ES
     - ``enemy_step_size``
     - Sets the enemy AI's tracking step to 2 times baseline (2 -> 4 px per
       enemy update by default).
    * - 10
      - LE
      - ``lazy_enemy``
      - Switches the enemy from continuous tracking to approach-only tracking,
        matching the existing ``lazy_enemy`` behavior.
    * - 11
      - EI
      - ``enemy_move_interval``
      - Replaces the native enemy schedule with movement on every Nth raw
        frame. The parameter-vector value 0 retains native Pong behavior.

Bits 0--6 are **delta actions**. The wrapper compares the state before and
after the normal Pong physics step and applies:

.. code-block:: text

   modified = pre_step_value + strength * (post_step_value - pre_step_value)

The configured value is then clamped. ``max_abs_delta`` additionally caps
motion-style changes and passes through discontinuities such as a Pong point
reset.

Bits 7--10 are **persistent pre-step state actions**. They are reset to their
baseline when inactive and set from that baseline when active, on every step.
They therefore do not compound (for example, a 0.5x paddle never becomes
0.5x squared on the next frame).

``ACCEL_MAX_LEVEL_SIZE: 4`` means that up to four of these ten actions are
active in one sampled mask. There are 562 reachable masks. The dedicated
``pqn_jaxatari_object_jaxued_accel_mask``
configuration uses ``ACCEL_BUFFER_CAPACITY: 512``. The runner also clamps an overlarge capacity
automatically when duplicate checking is enabled.

Physics limits
--------------

``JaxPong`` defines ``MAX_BALL_SPEED = 4`` and now clamps both velocity
components in the core dynamics. This is the authoritative physical velocity
limit. The ACCEL trajectory actions default to the same four-pixel maximum,
so they cannot produce an effective eight-pixel step merely because
``ACCEL_STRENGTH`` is 2. Set ``ACCEL_MAX_BALL_TRAJECTORY_STEP: 5.0`` only for
an intentional faster-Pong experiment.

Logging, evaluation, and videos
--------------------------------

The runner logs JAXUED sampler summaries under ``level_sampler/*`` and branch
counts under ``updates/*``. Protagonist diagnostics include
``agent/gradient_applied``, epsilon, greedy/selected non-no-op fractions,
per-action greedy fractions, and reward incidence. These metrics distinguish
an untrained greedy no-op policy from a curriculum that is merely collecting
score-only rollouts.

Base and prebuilt-mod evaluation use ordinary JAXAtari environments. They do
not have an ACCEL level. The separate ``video/train/accel_*`` videos use the
current sampled level. In mask mode their top banner contains one labelled
panel per mask bit: green with a white underline means active; charcoal means
inactive. In parameter-vector mode each labelled column is a low-to-high
bucket strip: blue marks a baseline selection, amber marks the selected
non-baseline bucket, and a white tick shows the baseline bucket. The W&B
caption lists exact decoded values. Only the
video artifact is stored under ``video/*``; related scalar metadata is under
``video_metrics/*``.

Every periodic evaluation receives a distinct W&B media key, for example
``video/eval_001/base`` and ``video/eval_001/mod_lazy_enemy``. Thus the media
section retains a representative base and modified-environment video at every
evaluation instead of replacing earlier clips at ``video/base``. ``FINAL_EVAL``
defaults to true and produces a separate final representation at
``video/final/base`` and ``video/final/mod_lazy_enemy`` after the last training
update, even when periodic evaluation is disabled.

Running the regular experiment
------------------------------

From the WSL conda environment:

.. code-block:: bash

   JAX_PLATFORMS=cuda \
   PYTHONPATH=./scripts/benchmarks:./src:./jaxued/jaxued/src \
   python scripts/benchmarks/jaxatari_accel_plr.py \
     --config-name=pqn_jaxatari_object_jaxued_accel

For the old Boolean-mask baseline, use
``pqn_jaxatari_object_jaxued_accel_mask`` as the config name instead.

For Seaquest, use:

.. code-block:: bash

   JAX_PLATFORMS=cuda \
   PYTHONPATH=./scripts/benchmarks:./src:./jaxued/jaxued/src \
   python scripts/benchmarks/jaxatari_accel_plr.py \
     --config-name=pqn_jaxatari_object_jaxued_accel_seaquest
