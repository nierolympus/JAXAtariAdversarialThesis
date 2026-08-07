JAXAtari ACCEL / PLR with JAXUED
=================================

``scripts/benchmarks/jaxatari_accel_plr.py`` adapts JAXUED's PLR/ACCEL
curriculum algorithm to JAXAtari Pong. It is not a maze generator: in this
runner, a *level* is a fixed-length JAX array that selects adversarial
environment dynamics for a rollout. The default experiment uses a quantized
integer parameter vector; the former Boolean-mask experiment remains
available for direct comparison.

The normal experiment configuration is
``scripts/benchmarks/config/alg/pqn_jaxatari_object_jaxued_accel.yaml``.

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
  * **mutation**: mutate the last replay masks, roll out, score, and insert
    them. This branch occurs after a replay in ACCEL mode.

  Selecting ``JAXUED_ALG: plr`` removes the mutation branch and keeps the
  standard new/replay schedule.
* The runner uses JAXUED score utilities:

  * ``positive_value_loss`` for ``JAXUED_SCORE_FUNCTION: pvl``;
  * ``compute_max_returns`` and ``max_mc`` for ``max_mc``;
  * a dense local Q-value PVL proxy (``q_pvl_all_steps``) by default. It is
    useful for Atari because a short rollout window does not necessarily end
    an episode.
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
created by the normal environment ``reset``; the mask is supplied to the
outer ACCEL wrapper on every step.

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
#. Reset a batch of ordinary Pong states and hold one sampled mask per
   environment fixed during that rollout.
#. For every policy step, call the wrapper with that same mask and
   ``ACCEL_STRENGTH``.
#. Compute per-mask UED scores, insert or update the sampler, then log the
   branch and sampler statistics.

The runner has ``EXPLORATORY_GRAD_UPDATES: true`` in the regular config. This
is important for sparse-reward Pong: it trains on DR, replay, and mutation
rollouts. Setting it to ``false`` restores the Maze-PLR score-only exploratory
option, where only replay rollouts update the protagonist.

Parameter-vector semantics (default)
------------------------------------

The parameter-vector encoding stores an integer vector with ten dimensions.
Each integer indexes a finite list of bins, so JAXUED duplicate checking and
replay remain exact while the search space is far larger than a Boolean mask.

The first seven dimensions select independent multipliers for the existing
physics-caused delta targets: ball X/Y position, ball X/Y velocity, player
Y, enemy Y, and player speed. Their default bins range from 0.5 to 2.5 and
include 1.0 as normal Pong. The final three dimensions set player paddle
height, paddle width, and enemy tracking step before raw Pong physics.

Default bins are height [8, 10, 12, 14, 16, 20, 24, 28, 32], width
[2, 3, 4, 5, 6, 7, 8], and enemy step [1, 2, 3, 4]. The exact base-Pong
vector uses 1.0 multipliers, height 16, width 4, and enemy step 2.
Random generation includes that base vector with
ACCEL_PARAMETER_BASELINE_PROB. Mutation moves one parameter by one bin, with
an optional second one-bin mutation. The default grid contains
9^7 * 9 * 7 * 4 = 1,205,308,188 exact levels.

The default regular config uses this encoding with a 512-entry sampler.
It logs decoded values under accel_parameters/*. In training video overlays,
highlighted panels are dimensions that differ from base Pong; the W&B caption
contains all exact decoded values.

Legacy mask semantics and the eleven Pong actions
--------------------------------------------------

There are eleven mask bits. Their order is stable and is used by
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
     - PH
     - ``player_paddle_height_grow``
     - Sets the paddle to 1.75 times its baseline height (16 -> 28 px by
       default). Rendering, object observation, movement bounds, and the
       collision hitbox all use this state value.
   * - 8
     - SH
     - ``player_paddle_height_shrink``
     - Sets the paddle to 0.5 times its baseline height (16 -> 8 px by
       default). It has the same rendering, observation, movement-bound, and
       collision effects as PH, but is mutually exclusive with PH.
   * - 9
     - PW
     - ``player_paddle_width``
     - Sets the paddle to 1.5 times its baseline width (4 -> 6 px by default).
       Rendering, object observation, and the horizontal collision interval
       use this value.
   * - 10
     - ES
     - ``enemy_step_size``
     - Sets the enemy AI's tracking step to 2 times baseline (2 -> 4 px per
       enemy update by default).

Bits 0--6 are **delta actions**. The wrapper compares the state before and
after the normal Pong physics step and applies:

.. code-block:: text

   modified = pre_step_value + strength * (post_step_value - pre_step_value)

The configured value is then clamped. ``max_abs_delta`` additionally caps
motion-style changes and passes through discontinuities such as a Pong point
reset.

Bits 7--10 are **persistent pre-step state actions**. They are reset to their
baseline when inactive and set from that baseline when active, on every step.
They therefore do not compound (for example, a 1.75x paddle never becomes
1.75x squared on the next frame).

``ACCEL_MAX_LEVEL_SIZE: 2`` means that up to two of these eleven actions are
active in one sampled mask. It does not mean that only two actions exist.
PH and SH are mutually exclusive: random sampling and mutation never produce
a mask that tries to grow and shrink the paddle simultaneously. There are
``C(11,0) + C(11,1) + C(11,2) - 1 = 66`` reachable masks. The dedicated
``pqn_jaxatari_object_jaxued_accel_mask`` configuration uses
``ACCEL_BUFFER_CAPACITY: 66``. The runner also clamps an overlarge capacity
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
inactive. In parameter-vector mode highlighted panels differ from base Pong
and the W&B caption lists exact decoded values. Only the
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
