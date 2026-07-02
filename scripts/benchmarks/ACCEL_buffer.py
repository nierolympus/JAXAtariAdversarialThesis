"""
ACCEL-style level buffer for adversarial environment manipulation via a
static-per-episode SET of mod actions (no antagonist).

A "level" is a frozenset of mod-action indices that stay active for the
WHOLE episode. How a level is actually applied to the env is the
wrapper's job (see accel_dual_mode_wrapper.py), not this module's -- this
file is only about which actions are in the set, sampling levels by
regret, and mutating sets (add/remove one action).

NO-OP CONVENTION (corrected from an earlier draft): there is no reserved
"no-op" index. mod_action_space_n is simply the number of real mod specs
(0-indexed, 0..mod_action_space_n-1, all real). The EMPTY set already
means "no modification" for free -- nothing needs to be reserved for it.
This matches the wrapper side, where an all-False level_mask is a true
no-op by construction.

Regret is approximated via `RegretTracker` (no antagonist); swap-out
point clearly marked below.
"""

from __future__ import annotations

import dataclasses
from typing import FrozenSet, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Level representation
# ---------------------------------------------------------------------------

# A level is a frozenset of mod-action indices, all 0-indexed real specs.
# frozenset() (empty) = no modification at all.
Level = FrozenSet[int]
ConflictGroup = FrozenSet[int]

EMPTY_LEVEL: Level = frozenset()


@dataclasses.dataclass
class LevelEntry:
    level: Level
    regret: float
    n_sampled: int = 0


# ---------------------------------------------------------------------------
# Regret proxy (NO antagonist): positive value loss
# ---------------------------------------------------------------------------
#
# The earlier version of this tracker used `running_max_reward -
# episode_reward`, a single GLOBAL scalar shared across every level ever
# played. That has two real problems:
#
#   1. NOT COMPARABLE ACROSS TIME: running_max only ever increases, so
#      regret scores computed early in training (when running_max is
#      still small) are systematically smaller than scores computed late
#      in training for an EQUALLY hard level -- yet maybe_insert() directly
#      compares regret values recorded at different times. Comparing
#      apples picked at different points in a moving orchard.
#   2. THE IMPOSSIBLE-LEVEL TRAP: a level the agent can never solve has
#      episode_reward pinned near its floor forever, so
#      running_max - episode_reward stays large forever -- exactly the
#      failure mode the regret signal is supposed to avoid (see the
#      D-PAIRED writeup's warning about impossible environments).
#
# ACCEL/PLR's actual proxy when there's no antagonist is POSITIVE VALUE
# LOSS: how much the agent's own value estimate underestimated the
# (lambda-)return it actually achieved on THIS level, in THIS round only.
# No cross-round global state, so nothing goes stale:
#
#       regret(level) = mean_t( max(0, lambda_target_t - chosen_q_t) )
#
# computed over one round's full NUM_STEPS rollout for that level (in
# original time order, BEFORE minibatch shuffling), using values your PQN
# loop already produces: chosen_q_t = the Q-value the network predicted
# for the action actually taken (see Transition.q_val, indexed by the
# chosen action), and lambda_target_t = the lambda-return target computed
# in _get_target. This is the Q-learning analogue of GAE-based positive
# value loss used in the original PPO-based PLR/ACCEL.
#
# Why max(0, ...) and not the raw signed gap: a level the agent already
# correctly judges as hopeless has the network UNDERpredicting its own
# (low) achieved return... no -- more precisely, on a truly impossible
# level the agent's value function, once trained even briefly, correctly
# predicts a low return, so target ~= q and the gap is ~0 either way. The
# real protection is structural: clipping at 0 means OVERestimation
# (target < q) never contributes positive regret, so a level where the
# agent already (correctly or not) believes things are fine is never
# treated as a regret source -- only genuine "I was too pessimistic, this
# was learnable" moments count.
#
# No antagonist, no running ceiling, no EMA -- the score is intrinsic to
# the (level, current agent) pair at the moment it's computed, which is
# exactly the property ACCEL relies on when it re-evaluates a level every
# time it's resampled as a parent.


def positive_value_loss_regret(chosen_qvals, lambda_targets) -> float:
    """
    Args:
        chosen_qvals: array, shape (NUM_STEPS, NUM_ENVS) -- the Q-value
            predicted for the action actually taken, time-ordered, BEFORE
            any minibatch shuffling. This is `transitions.q_val` indexed
            by `transitions.action` (see integration notes below for the
            exact gather, since Transition.q_val stores ALL action
            q-values, not just the chosen one).
        lambda_targets: array, shape (NUM_STEPS, NUM_ENVS) -- your
            existing `lambda_targets` array, same time ordering.

    Returns a single Python float: mean positive value loss over the
    whole rollout, i.e. the regret score for whichever level was active
    during this round.
    """
    import jax.numpy as jnp

    per_step = jnp.maximum(0.0, lambda_targets - chosen_qvals)
    return float(jnp.mean(per_step))


# ---------------------------------------------------------------------------
# Level buffer
# ---------------------------------------------------------------------------


class ACCELLevelBuffer:
    """Fixed-capacity buffer of (level, regret) pairs with regret-weighted
    sampling and add/remove mutation, with optional conflict-group aware
    mutation so semantically opposite actions (e.g. increase/decrease the
    same parameter) never both end up in the same level.
    """

    def __init__(
            self,
            capacity: int,
            mod_action_space_n: int,
            max_level_size: int,
            rng: np.random.Generator,
            replacement_margin: float = 0.0,
            conflict_groups: Optional[Sequence[ConflictGroup]] = None,
    ):
        """
        Args:
            capacity: max number of levels kept in the buffer.
            mod_action_space_n: total number of distinct mod specs
                available (i.e. len(mod_specs)). ALL indices 0..n-1 are
                real specs -- there is no reserved no-op slot. The empty
                level (frozenset()) is the no-op.
            max_level_size: cap on simultaneous mod actions per level.
            replacement_margin: a child level must beat the buffer's
                current minimum regret by at least this margin to be
                inserted once the buffer is full. 0.0 = any improvement
                qualifies.
            conflict_groups: optional list of action-index sets that are
                mutually exclusive (e.g. {increase_ball_speed_x,
                decrease_ball_speed_x}). Mutation will never let a level
                contain two members of the same group -- adding one
                evicts the other from that level automatically. If None,
                no conflict checking is performed.
        """
        self.capacity = capacity
        self.mod_action_space_n = mod_action_space_n
        self.max_level_size = max_level_size
        self.rng = rng
        self.replacement_margin = replacement_margin
        self.conflict_groups: List[ConflictGroup] = (
            list(conflict_groups) if conflict_groups else []
        )
        self.entries: List[LevelEntry] = []

    # -- conflict resolution -----------------------------------------------

    def _group_of(self, action: int) -> Optional[ConflictGroup]:
        for group in self.conflict_groups:
            if action in group:
                return group
        return None

    def _add_with_conflict_resolution(self, level: Level, new_action: int) -> Level:
        """Return level + new_action, first removing any existing member
        of `level` that shares a conflict group with `new_action`."""
        group = self._group_of(new_action)
        if group is not None:
            level = frozenset(a for a in level if a not in group)
        return frozenset(level | {new_action})

    # -- initialization ------------------------------------------------

    def seed_random(self, n: int) -> None:
        """Populate the buffer with n random small, conflict-free levels
        (including possibly the empty/no-mod level)."""
        for _ in range(n):
            level = self._random_level()
            self.entries.append(LevelEntry(level=level, regret=0.0))

    def _random_level(self) -> Level:
        size = int(self.rng.integers(0, self.max_level_size + 1))
        level: Level = EMPTY_LEVEL
        candidates = list(range(self.mod_action_space_n))  # ALL indices are real specs now
        self.rng.shuffle(candidates)
        for a in candidates:
            if len(level) >= size:
                break
            group = self._group_of(a)
            if group is not None and any(a2 in group for a2 in level):
                continue  # skip, would conflict with something already added
            level = frozenset(level | {a})
        return level

    # -- sampling ------------------------------------------------------

    def sample(self, temperature: float = 1.0) -> LevelEntry:
        """Regret-weighted sampling. Higher temperature -> more uniform;
        lower -> more greedy towards high-regret levels."""
        if not self.entries:
            raise RuntimeError("Buffer is empty; call seed_random() first.")
        regrets = np.array([e.regret for e in self.entries], dtype=np.float64)
        regrets = regrets - regrets.min()  # shift non-negative
        if temperature <= 0:
            idx = int(np.argmax(regrets))
        else:
            logits = regrets / max(temperature, 1e-6)
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs = probs / probs.sum()
            idx = int(self.rng.choice(len(self.entries), p=probs))
        self.entries[idx].n_sampled += 1
        return self.entries[idx]

    # -- mutation --------------------------------------------------------

    def mutate(self, parent: Level) -> Level:
        """Add or remove a single mod action from the parent level,
        respecting conflict groups when adding.

        Direction choice:
            - if parent is empty -> must add
            - if parent is at max_level_size -> must remove
            - otherwise -> 50/50 add vs remove
        """
        can_add = len(parent) < self.max_level_size and len(parent) < self.mod_action_space_n
        can_remove = len(parent) > 0

        if can_add and can_remove:
            do_add = self.rng.random() < 0.5
        elif can_add:
            do_add = True
        elif can_remove:
            do_add = False
        else:
            return parent  # degenerate: nothing possible

        if do_add:
            available = [a for a in range(self.mod_action_space_n) if a not in parent]
            if not available:
                return parent
            new_action = int(self.rng.choice(available))
            return self._add_with_conflict_resolution(parent, new_action)
        else:
            to_remove = int(self.rng.choice(list(parent)))
            return frozenset(parent - {to_remove})

    # -- insertion -------------------------------------------------------

    def maybe_insert(self, level: Level, regret: float) -> bool:
        """Insert/update a (level, regret) pair. Returns whether stored."""
        for e in self.entries:
            if e.level == level:
                e.regret = regret  # refresh with latest estimate
                return True

        if len(self.entries) < self.capacity:
            self.entries.append(LevelEntry(level=level, regret=regret))
            return True

        worst_idx = int(np.argmin([e.regret for e in self.entries]))
        worst_regret = self.entries[worst_idx].regret
        if regret > worst_regret + self.replacement_margin:
            self.entries[worst_idx] = LevelEntry(level=level, regret=regret)
            return True
        return False

    # -- introspection ---------------------------------------------------

    def stats(self) -> dict:
        if not self.entries:
            return {"buffer_size": 0}
        regrets = [e.regret for e in self.entries]
        sizes = [len(e.level) for e in self.entries]
        return {
            "buffer_size": len(self.entries),
            "buffer_mean_regret": float(np.mean(regrets)),
            "buffer_max_regret": float(np.max(regrets)),
            "buffer_mean_level_size": float(np.mean(sizes)),
            "buffer_max_level_size": int(np.max(sizes)),
        }


# ---------------------------------------------------------------------------
# Turning a Level (a SET) into a fixed-shape array for jit/vmap
# ---------------------------------------------------------------------------


def level_to_action_sequence(level: Level) -> Tuple[int, ...]:
    """Deterministic, ascending-index ordering of a level's actions.
    Mainly useful for logging/debugging -- the wrapper itself consumes
    the mask form below, not this sequence."""
    return tuple(sorted(level))


def level_to_mask(level: Level, mod_action_space_n: int) -> np.ndarray:
    """Bool array, shape (mod_action_space_n,). mask[i] = True iff spec i
    is active in this level. All indices 0..n-1 are real specs; no
    reserved no-op slot. The all-False mask (empty level) is the no-op."""
    mask = np.zeros((mod_action_space_n,), dtype=bool)
    for a in level:
        mask[a] = True
    return mask