from dataclasses import dataclass, field, replace
from typing import Self, TextIO
from pathlib import Path

import jax
import jax.numpy as jnp

from blkjx.blackjack import Game, _is_pair
from blkjx.utils import (
    Action,
    ActionMask,
    Array,
    Bool,
    Float16,
    PRNGKey,
    Cards,
    RANKS,
    SBool,
    SFloat16,
    SUInt8,
    UInt8,
)


Observation = UInt8[Array, "2"]
"""Observation space for a single hand, `(dealer_up, encoded_hand)`

`dealer_up` is stored as `Rank - 1` since `Rank = 0 = NONE` is not possible.
This only generates "basic strategy" so hands are encoded according to the table below
Hand totals are categorized as either pairs, hard, soft, or bust and encoded as shown in the table
below. The special value `(0, 0)` is used for terminal states, and `Q(s = (0, 0), a) = 0`

category         encoding    encoding range
--------         ----------  --------------
terminal         0           0
pair             rank        [1, 14)
soft (non-pair)  total + 2   [14, 24)
hard (non-pair)  total + 20  [24, 42)
bust             42          42
"""

Support = SFloat16
"""Game rewards encoded for QTable, currently the encoding is identity"""

QTable = Float16[Support, "13 43 5"]
"""Q(s, a)

Table is indexed as `table[s[0], s[1], a]`
"""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ReplayBuffer:
    observation: UInt8[Observation, "*batch"]
    action: UInt8[Array, "*batch"]
    post_observation: UInt8[Observation, "*batch"]
    post_actions: Bool[ActionMask, "*batch"]
    reward: Float16[Array, "*batch"]
    done: Bool[Array, "*batch"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class QLearn:
    alpha: SFloat16 = field(
        default_factory=lambda: jnp.array(0.01, jnp.float16),
        metadata={"static": True},
    )
    """Learning Rate"""

    gamma: SFloat16 = field(
        default_factory=lambda: jnp.array(0.99, jnp.float16),
        metadata={"static": True},
    )
    """Discount Factor"""

    table: QTable = field(default_factory=lambda: jnp.zeros((13, 43, 5), jnp.float16))

    def train_step(
        self, key: PRNGKey, games: Game, log: None | TextIO
    ) -> tuple[Game, ReplayBuffer]:
        observations = observe(games)
        mask = games.legal_actions()[..., 0, :]
        actions = policy(key, self.table, observations, mask)
        games, rewards, dones = jax.vmap(Game.step)(games, actions)
        rewards = rewards[..., 0]
        post_observations = observe(games)
        post_actions = games.legal_actions()[..., 0, :]
        replay = ReplayBuffer(
            observation=observations,
            action=actions,
            post_observation=post_observations,
            post_actions=post_actions,
            reward=rewards,
            done=dones,
        )
        if log:
            jax.debug.callback(_debug_print, games, replay, log)
        games = _reset(games, dones)
        return (games, replay)

    def train_update(self, replay: ReplayBuffer) -> Self:
        obs, post_obs = replay.observation, replay.post_observation
        q = self.table[obs[..., 0], obs[..., 1], replay.action]

        # max_{a'} Q(s', a') but restrict a' to legal actions
        post_q = self.table[post_obs[..., 0], post_obs[..., 1], :]
        post_q = jax.vmap(
            lambda q, q_min, mask: jnp.max(q, initial=q_min, where=mask)
        )(post_q, jnp.min(post_q, axis=-1), replay.post_actions)

        delta = self.alpha * (replay.reward + self.gamma * ~replay.done * post_q - q)
        table = self.table.at[obs[..., 0], obs[..., 1], replay.action].add(delta)
        return replace(self, table=table)

    def train(
        self,
        key: PRNGKey,
        batch_size: int = 1024,
        steps: int = 1024,
        log: None | TextIO = None,
    ) -> Self:
        games = _deal(key, batch_size)
        keys = jax.random.split(key, steps)

        def train_body(i, x: tuple[QLearn, Game, PRNGKey]):
            qlearn, games, keys = x
            games, replay = qlearn.train_step(keys[i], games, log)
            qlearn = qlearn.train_update(replay)
            return qlearn, games, keys

        qlearn, _, _ = jax.lax.fori_loop(0, steps, train_body, (self, games, keys))
        return qlearn

    def replace(self, **kwargs) -> Self:
        return replace(self, **kwargs)

    def save(self, file: str | Path):
        jnp.savez(file, alpha=self.alpha, gamma=self.gamma, table=self.table)

    @staticmethod
    def load(file: str | Path) -> Self:
        data = jnp.load(file)
        return QLearn(alpha=data["alpha"], gamma=data["gamma"], table=data["table"])


def observe(game: Game) -> UInt8[Observation, "*batch"]:
    """Return `Observation`s for a Game or batched Game set"""
    dealer_up = jnp.argmax(game.dealer, axis=-1).astype(jnp.uint8)
    hands = game.hands[..., 0, :]

    term_mask = ~game.done()

    pair_mask = _is_pair(hands)
    encoded_pair = jnp.argmax(hands, axis=-1).astype(jnp.uint8) + 1

    rank_values = jnp.clip(jnp.arange(1, 14, dtype=jnp.int8), max=10)
    hard_total = jnp.dot(hands, rank_values)
    is_soft = (hard_total < 12) & (hands[..., 0] > 0)
    offset = jnp.array(12, jnp.uint8) + ~is_soft * jnp.array(8, jnp.uint8)
    encoded_total = jnp.clip(hard_total + offset, max=42).astype(jnp.uint8)

    encoded = (pair_mask * encoded_pair) + (~pair_mask * encoded_total)
    obs = jnp.concatenate(
        (dealer_up[..., jnp.newaxis], encoded[..., jnp.newaxis]), axis=-1
    )
    return term_mask[..., jnp.newaxis] * obs


def policy(
    key: PRNGKey,
    table: QTable,
    observation: UInt8[Observation, "*batch"],
    mask: Bool[ActionMask, "*batch"],
) -> UInt8[Array, "*batch"]:
    """Choose actions from softmax policy using the QTable"""
    # Because observations are an incomplete representation of the game state
    # An actions legality cannot be determined from the observation alone.
    # It is very important that illegal actions are extremely rarely chosen
    # Otherwise, they will incorrectly contribute significantly to the QValue
    supports = table[observation[..., 0], observation[..., 1], :]
    supports = jnp.where(mask, supports, -20.0)
    dist = 2.0 * supports - jnp.inf * ~mask
    return jax.random.categorical(key, dist).astype(jnp.uint8)


def _deal(key: PRNGKey, batch_size: int) -> Game:
    keys = jax.random.split(key, batch_size)
    games = jax.vmap(Game.from_deal)(keys)
    actions = games.actions.at[..., Action.SPLIT].set(False)
    return games.replace(actions=actions)


def _reset(games: Game, dones: Bool[Array, "*batch"]) -> Game:
    # This always deals a full batch of new games, and only uses some. Consider profiling
    reset_games = jax.vmap(Game.reset)(games)
    reset_actions = reset_games.actions.at[..., Action.SPLIT].set(False)

    return Game(
        key=jnp.where(dones, reset_games.key, games.key),
        play=jnp.where(dones, reset_games.play, games.play),
        split=jnp.where(dones, reset_games.split, games.split),
        deck=jnp.where(dones[..., jnp.newaxis], reset_games.deck, games.deck),
        dealer=jnp.where(dones[..., jnp.newaxis], reset_games.dealer, games.dealer),
        hands=jnp.where(
            dones[..., jnp.newaxis, jnp.newaxis], reset_games.hands, games.hands
        ),
        wager=jnp.where(dones[..., jnp.newaxis], reset_games.wager, games.wager),
        status=jnp.where(dones[..., jnp.newaxis], reset_games.status, games.status),
        actions=jnp.where(
            dones[..., jnp.newaxis, jnp.newaxis], reset_actions, games.actions
        ),
    )


def _debug_print(game: Game, replay: ReplayBuffer, log: TextIO):
    batch_size = len(game.key)
    for i in range(batch_size):
        dealer = _repr_hand(game.dealer[i])
        hands = "".join(_repr_hand(h) for h in game.hands[i])
        actions = _repr_actions(game.actions[i, 0])
        observation = _repr_obs(replay.observation[i])
        action = Action(replay.action[i]).name
        reward = replay.reward[i]
        post_observation = _repr_obs(replay.post_observation[i])
        log.write(
            f"Dealer: {dealer}\n"
            f"Hands: {hands} {game.hands[i, 0]}\n"
            f"Actions: {actions}\n"
            f"Observation: {observation}\n"
            f"Action: {action}\n"
            f"Reward: {reward}\n"
            f"Post Observation: {post_observation}\n"
            f"Done: {replay.done}\n"
            f"\n"
        )


def _repr_hand(cards: UInt8) -> str:
    return "".join(n * c for n, c in zip(cards, RANKS))


def _repr_actions(mask: ActionMask):
    return [Action(i).name for i, v in enumerate(mask) if v]


def _repr_obs(obs: Observation) -> str:
    dealer = RANKS[obs[0]]

    if obs[1] == 0:
        return "term"
    elif obs[1] < 14:
        hand = f"pair {RANKS[obs[1] - 1]}"
    elif obs[1] < 24:
        hand = f"soft {obs[1] - 2}"
    elif obs[1] < 42:
        hand = f"hard {obs[1] - 20}"
    else:
        hand = "bust"

    return f"{dealer} {hand}"
