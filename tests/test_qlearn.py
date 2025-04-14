from blkjx.qlearn import observe, QLearn, _deal
from blkjx.blackjack import Game
from blkjx.utils import cards_from_str as c, rank_from_str as r

from beartype.claw import beartype_package
import jax
import jax.numpy as jnp

beartype_package("blkjx")


def test_observe_jits():
    key = jax.random.key(0)
    game = Game.from_deal(key)
    jax.jit(observe)(game)


def test_observe():
    key = jax.random.key(0)
    game = Game.from_hands(key, r("J"), c("68"))
    obs = observe(game)
    assert jnp.array_equal(obs, [10, 34])


def test_observe_batched():
    key = jax.random.key(0)

    games = jax.vmap(Game.from_hands)(
        jax.random.split(key, 5),
        jnp.array([r("J"), r("5"), r("8"), r("A"), r("2")]),
        jnp.array([c("J4"), c("33"), c("A4"), c("A6J"), c("JJ8")]),
    )

    # Force game 0 to be terminated
    games = games.replace(play=games.play.at[0].set(1))

    obs = observe(games)
    exp_obs = jnp.array([
        [0, 0],  # Term
        [4, 3],  # Pair of 3s
        [7, 17], # Soft 15
        [0, 37], # Hard 17
        [1, 42], # Bust
    ])
    assert jnp.array_equal(obs, exp_obs)

def test_qlearn_train_step():
    qlearn = QLearn()

    key = jax.random.key(0)
    games = _deal(key, 4)
    games, replay = qlearn.train_step(key, games, None)