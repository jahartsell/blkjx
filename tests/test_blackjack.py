from contextlib import ExitStack
from unittest.mock import patch

from beartype.claw import beartype_package
import jax
import jax.numpy as jnp

from blkjx.blackjack import Game, _cards_total
from blkjx.utils import (
    cards_from_str as c,
    rank_from_str as r,
    RANKS,
    Status,
    Action,
    ActionMask,
)

beartype_package("blkjx")


class PatchDraws:
    """Context manager to disable jit and monkey patch which cards will be drawn from a deck"""

    def __init__(self, cards: str):
        self.stack = ExitStack()
        self.cards = [jnp.array(RANKS.find(c), jnp.uint8) for c in cards]

    def __enter__(self):
        # Mocking uses side effects, which requires jit to be disabled
        self.stack.enter_context(jax.disable_jit())
        mock = self.stack.enter_context(patch("blkjx.blackjack._choice"))
        mock.side_effect = self.cards
        return mock

    def __exit__(self, exc_type, exc_value, traceback):
        return self.stack.__exit__(exc_type, exc_value, traceback)


def actions_equal(mask: ActionMask, expected: list[Action]):
    exp_mask = jnp.zeros(len(Action), jnp.bool)
    for i in expected:
        exp_mask = exp_mask.at[i].set(True)
    return jnp.array_equal(mask, exp_mask)


def test_totals():
    cards = jnp.zeros((3, 13), jnp.uint8)
    cards = cards.at[0].set(c("4K"))
    cards = cards.at[1].set(c("AJ"))
    total = _cards_total(cards)
    assert jnp.array_equal(total, [14, 21, 0])

def test_game_from_hands():
    key = jax.random.key(1234)
    game = Game.from_hands(key, r("4"), c("4K"))
    exp_deck = jnp.array(
        [24, 24, 24, 22, 24, 24, 24, 24, 24, 24, 24, 24, 23], jnp.uint8
    )

    assert jnp.array_equal(game.key, key)
    assert game.play == 0
    assert game.split == 1
    assert jnp.array_equal(game.deck, exp_deck)
    assert jnp.array_equal(game.dealer, c("4"))
    assert jnp.array_equal(game.hands[0], c("4K"))
    assert jnp.all(game.hands[1:] == 0)
    assert game.wager[0] == 1
    assert jnp.all(game.wager[1:] == 0)
    assert game.status[0] == Status.PLAY
    assert jnp.all(game.status[1:] == Status.EMPTY)
    assert game.actions[0, Action.HIT] == Status.PLAY
    assert jnp.all(game.status[1:] == Status.EMPTY)
    assert actions_equal(
        game.actions[0],
        [
            Action.HIT,
            Action.STAND,
            Action.DOUBLE,
            Action.SURRENDER,
        ],
    )
    assert jnp.all(game.actions[1:] == 0)


def test_game_from_hands_splits_pairs():
    key = jax.random.key(0)
    game = Game.from_hands(key, r("4"), c("66"))
    assert game.actions[0, Action.SPLIT] == 1


def test_game_from_hands_no_surrender_when_dealer_has_ace():
    key = jax.random.key(0)
    game = Game.from_hands(key, r("A"), c("66"))
    assert game.actions[0, Action.SURRENDER] == 0


def test_game_from_deal():
    key = jax.random.key(0)

    with PatchDraws("A3T"):
        game = Game.from_deal(key)

    assert jnp.array_equal(game.dealer, c("T"))
    assert jnp.array_equal(game.hands[0], c("A3"))


def test_game_hit_jits():
    key = jax.random.key(0)
    game = Game.from_deal(key)
    jax.jit(Game._action_hit)(game)


def test_game_hit_when_doesnt_bust():
    key = jax.random.key(0)
    game = Game.from_hands(key, r("T"), c("66"))

    with PatchDraws("3"):
        game = game._action_hit()

    assert jnp.array_equal(game.hands[0], c("663"))
    assert game.status[0] == Status.PLAY
    assert actions_equal(
        game.actions[0],
        [
            Action.HIT,
            Action.STAND,
            Action.DOUBLE,
            Action.SURRENDER,
        ],
    )


def test_game_hit_when_21():
    key = jax.random.key(0)
    game = Game.from_hands(key, r("T"), c("Q6"))

    # Player draws 5 and stands
    with PatchDraws("5"):
        game = game._action_hit()

    assert jnp.array_equal(game.hands[0], c("Q65"))
    assert game.status[0] == Status.STAND


def test_game_hit_when_busts():
    key = jax.random.key(0)
    game = Game.from_hands(key, r("T"), c("Q6"))

    # Player draws 7 and busts
    with PatchDraws("7"):
        game = game._action_hit()

    assert jnp.array_equal(game.hands[0], c("Q67"))
    assert game.status[0] == Status.BUST


def test_game_stand_jits():
    key = jax.random.key(0)
    game = Game.from_deal(key)
    jax.jit(Game._action_stand)(game)


def test_game_stand():
    key = jax.random.key(0)
    game = Game.from_hands(key, r("T"), c("Q6"))
    game = game._action_stand()

    assert jnp.array_equal(game.hands[0], c("Q6"))
    assert game.status[0] == Status.STAND


def test_game_double_jits():
    key = jax.random.key(0)
    game = Game.from_deal(key)
    jax.jit(Game._action_double)(game)


def test_game_double_when_doesnt_bust():
    key = jax.random.key(0)
    game = Game.from_hands(key, r("T"), c("J6"))

    with PatchDraws("3"):
        game = game._action_double()

    assert jnp.array_equal(game.hands[0], c("J63"))
    assert game.status[0] == Status.PLAY
    assert game.wager[0] == 2
    assert actions_equal(
        game.actions[0],
        [
            Action.STAND,
            Action.DOUBLE,
        ],
    )


def test_game_double_when_21():
    key = jax.random.key(0)
    game = Game.from_hands(key, r("T"), c("Q6"))

    with PatchDraws("5"):
        game = game._action_double()

    assert jnp.array_equal(game.hands[0], c("Q65"))
    assert game.status[0] == Status.STAND
    assert game.wager[0] == 2


def test_game_double_when_busts():
    key = jax.random.key(0)
    game = Game.from_hands(key, r("T"), c("Q6"))

    with PatchDraws("7"):
        game = game._action_double()

    assert jnp.array_equal(game.hands[0], c("Q67"))
    assert game.status[0] == Status.BUST
    assert game.wager[0] == 2


def test_game_double_atmost_twice():
    key = jax.random.key(0)
    game = Game.from_hands(key, r("T"), c("56"))

    with PatchDraws("34"):
        game = game._action_double()
        game = game._action_double()

    assert jnp.array_equal(game.hands[0], c("5634"))
    assert game.status[0] == Status.PLAY
    assert game.wager[0] == 3
    assert actions_equal(game.actions[0], [Action.STAND])


def test_game_split_jits():
    key = jax.random.key(0)
    game = Game.from_hands(key, r("T"), c("66"))
    jax.jit(Game._action_split)(game)


def test_game_split():
    key = jax.random.key(0)
    game = Game.from_hands(key, r("T"), c("66"))

    with PatchDraws("76"):
        game = game._action_split()

    assert game.split == 2

    # Hand 1 draws a 7, cannot split again
    assert jnp.array_equal(game.hands[0], c("67"))
    assert game.status[0] == Status.PLAY
    assert game.wager[0] == 1
    assert actions_equal(
        game.actions[0],
        [
            Action.HIT,
            Action.STAND,
            Action.DOUBLE,
        ],
    )

    # Hand 2 draws a 6, can split again
    assert jnp.array_equal(game.hands[1], c("66"))
    assert game.status[1] == Status.PLAY
    assert game.wager[1] == 1
    assert actions_equal(
        game.actions[1],
        [
            Action.HIT,
            Action.STAND,
            Action.SPLIT,
            Action.DOUBLE,
        ],
    )


def test_game_split_first_21_stands():
    key = jax.random.key(0)
    game = Game.from_hands(key, r("T"), c("JJ"))

    with PatchDraws("A6"):
        game = game._action_split()

    assert game.split == 2

    assert jnp.array_equal(game.hands[0], c("JA"))
    assert game.status[0] == Status.STAND

    assert jnp.array_equal(game.hands[1], c("J6"))
    assert game.status[1] == Status.PLAY


def test_game_split_split_21_plays():
    key = jax.random.key(0)
    game = Game.from_hands(key, r("T"), c("JJ"))

    with PatchDraws("6A"):
        game = game._action_split()

    assert game.split == 2

    assert jnp.array_equal(game.hands[0], c("J6"))
    assert game.status[0] == Status.PLAY

    assert jnp.array_equal(game.hands[1], c("JA"))
    assert game.status[1] == Status.PLAY


def test_game_split_aces():
    key = jax.random.key(0)
    game = Game.from_hands(key, r("T"), c("AA"))

    with PatchDraws("AA"):
        game = game._action_split()

    assert game.split == 2

    # Splitting Aces cannot hit, double, or resplit
    assert jnp.array_equal(game.hands[0], c("AA"))
    assert game.status[0] == Status.PLAY
    assert game.wager[0] == 1
    assert actions_equal(game.actions[0], [Action.STAND])

    assert jnp.array_equal(game.hands[1], c("AA"))
    assert game.status[0] == Status.PLAY
    assert game.wager[0] == 1
    assert actions_equal(game.actions[1], [Action.STAND])


def test_game_surrender_jits():
    key = jax.random.key(0)
    game = Game.from_hands(key, r("T"), c("66"))
    jax.jit(Game._action_surrender)(game)


def test_game_surrender():
    key = jax.random.key(0)
    game = Game.from_hands(key, r("T"), c("J6"))
    game = game._action_surrender()

    assert game.status[0] == Status.SURRENDER

def test_game_score_when_win():
    key = jax.random.key(0)
    game = Game.from_deal(key)

    dealer = c("J6")
    hands = game.hands.at[0].set(c("A34"))
    game = game.replace(dealer=dealer, hands=hands)

    assert jnp.array_equal(game._score(),  [1., 0., 0., 0., 0.])

def test_game_score_when_lose():
    key = jax.random.key(0)
    game = Game.from_deal(key)

    dealer = c("J8")
    hands = game.hands.at[0].set(c("A33"))
    game = game.replace(dealer=dealer, hands=hands)

    assert jnp.array_equal(game._score(),  [-1., 0., 0., 0., 0.])

def test_game_score_when_push():
    key = jax.random.key(0)
    game = Game.from_deal(key)

    dealer = c("J8")
    hands = game.hands.at[0].set(c("A7"))
    game = game.replace(dealer=dealer, hands=hands)

    assert jnp.array_equal(game._score(),  [0., 0., 0., 0., 0.])

def test_game_score_when_blackjack():
    key = jax.random.key(0)
    game = Game.from_deal(key)

    dealer = c("J8")
    hands = game.hands.at[0].set(c("AJ"))
    game = game.replace(dealer=dealer, hands=hands)

    assert jnp.array_equal(game._score(),  [1.5, 0., 0., 0., 0.])

def test_game_score_when_both_blackjack_dealer_3_cards():
    key = jax.random.key(0)
    game = Game.from_deal(key)

    dealer = c("J83")
    hands = game.hands.at[0].set(c("AJ"))
    game = game.replace(dealer=dealer, hands=hands)

    assert jnp.array_equal(game._score(),  [1.5, 0., 0., 0., 0.])

def test_game_score_when_both_blackjack():
    key = jax.random.key(0)
    game = Game.from_deal(key)

    dealer = c("JA")
    hands = game.hands.at[0].set(c("AJ"))
    game = game.replace(dealer=dealer, hands=hands)

    assert jnp.array_equal(game._score(),  [0., 0., 0., 0., 0.])

def test_game_score_when_surrender():
    key = jax.random.key(0)
    game = Game.from_deal(key)

    dealer = c("J8")
    hands = game.hands.at[0].set(c("A9"))
    status = game.status.at[0].set(Status.SURRENDER)
    game = game.replace(dealer=dealer, hands=hands, status=status)

    assert jnp.array_equal(game._score(),  [-.5, 0., 0., 0., 0.])

def test_game_score_when_bust():
    key = jax.random.key(0)
    game = Game.from_deal(key)

    dealer = c("8A")
    hands = game.hands.at[0].set(c("88J"))
    game = game.replace(dealer=dealer, hands=hands)

    assert jnp.array_equal(game._score(),  [-1., 0., 0., 0., 0.])

def test_game_score_when_both_bust():
    key = jax.random.key(0)
    game = Game.from_deal(key)

    dealer = c("887")
    hands = game.hands.at[0].set(c("88J"))
    game = game.replace(dealer=dealer, hands=hands)

    assert jnp.array_equal(game._score(),  [-1., 0., 0., 0., 0.])

def test_step_jits():
    key = jax.random.key(0)
    game = Game.from_hands(key, r("A"), c("87"))

    jax.jit(Game.step)(game, Action.HIT)

def test_step():
    key = jax.random.key(0)
    game = Game.from_hands(key, r("A"), c("87"))

    with PatchDraws("286"):
        game, reward, done = game.step(Action.HIT) # Draw 2
        assert not done
        assert jnp.array_equal(reward, [0., 0., 0., 0., 0.])

        game, reward, done = game.step(Action.HIT) # Draw 8 and bust

    assert done
    assert jnp.array_equal(reward,  [-1., 0., 0., 0., 0.])