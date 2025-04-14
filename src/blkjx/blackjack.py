"""Blackjack game environment"""

from dataclasses import dataclass
from typing import ClassVar, Self

import jax
import jax.numpy as jnp

from blkjx.utils import (
    Action,
    ActionMask,
    Array,
    Cards,
    PRNGKey,
    Rank,
    Status,
    SBool,
    SUInt8,
    UInt8,
    Float16,
    Bool,
)


type Reward = Float16[Array, "5"]
"""Reward value for a `Game` step, signed earnings or losses per hand"""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Game:
    """Gamestate for blackjack

    The game is implemented with the following rules
        - 6 decks (configurable), shuffled between every hand
        - Dealer stands on 17 (including soft)
        - "Blackjack" payout is 3:2 and happens when a player gets 21 with the first 2 cards of the
          initial hand. If the dealer also gets 21 with 2 cards it is a push, otherwise player wins.
        - Player may surrender and recover half their bet at any time, unless
            * The dealer is showing an ace
            * The player has split or doubled
        - Player may double, increasing the wager by 1 and hitting. After doubling, the hand may not
          hit.
        - Player may redouble a doubled hand, increasing the wager by 1 and hitting. After
          redoubling the hand either busts or stands, no other actions permitted.
        - Player may split in a hands first action if the hand is a pair.
            * A player may split up to 4 times, for 5 total hands.
            * A pair of Aces may only be split in the first hand and cannot hit or double
    """

    NDECKS: ClassVar[int] = 6
    """Number of decks in use"""

    NHANDS: ClassVar[int] = 5
    """Maximum number of player hands, split limit + 1"""

    key: PRNGKey
    """Random state for the game"""

    play: SUInt8
    """Index of the hand to play.
    Hands are played in order, all hands with `index < play` are finished and all hands with
    `index > play` are awaiting play. The game is finished when `play == split`.
    """

    split: SUInt8
    """Index of the next empty slot in hands, or `NHANDS` if all spaces are full"""

    deck: Cards
    """Remaining deck"""

    dealer: Cards
    """Dealers hand.
    While the game is in progress, this only has one card. At game end this will contain the
    dealers final hand.
    """

    hands: UInt8[Cards, "5"]
    """Player hands"""

    wager: UInt8[Array, "5"]
    """Wager for each player hand"""

    status: UInt8[Array, "5"]
    """Status of each player hand"""

    actions: Bool[ActionMask, "5"]
    """Mask of actions valid for each player hand"""

    @staticmethod
    def from_hands(key: PRNGKey, dealer: Rank, player: Cards) -> Self:
        """Create a new game using the initial deal"""
        dealer_ace = dealer == 1
        player_pair = _is_pair(player)

        dealer = jax.nn.one_hot(dealer - 1, 13, dtype=jnp.uint8)
        deck = (Game.NDECKS * 4 * jnp.ones(13, jnp.uint8)) - player - dealer

        hands = jnp.zeros((Game.NHANDS, 13), jnp.uint8)
        hands = hands.at[0].set(player)
        wager = jax.nn.one_hot(0, Game.NHANDS, dtype=jnp.uint8)
        status = jnp.full(Game.NHANDS, Status.EMPTY, dtype=jnp.uint8)
        status = status.at[0].set(Status.PLAY)

        actions = jnp.zeros((Game.NHANDS, len(Action)), jnp.bool)
        actions = actions.at[0, Action.HIT].set(True)
        actions = actions.at[0, Action.STAND].set(True)
        actions = actions.at[0, Action.DOUBLE].set(True)
        actions = actions.at[0, Action.SPLIT].set(player_pair)
        actions = actions.at[0, Action.SURRENDER].set(~dealer_ace)

        return Game(
            key=key,
            play=jnp.array(0, jnp.uint8),
            split=jnp.array(1, jnp.uint8),
            deck=deck,
            dealer=dealer,
            hands=hands,
            wager=wager,
            status=status,
            actions=actions,
        )

    @staticmethod
    def from_deal(key: PRNGKey) -> Self:
        key, k0, k1, k2 = jax.random.split(key, 4)
        deck = Game.NDECKS * 4 * jnp.ones(13, jnp.uint8)

        c1 = _choice(k0, deck)
        deck = deck.at[c1].subtract(1)

        c2 = _choice(k1, deck)
        deck = deck.at[c2].subtract(1)

        dealer = _choice(k2, deck)
        deck = deck.at[dealer].subtract(1)
        dealer = dealer + 1  # Convert to Rank, 1-indexed

        player = jnp.zeros(13, jnp.uint8)
        player = player.at[c1].add(1)
        player = player.at[c2].add(1)

        return Game.from_hands(key, dealer, player)

    def _action_hit(self) -> Self:
        h = self.play

        key, deck, hand = _draw(self.key, self.deck, self.hands[h])
        hands = self.hands.at[h].set(hand)

        # Set new action flags, Double and Surrender flags are preserved
        actions, status = self.actions, self.status
        actions = actions.at[h, Action.HIT].set(True)
        actions = actions.at[h, Action.STAND].set(True)
        actions = actions.at[h, Action.SPLIT].set(False)
        status = status.at[h].set(_calc_status(hand))

        return self.replace(
            key=key, deck=deck, hands=hands, actions=actions, status=status
        )

    def _action_stand(self) -> Self:
        status = self.status.at[self.play].set(Status.STAND)
        return self.replace(status=status)

    def _action_double(self) -> Self:
        h = self.play
        key, deck, hand = _draw(self.key, self.deck, self.hands[h])
        hands = self.hands.at[h].set(hand)

        wager, actions = self.wager, self.actions
        can_redouble = wager[h] == 1
        wager = wager.at[h].add(1)
        actions = actions.at[h, Action.HIT].set(False)
        actions = actions.at[h, Action.STAND].set(True)
        actions = actions.at[h, Action.SPLIT].set(False)
        actions = actions.at[h, Action.DOUBLE].set(can_redouble)
        actions = actions.at[h, Action.SURRENDER].set(False)
        status = self.status.at[h].set(_calc_status(hand))

        return self.replace(
            key=key,
            deck=deck,
            hands=hands,
            wager=wager,
            actions=actions,
            status=status,
        )

    def _action_split(self) -> Self:
        h, s = self.play, self.split

        not_ace = ~self.hands[h, 0].astype(jnp.bool)

        # Remove card from first hand, its a pair so just divide counts by 2
        hand = self.hands[h] // 2

        # Draw for each hand
        key, deck, hands = self.key, self.deck, self.hands
        key, deck, hand_h = _draw(key, deck, hand)
        key, deck, hand_s = _draw(key, deck, hand)
        hands = hands.at[h].set(hand_h)
        hands = hands.at[s].set(hand_s)

        wager = self.wager.at[s].set(self.wager[h])

        actions = self.actions

        actions = actions.at[h, Action.HIT].set(not_ace)
        actions = actions.at[h, Action.STAND].set(True)
        actions = actions.at[h, Action.SPLIT].set(not_ace & _is_pair(hand_h))
        actions = actions.at[h, Action.DOUBLE].set(not_ace)
        actions = actions.at[h, Action.SURRENDER].set(False)

        actions = actions.at[s, Action.HIT].set(not_ace)
        actions = actions.at[s, Action.STAND].set(True)
        actions = actions.at[s, Action.SPLIT].set(not_ace & _is_pair(hand_s))
        actions = actions.at[s, Action.DOUBLE].set(not_ace)
        actions = actions.at[s, Action.SURRENDER].set(False)

        status = self.status
        status = status.at[h].set(_calc_status(hand_h))
        status = status.at[s].set(Status.PLAY)

        split = s + 1

        return self.replace(
            key=key,
            split=split,
            deck=deck,
            hands=hands,
            wager=wager,
            actions=actions,
            status=status,
        )

    def _action_surrender(self) -> Self:
        # Assumed to be playing first and only hand
        status = self.status.at[0].set(Status.SURRENDER)
        return self.replace(status=status)

    def _action_invalid(self) -> Self:
        return self

    def _dealer_turn(self) -> Self:
        """Take dealers turn, dealer stands on 17 (including soft)"""
        key, deck, dealer = self.key, self.deck, self.dealer

        def cond(x):
            key, deck, dealer = x
            return _cards_total(dealer) < 17

        def body(x):
            return _draw(*x)

        key, deck, dealer = jax.lax.while_loop(cond, body, (key, deck, dealer))
        return self.replace(key=key, deck=deck, dealer=dealer)

    def _score(self) -> Reward:
        """Calculate the reward"""
        # If dealer busts, set dealer_total to 1.
        # If player busts, set hand_total to 0.
        # Then if both bust, `dealer_total > total` -> dealer wins

        dealer_total = _cards_total(self.dealer)
        dealer_total = jax.lax.select(
            dealer_total > 21, jnp.array(1, jnp.uint8), _cards_total(self.dealer)
        )

        dealer_blackjack = (dealer_total == 21) & (jnp.sum(self.dealer) == 2)
        never_split = self.split == 1

        totals = _cards_total(self.hands)
        totals *= totals <= 21
        surrender = self.status == Status.SURRENDER
        two_cards = jnp.sum(self.hands, axis=1) == 2
        blackjack = never_split & two_cards & ~dealer_blackjack & (totals == 21)
        # 1 if player wins, -1 if player loses, 0 if push
        win_lose = jnp.sign(totals.astype(jnp.int8) - dealer_total.astype(jnp.int8))

        reward = surrender.astype(jnp.float16) * -0.5 * self.wager
        reward += (~surrender * blackjack).astype(jnp.float16) * 1.5
        reward += (~surrender * ~blackjack * win_lose).astype(jnp.float16) * self.wager

        return reward

    def done(self) -> SBool:
        """Return `True` if the game is in a terminal state"""
        return self.play == self.split

    def replace(self, **kwargs) -> Self:
        """Alias for `dataclasses.replace(self, **kwargs)`"""
        from dataclasses import replace

        return replace(self, **kwargs)

    def reset(self) -> Self:
        """Alias for `Game.from_deal(game.key)`"""
        return Game.from_deal(self.key)

    def step(self, action: SUInt8) -> tuple[Self, Reward, Bool]:
        valid = self.actions[self.play, action]
        action = jax.lax.select(valid, action, jnp.array(len(Action), jnp.uint8))

        # Order of handlers must match order of Action and invalid at the end
        game = jax.lax.switch(
            action,
            [
                Game._action_hit,
                Game._action_stand,
                Game._action_split,
                Game._action_double,
                Game._action_surrender,
                Game._action_invalid,
            ],
            self,
        )

        # Move to the next hand if needed
        play = game.play + (game.status[game.play] != Status.PLAY)
        game = game.replace(play=play)

        # 0 if invalid action, 1 if game continues, 2 if game ended
        game_status = jax.lax.select(game.play == game.split, 2, 1)
        game_status = jax.lax.select(valid, game_status, 0)

        def illegal(game: Game) -> tuple[Game, Reward, Bool]:
            # Illegal actions get a reward of -20
            return (
                game,
                jnp.full(5, -20.0, jnp.float16),
                jnp.array(True, jnp.bool),
            )

        def normal(game: Game) -> tuple[Game, Reward, Bool]:
            return (
                game,
                jnp.zeros(5, jnp.float16),
                jnp.array(False, jnp.bool),
            )

        def ended(game: Game) -> tuple[Game, Reward, Bool]:
            game = game._dealer_turn()
            reward = game._score()
            return (
                game,
                reward,
                jnp.array(True, jnp.bool),
            )

        return jax.lax.switch(game_status, [illegal, normal, ended], game)

    def legal_actions(self) -> Bool[ActionMask, "*batch"]:
        return self.actions[..., self.play, :]


def _choice(key: PRNGKey, deck: Cards) -> SUInt8:
    """Pick a single card from a set, returns Rank - 1

    This only exists as a patch-point for unit tests to control drawn cards.
    """
    return jax.random.choice(key, 13, p=deck)


def _draw(key: PRNGKey, deck: Cards, hand: Cards) -> tuple[PRNGKey, Cards, Cards]:
    """Draw a card from deck into hand, returns (key, deck, hand)"""
    key, k = jax.random.split(key, 2)
    card = _choice(k, deck)
    deck = deck.at[card].subtract(1)
    hand = hand.at[card].add(1)
    return key, deck, hand


def _is_pair(cards: UInt8[Cards, "*batch"]) -> Bool[Array, "*batch"]:
    pairs = 2 * jnp.eye(13, dtype=jnp.uint8)
    return jnp.any(jnp.all(cards[..., jnp.newaxis] == pairs, axis=-2), axis=-1)


def _cards_total(cards: UInt8[Cards, "*batch"]) -> UInt8[Array, "*batch"]:
    """Calculate the total of a hand"""
    values = jnp.clip(jnp.arange(1, 14, dtype=jnp.uint8), max=10)
    total = jnp.dot(cards, values)
    total += 10 * (total < 12) * (cards[..., 0] > 0)
    return total


def _calc_status(hand: UInt8[Cards, "*batch"]) -> UInt8[Array, "*batch"]:
    total = _cards_total(hand)
    status = Status.PLAY * (total < 21)
    status += Status.BUST * (total > 21)
    status += Status.STAND * (total == 21)
    return status
