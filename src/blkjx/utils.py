"""Type definitions and utilities"""

from enum import auto as auto

import jax.numpy as jnp
from jaxtyping import (
    Array as Array,
    Bool as Bool,
    Float16 as Float16,
    PRNGKeyArray as PRNGKey,  # noqa: F401
    Scalar as Scalar,
    UInt8 as UInt8,
)


# Scalar aliases


SBool = Bool[Scalar, ""]
"""Scalar Bool"""

SFloat16 = Float16[Scalar, ""]
"""Scalar Float16"""

SUInt8 = UInt8[Scalar, ""]
"""Scalar UInt8"""


# Blackjack types


ActionMask = Bool[Array, "5"]
"""Boolean vector where v[i] is True if `Action(i)` is valid"""


Rank = SUInt8
"""Rank of a card, a value in [0, 14) with 0 = NONE, 1 = Ace, ... 13 = King"""


Cards = UInt8[Array, "13"]
"""An unordered set of cards.
`cards[i]` indicates the number of each rank in the collection. Note: indexed by `Rank` - 1
`cards[0]` - # of Aces, `cards[1]` - # of twos, ... `cards[12]` = # of Kings
"""


class JaxEnumMeta(type):
    """Metaclass for enums whose values are JAX scalars, see `JaxEnum`"""

    def __new__(clsmeta, clsname, bases, namespace, **kwds):
        dtype = namespace.pop("DTYPE", None)

        auto_null = auto().value
        v_next = 0
        members = []
        for k, v in namespace.items():
            if isinstance(v, int):
                members.append((k, v))
                v_next = v + 1
            elif isinstance(v, auto):
                v = v.value if v.value != auto_null else v_next
                members.append((k, v))
                v_next = v + 1

        namespace["__slots__"] = ("name", "value")
        if dtype:
            namespace["_dtype"] = dtype

        for name, _ in members:
            del namespace[name]

        if "__repr__" not in namespace:

            def repr(self):
                return f"<{clsname}.{self.name}: {self.value}>"

            namespace["__repr__"] = repr

        cls = super().__new__(clsmeta, clsname, bases, namespace)

        instances = {n: cls.__new__(cls, n, v) for n, v in members}
        cls._instances = instances

        return cls

    def __call__(cls, value):
        instance = next((i for i in cls._instances.values() if i.value == value), None)
        if instance is None:
            raise ValueError(f"{value} is not a valid {cls.__name__}")
        return instance

    def __getattr__(cls, item):
        try:
            instance = cls._instances[item]
        except Exception:
            raise AttributeError(
                f"type object '{cls.__name__}' has no attribute '{item}'"
            )

        return jnp.array(instance.value, cls._dtype)

    def __len__(cls):
        return len(cls._instances)

    def __iter__(cls):
        return (jnp.array(i.value, cls._dtype) for i in cls._instances.values())


class JaxEnum(metaclass=JaxEnumMeta):
    """Base class for enums whose values are JAX scalars

    This works much like stdlib's `enum.Enum`; however, the enum must specify `DTYPE` and the
    associated class constants are Jax Arrays instead of class instances.
    """

    def __new__(cls, name, value):
        # Only called by JaxEnumMeta
        instance = super().__new__(cls)
        instance.name = name
        instance.value = value
        return instance


class Status(JaxEnum):
    """Status values for `Hand`

    The value for `EMPTY` is guaranteed zero, all other values are unspecified.
    """

    DTYPE = jnp.uint8

    EMPTY = 0
    """Not an active hand, value is guaranteed to be `0`"""

    PLAY = auto()
    """Active hand awaiting player action"""

    STAND = auto()
    """Inactive hand ended with the player standing"""

    BUST = auto()
    """Inactive hand ended with the player busting"""

    SURRENDER = auto()
    """Inactive hand ended with the player surrendering"""


class Action(JaxEnum):
    """Actions for a game of blackjack, values are unspecified but in the range [0, 5)"""

    DTYPE = jnp.uint8

    HIT = auto()
    STAND = auto()
    SPLIT = auto()
    DOUBLE = auto()
    SURRENDER = auto()


# Utility methods


RANKS = "A23456789TJQK"


def rank_from_str(msg: str) -> Rank:
    # Any invalid rank -> 0 -> NONE
    return jnp.array(RANKS.find(msg) + 1, jnp.uint8)


def cards_from_str(msg: str) -> Cards:
    cards = jnp.zeros(13, jnp.uint8)
    for c in msg:
        cards = cards.at[RANKS.find(c)].add(1)
    return cards
