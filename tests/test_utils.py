from blkjx.utils import Array, SBool, Status

from beartype.claw import beartype_package
import jax
import jax.numpy as jnp

beartype_package("blkjx")

@jax.jit
def is_u8(x: Array) -> SBool:
    return x.dtype == jnp.uint8

def test_jaxenum_constants_are_u8():
    assert is_u8(Status.PLAY)

def test_jaxenum_iterates():
    act = [s for s in Status]
    exp = [0, 1, 2, 3, 4]
    assert act == exp

def test_jaxenum_len():
    assert len(Status) == 5

def test_jaxenum_iterates_as_arrays():
    for s in Status:
        assert is_u8(s)

def test_jaxenum_repr():
    assert repr(Status(1)) == "<Status.PLAY: 1>"