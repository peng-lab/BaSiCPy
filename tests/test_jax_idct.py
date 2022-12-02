import jax.numpy as jnp
import scipy.fft
from jax._src.scipy.fft import _dct_interleave

from basicpy.tools import _jax_idct


def test_inverse_interleave():
    x = jnp.arange(24).reshape(4, 3, 2)

    for axis in range(x.ndim):
        y = _dct_interleave(x, axis=axis)
        y_inv = _jax_idct._dct_interleave_inverse(y, axis)
        assert jnp.allclose(x, y_inv)


def test_custom_jax_idct():
    x = jnp.arange(24).reshape(4, 3, 2)

    for axis in range(x.ndim):
        ours = _jax_idct.idct(x, norm="ortho", axis=axis)
        theirs = scipy.fft.idct(x, norm="ortho", axis=axis)
        assert jnp.allclose(ours, theirs)

        ours = _jax_idct.idct(x, axis=axis)
        theirs = scipy.fft.idct(x, axis=axis)
        assert jnp.allclose(ours, theirs)
