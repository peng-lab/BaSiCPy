"""Tests the 2D dct tools."""

# from basicpy.tools import dct2d_tools
from basicpy.tools.dct2d_tools import DCT_BACKENDS

import pytest
import scipy.fft
import numpy as np
import importlib

np.random.seed(1234)
arr_input = np.random.random((4, 4))
arr_forward_exp = scipy.fft.dct(
    scipy.fft.dct(arr_input.T, norm="ortho").T, norm="ortho"
)
arr_reverse_exp = scipy.fft.idct(
    scipy.fft.idct(arr_input.T, norm="ortho").T, norm="ortho"
)

# backends = ["JAX", "OPENCV", "SCIPY"]
backends = ["OPENCV", "SCIPY"]


@pytest.mark.parametrize("backend", backends)
def test_dct_backends(backend):
    dct2d = DCT_BACKENDS[backend].dct2d
    idct2d = DCT_BACKENDS[backend].idct2d

    # check that dct2d and idct2d are at least inverse functions
    assert np.allclose(arr_input, idct2d(dct2d(arr_input)))

    # check that results match scipy dct
    arr_forward_actual = dct2d(arr_input)
    arr_reverse_actual = idct2d(arr_input)

    assert np.allclose(arr_forward_exp, arr_forward_actual)
    assert np.allclose(arr_reverse_exp, arr_reverse_actual)


@pytest.mark.parametrize("backend", backends)
def test_dct_backend_import(monkeypatch, backend):
    import basicpy.tools.dct2d_tools

    idct2d = DCT_BACKENDS[backend].idct2d

    monkeypatch.setenv("BASIC_DCT_BACKEND", backend)
    importlib.reload(basicpy.tools.dct2d_tools)

    assert basicpy.tools.dct2d_tools.dct._backend == backend


def test_unrecognized_backend(monkeypatch):
    import basicpy.tools.dct2d_tools

    backend = "FAKE_BACKEND"

    monkeypatch.setenv("BASIC_DCT_BACKEND", "FAKE_BACKEND")

    # with pytest.raises(KeyError):
    #     importlib.reload(basicpy.tools.dct2d_tools)

    assert (
        basicpy.tools.dct2d_tools.dct._backend
        == basicpy.tools.dct2d_tools.DEFAULT_BACKEND
    )


@pytest.mark.parametrize("backend", backends)
def test_backend_not_installed(monkeypatch, backend):
    # TODO mimic package not installed by removing from path?
    ...


### BENCHMARKING ###
@pytest.mark.parametrize("backend", backends)
def test_dct_backends_benchmark_dct2d(backend, benchmark):
    dct2d = DCT_BACKENDS[backend].dct2d
    benchmark(dct2d, arr_input)


@pytest.mark.parametrize("backend", backends)
def test_dct_backends_benchmark_idct2d(backend, benchmark):
    idct2d = DCT_BACKENDS[backend].idct2d
    benchmark(idct2d, arr_input)
