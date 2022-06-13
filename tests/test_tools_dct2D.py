"""Tests the 2D dct tools."""

# from basicpy.tools import dct2d_tools
from basicpy.tools.dct_tools import DCT_BACKENDS

import pytest
import scipy.fft
import numpy as np
import importlib

np.random.seed(1234)
arr_input_2d = np.random.random((4, 4))
arr_input_3d = np.random.random((3, 4, 5))
arr_forward_exp_2d = scipy.fft.dctn(arr_input_2d, norm="ortho")
arr_reverse_exp_2d = scipy.fft.idctn(arr_input_2d, norm="ortho")
arr_forward_exp_3d = scipy.fft.dctn(arr_input_3d, norm="ortho")
arr_reverse_exp_3d = scipy.fft.idctn(arr_input_3d, norm="ortho")
backends = ["JAX", "SCIPY"]
# backends = ["OPENCV", "SCIPY"]


@pytest.mark.parametrize("backend", backends)
def test_dct_backends(backend):
    dct2d = DCT_BACKENDS[backend].dct2d
    idct2d = DCT_BACKENDS[backend].idct2d

    # check that dct2d and idct2d are at least inverse functions
    assert np.allclose(arr_input_2d, idct2d(dct2d(arr_input_2d)))

    # check that results match scipy dct
    arr_forward_actual = dct2d(arr_input_2d)
    arr_reverse_actual = idct2d(arr_input_2d)

    assert np.allclose(arr_forward_exp_2d, arr_forward_actual)
    assert np.allclose(arr_reverse_exp_2d, arr_reverse_actual)

    dct3d = DCT_BACKENDS[backend].dct3d
    idct3d = DCT_BACKENDS[backend].idct3d

    # check that dct2d and idct2d are at least inverse functions
    assert np.allclose(arr_input_3d, idct3d(dct3d(arr_input_3d)), atol=1e-5, rtol=1e-5)

    # check that results match scipy dct
    arr_forward_actual = dct3d(arr_input_3d)
    arr_reverse_actual = idct3d(arr_input_3d)

    assert np.allclose(arr_forward_exp_3d, arr_forward_actual, atol=1e-5, rtol=1e-5)
    assert np.allclose(arr_reverse_exp_3d, arr_reverse_actual, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("backend", backends)
def test_dct_backend_import(monkeypatch, backend):
    import basicpy.tools.dct_tools

    idct2d = DCT_BACKENDS[backend].idct2d

    monkeypatch.setenv("BASIC_DCT_BACKEND", backend)
    importlib.reload(basicpy.tools.dct_tools)

    assert basicpy.tools.dct_tools.dct._backend == backend


def test_unrecognized_backend(monkeypatch):
    import basicpy.tools.dct_tools

    backend = "FAKE_BACKEND"

    monkeypatch.setenv("BASIC_DCT_BACKEND", "FAKE_BACKEND")

    # with pytest.raises(KeyError):
    #     importlib.reload(basicpy.tools.dct2d_tools)

    assert (
        basicpy.tools.dct_tools.dct._backend == basicpy.tools.dct_tools.DEFAULT_BACKEND
    )


@pytest.mark.parametrize("backend", backends)
def test_backend_not_installed(monkeypatch, backend):
    # TODO mimic package not installed by removing from path?
    ...


### BENCHMARKING ###
@pytest.mark.parametrize("backend", backends)
def test_dct_backends_benchmark_dct2d(backend, benchmark):
    dct2d = DCT_BACKENDS[backend].dct2d
    benchmark(dct2d, arr_input_2d)


@pytest.mark.parametrize("backend", backends)
def test_dct_backends_benchmark_idct2d(backend, benchmark):
    idct2d = DCT_BACKENDS[backend].idct2d
    benchmark(idct2d, arr_input_2d)
