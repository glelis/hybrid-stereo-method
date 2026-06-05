# tests/test_fni_roundtrip.py
"""Fase 3.2 — round-trip FNI Python -> Python.

Padrões ASSIMÉTRICOS em x e y: um flip ou transposição silenciosa não passa.
Se um teste falhar, NÃO conserte o teste: registre o achado (IO-xx) com a saída.
"""
import numpy as np
import pytest

from hybrid_stereo_method.infrastructure.io.image_io import (
    convert_image_array_to_fni,
    read_fni_to_image_array,
)


def _asymmetric(shape):
    """Array sem nenhuma simetria de flip/transposição."""
    return (np.arange(np.prod(shape), dtype=np.float64).reshape(shape) ** 1.5 + 0.125) / 7.0


def test_roundtrip_2d(tmp_path):
    arr = _asymmetric((5, 9))  # retangular: transposição também quebraria o shape
    path = tmp_path / "a.fni"
    convert_image_array_to_fni(arr, path)
    back = read_fni_to_image_array(path)
    assert back.shape == arr.shape
    np.testing.assert_allclose(back, arr, rtol=1e-6)


def test_roundtrip_3channel(tmp_path):
    arr = _asymmetric((4, 6, 3))
    path = tmp_path / "b.fni"
    convert_image_array_to_fni(arr, path)
    back = read_fni_to_image_array(path)
    assert back.shape == arr.shape
    np.testing.assert_allclose(back, arr, rtol=1e-6)


def test_roundtrip_negative_and_large_values(tmp_path):
    arr = np.array([[-1.5e6, 3.25e-7], [0.0, -0.0]], dtype=np.float64)
    path = tmp_path / "c.fni"
    convert_image_array_to_fni(arr, path)
    np.testing.assert_allclose(read_fni_to_image_array(path), arr, rtol=1e-6, atol=1e-30)


def test_nan_handling_documented(tmp_path):
    """Sonda de comportamento: normais com NaN (sombra) são escritas em FNI
    pelo pipeline. Este teste DOCUMENTA o que o round-trip Python faz com NaN
    (o comportamento do lado C é avaliado na auditoria INT)."""
    arr = np.array([[1.0, np.nan], [2.0, 3.0]])
    path = tmp_path / "d.fni"
    convert_image_array_to_fni(arr, path)
    back = read_fni_to_image_array(path)
    assert np.isnan(back[0, 1]), "NaN não sobreviveu ao round-trip — registrar comportamento real"
    np.testing.assert_allclose(back[~np.isnan(back)], arr[~np.isnan(arr)], rtol=1e-6)
