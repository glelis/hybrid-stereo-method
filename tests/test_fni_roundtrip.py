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


def test_roundtrip_preserves_float64_precision(tmp_path):
    """IO-01: %.7e truncava float64 a ~8 dígitos; com %.16e o ARQUIVO carrega
    a precisão de float64 (o reader aloca float32 — o contrato do round-trip
    Python é float32; o arquivo deve preservar o dado da fonte)."""
    arr = np.array([[0.123456789012345, -9.87654321098765e10]])
    path = tmp_path / "p.fni"
    convert_image_array_to_fni(arr, path)
    text = path.read_text()
    assert "+1.2345678901234500e-01" in text, "writer ainda trunca a 7 casas (IO-01)"


def test_truncated_fni_raises(tmp_path):
    """IO-02: pixels ausentes ficavam silenciosamente 0."""
    arr = _asymmetric((4, 4))
    path = tmp_path / "t.fni"
    convert_image_array_to_fni(arr, path)
    lines = path.read_text().splitlines(keepends=True)
    data_idx = [i for i, ln in enumerate(lines) if ln.strip() and ln.split()[0].isdigit()]
    del lines[data_idx[5]]
    del lines[data_idx[4]]
    path.write_text("".join(lines))
    with pytest.raises(ValueError, match="incomplete"):
        read_fni_to_image_array(path)


def test_malformed_short_line_raises(tmp_path):
    """IO-03: linha com menos campos era pulada em silêncio (pixel ficava 0)."""
    path = tmp_path / "m.fni"
    path.write_text(
        "begin float_image_t (format of 2006-03-25)\n"
        "NC = 1\nNX = 2\nNY = 1\n"
        "    0     0 +1.0000000e+00\n"
        "    1\n"  # truncada
        "\nend float_image_t\n"
    )
    with pytest.raises(ValueError, match="[Mm]alformed"):
        read_fni_to_image_array(path)


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
