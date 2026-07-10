"""Seleção dos sMos.npy por luz para o fotométrico (exclui só a pasta average)."""
from hybrid_stereo_method.hybrid.main import select_per_light_smos


def test_selects_per_light_and_excludes_average():
    files = [
        "/out/run1/multifocus_stereo/L000/sMos.npy",
        "/out/run1/multifocus_stereo/L001/sMos.npy",
        "/out/run1/multifocus_stereo/average/sMos.npy",
        "/out/run1/multifocus_stereo/L000/zMos.fni",
    ]
    assert select_per_light_smos(files) == [
        "/out/run1/multifocus_stereo/L000/sMos.npy",
        "/out/run1/multifocus_stereo/L001/sMos.npy",
    ]


def test_av_substring_elsewhere_in_path_does_not_exclude():
    """Regressão: run em pasta com 'av' no nome (ex.: 03_wavelet) zerava a lista."""
    files = [
        "/out/hybrid_sweep/03_wavelet/run1/multifocus_stereo/L000/sMos.npy",
        "/out/hybrid_sweep/03_wavelet/run1/multifocus_stereo/average/sMos.npy",
    ]
    assert select_per_light_smos(files) == [
        "/out/hybrid_sweep/03_wavelet/run1/multifocus_stereo/L000/sMos.npy",
    ]


def test_natural_sort_order():
    files = [
        "/out/run1/multifocus_stereo/L010/sMos.npy",
        "/out/run1/multifocus_stereo/L002/sMos.npy",
    ]
    assert select_per_light_smos(files) == [
        "/out/run1/multifocus_stereo/L002/sMos.npy",
        "/out/run1/multifocus_stereo/L010/sMos.npy",
    ]
