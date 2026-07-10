"""As visualizações do fotométrico devem apenas salvar arquivos, sem abrir janelas."""
import numpy as np
import pytest

from hybrid_stereo_method.photometric import visualization


@pytest.fixture(autouse=True)
def forbid_gui(monkeypatch):
    """Qualquer chamada de GUI (popup/bloqueio) falha o teste."""

    def _boom(*args, **kwargs):
        raise AssertionError("GUI window opened — pipeline must be headless")

    monkeypatch.setattr(visualization.cv2, "imshow", _boom)
    monkeypatch.setattr(visualization.cv2, "waitKey", _boom)
    monkeypatch.setattr(visualization.plt, "show", _boom)


@pytest.fixture
def normals():
    rng = np.random.default_rng(0)
    n = rng.normal(size=(6, 5, 3))
    return (n / np.linalg.norm(n, axis=2, keepdims=True)).astype(np.float64)


def test_disp_normalmap_saves_without_window(tmp_path, normals):
    visualization.disp_normalmap(normal=normals, height=6, width=5, save_path=str(tmp_path))
    assert (tmp_path / "normal_map.png").exists()


def test_disp_channels_saves_without_window(tmp_path, normals):
    visualization.disp_channels(normal_in=normals, height=6, width=5, save_path=str(tmp_path))
    assert (tmp_path / "Channels.png").exists()


def test_disp_channels_3d_saves_without_window(tmp_path, normals):
    visualization.disp_channels_3d(normal_in=normals, height=6, width=5, save_path=str(tmp_path))
    assert (tmp_path / "Channels_3D.png").exists()
