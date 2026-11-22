"""MF-14: luzes L<n> detectadas em qualquer nível, não só no pai imediato."""

import os

from hybrid_stereo_method.hybrid.main import collect_light_dirs


def test_clean_layout_without_stray_files_in_light_dirs():
    root = "/d/ds"
    files = [os.path.join(root, f"L{i}", f"zf{k}", "sVal.png") for i in (10, 2, 0) for k in (0, 1)]
    files.append(os.path.join(root, "lights.npy"))
    assert collect_light_dirs(files, root) == ["L0", "L2", "L10"]


def test_nested_layout_and_non_light_names_ignored():
    root = "/d/lamb-gls"
    files = [
        f"{root}/out/stQ/0512x0384/L000/zf007.2000-df051.2000/sVal.png",
        f"{root}/out/stQ/0512x0384/L001/sharp/sVal.png",
        f"{root}/out/stQ/0512x0384/Lights/readme.txt",
        "/elsewhere/L005/zf0/sVal.png",  # fora do data_path
    ]
    assert collect_light_dirs(files, root) == ["L000", "L001"]


def test_light_dirs_with_suffix():
    """stF-melon24 (2025-02-09): pastas de luz L000-amb0.00-glo0.00."""
    root = "/d/stF"
    files = [os.path.join(root, f"L{i:03d}-amb0.00-glo0.00", "zf026.2500-df015.0000", "sVal.png") for i in (1, 0)]
    files.append(os.path.join(root, "Lights", "x.txt"))
    assert collect_light_dirs(files, root) == ["L000-amb0.00-glo0.00", "L001-amb0.00-glo0.00"]


def test_common_zf_dirs_drops_planes_missing_in_some_light():
    from hybrid_stereo_method.hybrid.main import common_zf_dirs

    root = "/d/stF"
    files = []
    for light, planes in (("L000-a", ["zf026", "zf060", "zf093"]), ("L001-a", ["zf026", "zf093"])):
        files += [os.path.join(root, light, p, "sVal.png") for p in planes]
    lights = collect_light_dirs(files, root)
    assert common_zf_dirs(files, root, lights) == ["zf026", "zf093"]


def test_common_zf_dirs_without_lights_uses_all():
    from hybrid_stereo_method.hybrid.main import common_zf_dirs

    files = ["/d/x/zf2/sVal.png", "/d/x/zf10/sVal.png"]
    assert common_zf_dirs(files, "/d/x", []) == ["zf2", "zf10"]
