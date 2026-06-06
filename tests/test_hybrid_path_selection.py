# tests/test_hybrid_path_selection.py
"""Tests for MF-01 and MF-02 fixes: exact path-component filtering and natural ordering.

MF-01: substring match `"zf1" in file` also matches zf10/zf11/zf12 with >=10 planes.
MF-02: sorted() is lexicographic; with 12 planes the order is zf1,zf10,zf11,zf12,zf2,...,
       misaligning average_images_paths from the YAML z_foc list (natural ascending order).
"""

from pathlib import Path

from hybrid_stereo_method.hybrid.main import (
    collect_dirs_with_prefix,
    select_files_by_parent_dir,
    select_light_stack_files,
)

# ---------------------------------------------------------------------------
# Fixtures: a synthetic 12-plane, 12-light dataset layout
# ---------------------------------------------------------------------------


def _make_stack_paths(n_lights: int, n_zf: int, filename: str = "sVal.png") -> list[str]:
    """Produce fake path strings matching the real L<n>/zf<m>/<filename> layout.

    In this layout the immediate parent of every sVal.png is always a `zf*` directory.
    This means collect_dirs_with_prefix(..., prefix="L") returns [] — that is the
    known MF-14 bug (tracked separately). These paths are used for zf-level tests.
    """
    paths = []
    for light in range(n_lights):
        for zf in range(1, n_zf + 1):
            paths.append(f"/data/raw/dataset/L{light}/zf{zf}/{filename}")
    return paths


def _make_light_dir_paths(n_lights: int, filename: str = "selected-pixels.png") -> list[str]:
    """Produce paths where the immediate parent IS an L* directory.

    In real datasets some files (e.g. selected-pixels.png) live directly under L*/,
    so the immediate parent is the L* directory. These paths exercise collect_dirs_with_prefix
    for the L* prefix.
    """
    return [f"/data/raw/dataset/L{i}/{filename}" for i in range(n_lights)]


PATHS_12L_12ZF = _make_stack_paths(n_lights=12, n_zf=12)
PATHS_12L_DIRECT = _make_light_dir_paths(n_lights=12)  # parent = L*, for L* prefix tests

# A couple of extras to verify they don't sneak in
PATHS_WITH_EXTRAS = PATHS_12L_12ZF + [
    "/data/raw/dataset/L0/selected-pixels.png",  # parent = L0, not a zf dir
    "/data/raw/dataset/L0/zf1/other_file.png",  # wrong filename
    "/data/raw/dataset/lights.npy",  # top-level, no zf parent
    "/data/raw/dataset/sharp/hAvg.png",  # no zf/L parent at all
]


# ===========================================================================
# MF-01 regression: "zf1" must NOT match files under zf10/zf11/zf12
# ===========================================================================


class TestSelectFilesByParentDir:
    """Tests for select_files_by_parent_dir (fixes MF-01)."""

    def test_zf1_does_not_match_zf10_zf11_zf12(self):
        """Regression for MF-01: with 12 planes selecting 'zf1' must return only zf1 files."""
        result = select_files_by_parent_dir(
            PATHS_12L_12ZF, dir_name="zf1", filename_part="sVal.png"
        )
        # Should be exactly 12 files (one per light) — all with parent zf1, none from zf10/11/12
        assert len(result) == 12, (
            f"Expected 12 files (one per light in zf1), got {len(result)}: {result}"
        )
        for path in result:
            parent = Path(path).parent.name
            assert parent == "zf1", f"Expected parent dir 'zf1', got '{parent}' for path: {path}"

    def test_does_not_select_l0_when_filtering_by_zf(self):
        """A file directly under L0 (no zf parent) must not be selected when filtering by zf dirs."""
        result = select_files_by_parent_dir(
            PATHS_WITH_EXTRAS, dir_name="L0", filename_part="sVal.png"
        )
        # All sVal.png under L0 have parent zf*, not L0
        assert len(result) == 0, (
            f"Expected 0 files (L0/sVal.png doesn't exist), got {len(result)}: {result}"
        )

    def test_filename_part_is_respected(self):
        """Only files whose basename contains filename_part are returned."""
        # There is 'other_file.png' under L0/zf1 in PATHS_WITH_EXTRAS
        result_sval = select_files_by_parent_dir(
            PATHS_WITH_EXTRAS, dir_name="zf1", filename_part="sVal.png"
        )
        result_other = select_files_by_parent_dir(
            PATHS_WITH_EXTRAS, dir_name="zf1", filename_part="other_file.png"
        )
        # sVal.png: 12 lights × 1 plane  (one from L0/zf1/other_file.png excluded)
        assert len(result_sval) == 12
        # other_file.png: exactly 1 (only under L0/zf1)
        assert len(result_other) == 1

    def test_returns_paths_in_natural_order(self):
        """Files are returned in natural order (L0, L1, ..., L9, L10, L11), not lexicographic."""
        result = select_files_by_parent_dir(
            PATHS_12L_12ZF, dir_name="zf1", filename_part="sVal.png"
        )
        # Extract the light number from each path
        light_nums = [int(Path(p).parent.parent.name.lstrip("L")) for p in result]
        assert light_nums == sorted(light_nums), (
            f"Expected natural order of light dirs, got: {light_nums}"
        )

    def test_empty_input_returns_empty(self):
        """Edge case: empty file list returns empty list."""
        result = select_files_by_parent_dir([], dir_name="zf1", filename_part="sVal.png")
        assert result == []

    def test_no_match_returns_empty(self):
        """Non-existent dir_name returns empty list."""
        result = select_files_by_parent_dir(
            PATHS_12L_12ZF, dir_name="zf99", filename_part="sVal.png"
        )
        assert result == []


# ===========================================================================
# MF-02: natural ordering of zf1..zf12 and L0..L11
# ===========================================================================


class TestCollectDirsWithPrefix:
    """Tests for collect_dirs_with_prefix (fixes MF-02)."""

    def test_zf_dirs_returned_in_natural_order(self):
        """With 12 zf planes, dirs must be [zf1, zf2, ..., zf9, zf10, zf11, zf12]."""
        result = collect_dirs_with_prefix(PATHS_12L_12ZF, prefix="zf")
        expected = [f"zf{i}" for i in range(1, 13)]
        assert result == expected, f"Natural sort expected {expected}, got {result}"

    def test_lexicographic_order_is_WRONG(self):
        """Confirm that sorted() would give the wrong order (the bug MF-02 describes)."""
        dirs = {Path(p).parent.name for p in PATHS_12L_12ZF if Path(p).parent.name.startswith("zf")}
        lexicographic = sorted(dirs)
        natural = [f"zf{i}" for i in range(1, 13)]
        # lexicographic and natural differ — this is the bug
        assert lexicographic != natural, (
            "sorted() happened to give the same order as natsorted — test assumptions wrong"
        )
        # Specifically, zf10 appears before zf2 in lexicographic order
        assert lexicographic.index("zf10") < lexicographic.index("zf2"), (
            "Expected zf10 before zf2 in lexicographic order (the known-bad order)"
        )

    def test_light_dirs_returned_in_natural_order(self):
        """With 12 light dirs where immediate parent IS L*, result must be [L0, L1, ..., L10, L11].

        Note: in the canonical L<n>/zf<m>/sVal.png layout, the immediate parent of sVal.png is
        always zf*, so collect_dirs_with_prefix(stack_paths, prefix="L") returns [] — that is the
        separate MF-14 bug. Here we use paths where the immediate parent IS the L* dir (as happens
        with stray files such as selected-pixels.png in real datasets).
        """
        result = collect_dirs_with_prefix(PATHS_12L_DIRECT, prefix="L")
        expected = [f"L{i}" for i in range(12)]
        assert result == expected, f"Natural sort expected {expected}, got {result}"

    def test_light_dirs_lexicographic_order_is_WRONG(self):
        """Confirm sorted() gives wrong order for L0..L11 (same class of bug as MF-02 for zf)."""
        dirs = {
            Path(p).parent.name for p in PATHS_12L_DIRECT if Path(p).parent.name.startswith("L")
        }
        lexicographic = sorted(dirs)
        natural = [f"L{i}" for i in range(12)]
        assert lexicographic != natural, (
            "sorted() happened to give the same order as natsorted — test assumptions wrong"
        )
        # L10 appears before L2 in lexicographic order
        assert lexicographic.index("L10") < lexicographic.index("L2"), (
            "Expected L10 before L2 in lexicographic order"
        )

    def test_only_prefix_matching_dirs_are_included(self):
        """collect_dirs_with_prefix must exclude dirs that don't start with the prefix."""
        result_zf = collect_dirs_with_prefix(PATHS_WITH_EXTRAS, prefix="zf")
        for name in result_zf:
            assert name.startswith("zf"), f"Got non-zf dir: {name}"

        # PATHS_WITH_EXTRAS has one file directly under L0 (selected-pixels.png)
        # so L0 appears as an immediate parent
        result_L = collect_dirs_with_prefix(PATHS_WITH_EXTRAS, prefix="L")
        for name in result_L:
            assert name.startswith("L"), f"Got non-L dir: {name}"

    def test_single_digit_dirs_unaffected(self):
        """With <=9 planes, natural and lexicographic order coincide — no regression."""
        paths_single = _make_stack_paths(n_lights=6, n_zf=9)
        result = collect_dirs_with_prefix(paths_single, prefix="zf")
        expected = [f"zf{i}" for i in range(1, 10)]
        assert result == expected

    def test_empty_input_returns_empty(self):
        """Edge case: empty file list returns empty list."""
        result = collect_dirs_with_prefix([], prefix="zf")
        assert result == []


# ===========================================================================
# Real float-encoded zf directory names (e.g. zf015.0000-df020.0000)
# ===========================================================================

# 12 focal planes with float-encoded names, step of 10 in the leading integer
_FLOAT_ZF_DIRS = [f"zf{15 + i * 10:03d}.0000-df020.0000" for i in range(12)]
# Expected natural order is numeric ascending on the leading number
_FLOAT_ZF_DIRS_NATURAL = _FLOAT_ZF_DIRS  # already in ascending order

# Build a synthetic 12-light × 12-zf path list using the float-encoded names
_FLOAT_ZF_PATHS: list[str] = [
    f"/data/raw/dataset/L{light}/{zf}/sVal.png" for light in range(12) for zf in _FLOAT_ZF_DIRS
]


class TestFloatEncodedZfNames:
    """Tests for the real-world zf naming format: zf015.0000-df020.0000 ... zf125.0000-df020.0000."""

    def test_collect_dirs_returns_correct_natural_order(self):
        """collect_dirs_with_prefix must return all 12 float-encoded zf dirs in numeric order."""
        result = collect_dirs_with_prefix(_FLOAT_ZF_PATHS, prefix="zf")
        assert result == _FLOAT_ZF_DIRS_NATURAL, f"Expected {_FLOAT_ZF_DIRS_NATURAL}, got {result}"

    def test_collect_dirs_returns_exactly_12_entries(self):
        """There are exactly 12 distinct zf dirs in the float-encoded dataset."""
        result = collect_dirs_with_prefix(_FLOAT_ZF_PATHS, prefix="zf")
        assert len(result) == 12

    def test_select_files_no_bleed_between_adjacent_zf_dirs(self):
        """Selecting zf015.0000-df020.0000 must NOT return files from zf025.0000-df020.0000."""
        first_dir = _FLOAT_ZF_DIRS[0]  # zf015.0000-df020.0000
        second_dir = _FLOAT_ZF_DIRS[1]  # zf025.0000-df020.0000
        result = select_files_by_parent_dir(
            _FLOAT_ZF_PATHS, dir_name=first_dir, filename_part="sVal.png"
        )
        assert len(result) == 12, f"Expected 12 files (one per light), got {len(result)}"
        for path in result:
            assert Path(path).parent.name == first_dir, (
                f"Expected parent '{first_dir}', got '{Path(path).parent.name}'"
            )
        # Confirm none of the returned paths come from the second dir
        for path in result:
            assert Path(path).parent.name != second_dir


# ===========================================================================
# select_light_stack_files: per-light call-site filtering
# ===========================================================================


class TestSelectLightStackFiles:
    """Tests for the extracted select_light_stack_files helper.

    This helper pins the three-part production condition:
      parent.name starts with "zf"  AND  grandparent.name == light_dir  AND  "sVal.png" in name
    """

    # Build a mixed path list that exercises all edge cases
    _PATHS = [
        "/data/raw/dataset/L0/zf1/sVal.png",  # should be selected for L0
        "/data/raw/dataset/L0/zf1/marker.txt",  # wrong filename — excluded
        "/data/raw/dataset/L0/extra/sVal.png",  # parent "extra" not zf* — excluded
        "/data/raw/dataset/L10/zf1/sVal.png",  # grandparent L10, not L0 — excluded for L0
        "/data/raw/dataset/L0/zf1/sub/sVal.png",  # deeper nesting: grandparent is zf1 — excluded
        "/data/raw/dataset/L0/zf2/sVal.png",  # second zf dir under L0 — selected
        "/data/raw/dataset/L10/zf2/sVal.png",  # L10, not L0 — excluded for L0
    ]

    def test_l0_selects_only_l0_zf_sval(self):
        """L0 filter returns only sVal.png files directly under L0/zf*/."""
        result = select_light_stack_files(self._PATHS, "L0")
        assert set(result) == {
            "/data/raw/dataset/L0/zf1/sVal.png",
            "/data/raw/dataset/L0/zf2/sVal.png",
        }, f"Unexpected result for L0: {result}"

    def test_marker_txt_excluded(self):
        """marker.txt under L0/zf1 must not appear in the selection."""
        result = select_light_stack_files(self._PATHS, "L0")
        assert not any("marker.txt" in f for f in result)

    def test_non_zf_parent_excluded(self):
        """L0/extra/sVal.png (parent 'extra', not zf*) must not be selected."""
        result = select_light_stack_files(self._PATHS, "L0")
        assert not any("extra" in Path(f).parent.name for f in result)

    def test_l10_not_selected_for_l0(self):
        """Files under L10/ must not appear when filtering for L0."""
        result = select_light_stack_files(self._PATHS, "L0")
        assert not any(Path(f).parent.parent.name == "L10" for f in result)

    def test_deeper_nesting_excluded(self):
        """L0/zf1/sub/sVal.png has grandparent zf1, not L0 — must be excluded."""
        result = select_light_stack_files(self._PATHS, "L0")
        assert "/data/raw/dataset/L0/zf1/sub/sVal.png" not in result

    def test_l10_selects_own_files(self):
        """Filtering for L10 returns exactly the L10 sVal.png files."""
        result = select_light_stack_files(self._PATHS, "L10")
        assert set(result) == {
            "/data/raw/dataset/L10/zf1/sVal.png",
            "/data/raw/dataset/L10/zf2/sVal.png",
        }, f"Unexpected result for L10: {result}"

    def test_result_is_in_natural_order(self):
        """Results are sorted in natural order."""
        paths = [f"/data/raw/dataset/L0/zf{i}/sVal.png" for i in range(1, 13)]
        result = select_light_stack_files(paths, "L0")
        zf_nums = [int(Path(f).parent.name.lstrip("zf")) for f in result]
        assert zf_nums == sorted(zf_nums), f"Not in natural order: {zf_nums}"

    def test_empty_input_returns_empty(self):
        """Empty file list returns empty list."""
        assert select_light_stack_files([], "L0") == []


# ===========================================================================
# MF-14: collect_light_dirs — detects L<n> at any path depth
# ===========================================================================


def test_collect_light_dirs_finds_lights_at_any_depth(tmp_path):
    """MF-14: num layout limpo L<n>/zf<m>/sVal.png o pai imediato é sempre zf*,
    então a detecção por pai imediato devolve vazio. A detecção correta acha o
    componente L<n> em QUALQUER posição do caminho relativo ao dataset."""
    from hybrid_stereo_method.hybrid.main import collect_light_dirs

    data = tmp_path / "synth"
    files = []
    for li in range(3):
        for k in range(2):
            d = data / f"L{li}" / f"zf{k}"
            d.mkdir(parents=True)
            f = d / "sVal.png"
            f.write_bytes(b"")
            files.append(str(f))
    (data / "lights.npy").write_bytes(b"")
    files.append(str(data / "lights.npy"))
    # detritos que NÃO são luzes: prefixo L sem dígito
    (data / "Lixo").mkdir()
    f = data / "Lixo" / "x.png"
    f.write_bytes(b"")
    files.append(str(f))

    assert collect_light_dirs(files, str(data)) == ["L0", "L1", "L2"]


def test_collect_light_dirs_natural_order(tmp_path):
    from hybrid_stereo_method.hybrid.main import collect_light_dirs

    data = tmp_path / "d"
    files = []
    for name in ["L10", "L2", "L1"]:
        d = data / name / "zf0"
        d.mkdir(parents=True)
        f = d / "sVal.png"
        f.write_bytes(b"")
        files.append(str(f))
    assert collect_light_dirs(files, str(data)) == ["L1", "L2", "L10"]


def test_collect_light_dirs_skips_files_outside_data_path(tmp_path, caplog):
    """Arquivo fora do data_path não pode injetar luz fantasma (ex.: um
    componente L5 no caminho ACIMA da raiz do dataset)."""
    import logging

    from hybrid_stereo_method.hybrid.main import collect_light_dirs

    data = tmp_path / "synth"
    d = data / "L0" / "zf0"
    d.mkdir(parents=True)
    inside = d / "sVal.png"
    inside.write_bytes(b"")
    outside_root = tmp_path / "L5" / "other"
    outside_root.mkdir(parents=True)
    outside = outside_root / "sVal.png"
    outside.write_bytes(b"")

    with caplog.at_level(logging.WARNING):
        result = collect_light_dirs([str(inside), str(outside)], str(data))
    assert result == ["L0"]
    assert any("skipping" in r.message for r in caplog.records)
