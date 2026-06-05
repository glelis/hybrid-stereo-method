# tests/test_hybrid_path_selection.py
"""Tests for MF-01 and MF-02 fixes: exact path-component filtering and natural ordering.

MF-01: substring match `"zf1" in file` also matches zf10/zf11/zf12 with >=10 planes.
MF-02: sorted() is lexicographic; with 12 planes the order is zf1,zf10,zf11,zf12,zf2,...,
       misaligning average_images_paths from the YAML z_foc list (natural ascending order).
"""
import os

import pytest

from hybrid_stereo_method.hybrid.main import collect_dirs_with_prefix, select_files_by_parent_dir


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
    "/data/raw/dataset/L0/selected-pixels.png",        # parent = L0, not a zf dir
    "/data/raw/dataset/L0/zf1/other_file.png",         # wrong filename
    "/data/raw/dataset/lights.npy",                    # top-level, no zf parent
    "/data/raw/dataset/sharp/hAvg.png",                # no zf/L parent at all
]


# ===========================================================================
# MF-01 regression: "zf1" must NOT match files under zf10/zf11/zf12
# ===========================================================================

class TestSelectFilesByParentDir:
    """Tests for select_files_by_parent_dir (fixes MF-01)."""

    def test_zf1_does_not_match_zf10_zf11_zf12(self):
        """Regression for MF-01: with 12 planes selecting 'zf1' must return only zf1 files."""
        result = select_files_by_parent_dir(PATHS_12L_12ZF, dir_name="zf1", filename_part="sVal.png")
        # Should be exactly 12 files (one per light) — all with parent zf1, none from zf10/11/12
        assert len(result) == 12, (
            f"Expected 12 files (one per light in zf1), got {len(result)}: {result}"
        )
        for path in result:
            parent = os.path.basename(os.path.dirname(path))
            assert parent == "zf1", (
                f"Expected parent dir 'zf1', got '{parent}' for path: {path}"
            )

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
        result = select_files_by_parent_dir(PATHS_12L_12ZF, dir_name="zf1", filename_part="sVal.png")
        # Extract the light number from each path
        light_nums = [
            int(os.path.basename(os.path.dirname(os.path.dirname(p))).lstrip("L"))
            for p in result
        ]
        assert light_nums == sorted(light_nums), (
            f"Expected natural order of light dirs, got: {light_nums}"
        )

    def test_empty_input_returns_empty(self):
        """Edge case: empty file list returns empty list."""
        result = select_files_by_parent_dir([], dir_name="zf1", filename_part="sVal.png")
        assert result == []

    def test_no_match_returns_empty(self):
        """Non-existent dir_name returns empty list."""
        result = select_files_by_parent_dir(PATHS_12L_12ZF, dir_name="zf99", filename_part="sVal.png")
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
        assert result == expected, (
            f"Natural sort expected {expected}, got {result}"
        )

    def test_lexicographic_order_is_WRONG(self):
        """Confirm that sorted() would give the wrong order (the bug MF-02 describes)."""
        dirs = {os.path.basename(os.path.dirname(p)) for p in PATHS_12L_12ZF if
                os.path.basename(os.path.dirname(p)).startswith("zf")}
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
        assert result == expected, (
            f"Natural sort expected {expected}, got {result}"
        )

    def test_light_dirs_lexicographic_order_is_WRONG(self):
        """Confirm sorted() gives wrong order for L0..L11 (same class of bug as MF-02 for zf)."""
        dirs = {os.path.basename(os.path.dirname(p)) for p in PATHS_12L_DIRECT if
                os.path.basename(os.path.dirname(p)).startswith("L")}
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
