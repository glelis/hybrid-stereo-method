"""z_foc resolvido a partir das pastas zf* (item 1 da revisão de 80e3de3).

Um z_foc de outro dataset (melon24 [15..125] num lamb-gls [7.2..288.8])
colocava zMos e os hints absolutos em unidades erradas sem nenhum aviso.
"""

import pytest

from hybrid_stereo_method.infrastructure.utils import parse_zf_value, resolve_z_foc

LAMB = [f"zf{z:07.4f}-df051.2000" for z in (7.2, 32.8, 58.4, 84.0)]


def test_parse_zf_from_dir_file_and_averaged_frame():
    assert parse_zf_value("zf045.0000-df020.0000") == 45.0
    assert parse_zf_value("/d/L003/zf007.2000-df051.2000/sVal.png") == 7.2
    assert parse_zf_value("/out/average/images/average_zf288.8000-df051.2000.png") == 288.8
    assert parse_zf_value("/d/images/img_001.png") is None


def test_auto_parses_folder_values_in_order():
    assert resolve_z_foc("auto", LAMB) == [7.2, 32.8, 58.4, 84.0]
    assert resolve_z_foc(None, LAMB) == [7.2, 32.8, 58.4, 84.0]


def test_matching_list_is_accepted():
    assert resolve_z_foc([7.2, 32.8, 58.4, 84.0], LAMB) == [7.2, 32.8, 58.4, 84.0]


def test_list_from_another_dataset_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        resolve_z_foc([15.0, 25.0, 35.0, 45.0], LAMB)
    with pytest.raises(ValueError, match="does not match"):
        resolve_z_foc([7.2, 32.8, 58.4], LAMB)


def test_flat_dataset_keeps_list_and_rejects_auto():
    flat = ["/d/images/a.png", "/d/images/b.png"]
    assert resolve_z_foc([1, 2], flat) == [1.0, 2.0]
    with pytest.raises(ValueError, match="auto"):
        resolve_z_foc("auto", flat)
