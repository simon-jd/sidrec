import numpy as np
import pytest
import torch

from sidrec.semantic_ids.embed import item_text
from sidrec.semantic_ids.rqvae import RQVAE, add_dedupe_code, code_stats


def test_dedupe_code_and_collision_stats():
    codes3 = np.array([[1, 2, 3], [1, 2, 3], [0, 0, 0], [1, 2, 3], [1, 2, 4]])
    codes4 = add_dedupe_code(codes3)
    assert codes4[:, 3].tolist() == [0, 1, 0, 2, 0]
    assert len({tuple(c) for c in codes4}) == 5
    stats = code_stats(codes3, codes=8)
    assert stats["distinct_3code_ids"] == 3 and stats["collision_rate"] == pytest.approx(0.4)
    assert stats["items_in_collisions"] == 3 and stats["largest_collision_group"] == 3
    assert [lv["codes_used"] for lv in stats["per_level"]] == [2, 2, 3]


def test_rqvae_codes_have_three_levels():
    torch.manual_seed(0)
    model = RQVAE(16, hidden=(8,), latent=4, levels=3, codes=5)
    x = torch.randn(40, 16)
    model.init_codebooks(x)
    recon, q_loss, codes = model(x)
    assert codes.shape == (40, 3) and int(codes.max()) < 5
    assert recon.item() > 0 and q_loss.item() >= 0


def test_item_text_uses_title_brand_price_categories():
    meta = {
        "title": "Rose &amp; Oud  Cream",
        "price": 12.5,
        "brand": "Acme",
        "categories": [["Beauty", "Skin Care"], ["Beauty", "Fragrance"]],
    }
    expected = "Title: Rose & Oud Cream; Brand: Acme; Price: 12.50; Categories: Beauty, Skin Care, Fragrance"
    assert item_text(meta) == expected
    assert item_text({"title": "X", "categories": []}) == "Title: X"
