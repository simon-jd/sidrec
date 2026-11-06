import numpy as np
import pytest
import torch

from sidrec.semantic_ids.embed import item_text
from sidrec.semantic_ids.rqvae import RQVAE, add_dedupe_code, code_stats, train


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


def test_rqvae_training_learns_to_reconstruct_not_just_the_mean():
    # Normalized vectors around 64 directions plus a shared offset, like sentence embeddings. A decoder that
    # outputs the mean explains none of the variance; the trained one must explain a good share of it.
    g = torch.Generator().manual_seed(0)
    centers = torch.randn(64, 48, generator=g)
    x = centers[torch.randint(0, 64, (2048,), generator=g)] + 0.3 * torch.randn(2048, 48, generator=g)
    x = torch.nn.functional.normalize(x + 1.0, dim=1)
    model, log = train(x, epochs=200, seed=0)
    with torch.no_grad():
        z_hat = model.quantize(model.encoder(x))[0]
        x_hat = model.decoder(z_hat)
    explained = 1 - ((x - x_hat) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
    assert explained.item() > 0.7
    assert log[-1]["codes_used"][0] > 16
