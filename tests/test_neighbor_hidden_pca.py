import numpy as np
import pandas as pd
import torch
from torch import nn

from src.kronos_continual.neighbor_hidden_pca import (
    NeighborForecast, fit_context, fit_neighbors, peer_mean,
)


def test_peers_are_sorted_exclude_self_and_use_present_stocks_only():
    r = pd.DataFrame({"C": [1, 2, 3, 4], "B": [1, 2, 3, 4],
                      "A": [1, 2, 3, 4], "Z": [4, 3, 2, 1]})
    neighbors = fit_neighbors(r, k=2, overlap=4)
    assert neighbors["A"] == ("B", "C")
    assert neighbors["Z"] == ()
    x, count = peer_mean(["C", "Z", "A"], np.array([[3.], [50.], [100.]]), neighbors)
    np.testing.assert_array_equal(count, [1, 0, 1])
    np.testing.assert_array_equal(x[:, 0], [100., 0., 3.])


def test_fixed_center_and_eigenweight_preserve_relative_direction():
    x = np.array([[3., 6.], [-1., 6.], [1., 7.], [1., 5.]])
    ctx = fit_context([(x, np.ones(4, dtype=int))], rank=2)
    np.testing.assert_allclose(ctx.mean, [1., 6.])
    np.testing.assert_allclose(ctx.eigen, [2., .5])
    c, d = ctx.features(np.array([[3., 7.], [100., 100.]]), np.array([2, 0]))
    np.testing.assert_allclose(c[0, 0] / c[0, 1], 2.)
    np.testing.assert_allclose(d[0, 0] / d[0, 1], 8.)
    for feat in (c, d):
        np.testing.assert_array_equal(feat[1], 0.)


def test_neighbor_head_uses_only_peers_and_starts_at_zero():
    adapter = nn.Identity()
    base = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        base.weight.fill_(1.)
    model = NeighborForecast(adapter, base, context_width=2)
    joined = torch.tensor([[2., 3., 4., 6.], [10., 20., 4., 6.], [2., 3., 0., 0.]])
    base_output = base(joined[:, :2])
    torch.testing.assert_close(model(joined), base_output)
    with torch.no_grad():
        model.head.neighbor.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [0., 0.]]))
    torch.testing.assert_close(
        model(joined) - base_output, torch.tensor([[4., 6., 0.], [4., 6., 0.], [0., 0., 0.]]))
