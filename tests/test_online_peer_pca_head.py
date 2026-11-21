import numpy as np

from src.kronos_continual.online_peer_pca_head import OnlinePCA, aggregate_neighbors, correlation_weights


def test_all_stock_weights_match_pairwise_correlation_with_missing_data():
    from src.kronos_continual.online_peer_pca_head import cross_section_weights, weighted_feature_mean

    history = np.array([[0., 0., 3.], [1., 1., 2.], [2., np.nan, 1.], [3., 3., 0.]])
    neighbors = np.array([[1, 2], [0, 2], [0, 1]])
    expected = correlation_weights(history, neighbors, minimum=3)
    actual = cross_section_weights(history, minimum=3)
    np.testing.assert_allclose(np.take_along_axis(actual, neighbors, axis=1), expected)
    np.testing.assert_array_equal(np.diag(actual), 0.)
    hidden = np.array([[1., 2.], [3., 6.], [5., 10.]])
    available = ~np.eye(3, dtype=bool)
    values, counts = weighted_feature_mean(hidden, available, actual)
    np.testing.assert_array_equal(values, [[3., 6.], [1., 2.], [2., 4.]])
    np.testing.assert_array_equal(counts, [2, 2, 2])


def test_weights_follow_trailing_correlation_and_exclude_missing_neighbors():
    history = np.column_stack([np.arange(8), np.arange(8), -np.arange(8)])
    ids = np.array([[1, 2], [-1, -1], [-1, -1]])
    scores = correlation_weights(history, ids, minimum=4)
    assert scores[0, 0] > 0 and scores[0, 1] == 0
    hidden = np.array([[0., 0.], [2., 4.], [6., 8.]])
    values, counts = aggregate_neighbors(hidden, np.arange(3), ids, scores)
    np.testing.assert_array_equal(values, [[2., 4.], [0., 0.], [0., 0.]])
    np.testing.assert_array_equal(counts, [2, 0, 0])
    values, _ = aggregate_neighbors(hidden[[0, 2]], np.array([0, 2]), ids, scores[[0, 2]])
    np.testing.assert_array_equal(values, [[6., 8.], [0., 0.]])


def test_pca_uses_previous_state_and_skips_rows_without_neighbors():
    pca = OnlinePCA(np.zeros(2), np.array([[1.], [0.]]), np.ones(1), decay=.5)
    values = np.array([[1., 4.], [1., -4.], [1000., 1000.]])
    available = np.array([True, True, False])
    expected = [[1., 0.], [1., 0.], [0., 0.]]
    np.testing.assert_array_equal(pca.reconstruct(values, available), expected)
    pca.update(values, available)
    np.testing.assert_allclose(pca.reconstruct(values, available), [[0., 4.], [0., -4.], [0., 0.]], atol=1e-12)
