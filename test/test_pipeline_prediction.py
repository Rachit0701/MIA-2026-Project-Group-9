import unittest

import numpy as np

from pipeline import _predict_in_chunks


class TrackingClassifier:
    classes_ = np.array([2, 4, 7])

    def __init__(self):
        self.batch_sizes = []

    def predict(self, features):
        raise AssertionError('predict should not be called separately')

    def predict_proba(self, features):
        self.batch_sizes.append(features.shape[0])
        return features[:, :3].astype(np.float64)


class ChunkedPredictionTests(unittest.TestCase):
    def test_predicts_bounded_chunks_once_and_returns_float32_probabilities(self):
        classifier = TrackingClassifier()
        features = np.array([
            [0.1, 0.2, 0.7],
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.5, 0.4, 0.1],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        predictions, probabilities = _predict_in_chunks(classifier, features, chunk_size=2)

        np.testing.assert_array_equal(predictions, [7, 2, 4, 2, 7])
        np.testing.assert_allclose(probabilities, features)
        self.assertEqual(probabilities.dtype, np.float32)
        self.assertEqual(classifier.batch_sizes, [2, 2, 1])

    def test_rejects_non_positive_chunk_size(self):
        with self.assertRaisesRegex(ValueError, 'chunk_size must be positive'):
            _predict_in_chunks(TrackingClassifier(), np.ones((1, 3)), chunk_size=0)


if __name__ == '__main__':
    unittest.main()
