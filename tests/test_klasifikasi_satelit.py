import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from klasifikasi_satelit import (
    extract_color_features,
    load_eurosat,
    make_binary_labels,
    make_splits,
    save_metrics_plot,
)


class ClassificationHelpersTests(unittest.TestCase):
    def test_binary_labels_mark_only_forest_as_positive(self):
        labels = np.array([0, 1, 2, 1])

        actual = make_binary_labels(labels, forest_label=1)

        np.testing.assert_array_equal(actual, [0, 1, 0, 1])

    def test_splits_are_disjoint_and_stratified(self):
        labels = np.repeat(np.arange(10), 20)

        train, validation, test = make_splits(labels, seed=42)

        self.assertEqual((len(train), len(validation), len(test)), (140, 30, 30))
        self.assertEqual(len(set(train) & set(validation)), 0)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(set(validation) & set(test)), 0)
        for indices in (train, validation, test):
            class_counts = np.bincount(labels[indices], minlength=10)
            self.assertGreater(class_counts.min(), 0)
            self.assertLessEqual(class_counts.max() - class_counts.min(), 1)

    def test_color_features_have_expected_size_and_normalized_histograms(self):
        images = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        images[1, :, :, 0] = 255

        features = extract_color_features(images, bins=4)

        self.assertEqual(features.shape, (2, 18))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(features[:, 6:].reshape(2, 3, 4).sum(axis=2), 1)

    def test_load_eurosat_reads_class_folders_and_resizes_rgb_images(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            for class_name, color in (("Forest", (0, 120, 0)), ("River", (0, 0, 200))):
                class_dir = data_dir / class_name
                class_dir.mkdir()
                Image.new("RGB", (20, 30), color).save(class_dir / "sample.jpg")

            images, labels, class_names = load_eurosat(data_dir)

        self.assertEqual(class_names, ["Forest", "River"])
        self.assertEqual(images.shape, (2, 64, 64, 3))
        np.testing.assert_array_equal(labels, [0, 1])

    def test_save_metrics_plot_creates_png_from_model_metrics(self):
        results = {
            "models": {
                "random_forest": {
                    "accuracy": 0.9,
                    "precision_macro": 0.8,
                    "recall_macro": 0.7,
                    "f1_macro": 0.75,
                    "confusion_matrix": [[80, 10], [5, 5]],
                },
                "cnn": {
                    "accuracy": 0.85,
                    "precision_macro": 0.75,
                    "recall_macro": 0.8,
                    "f1_macro": 0.77,
                    "confusion_matrix": [[75, 15], [3, 7]],
                },
            }
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "results" / "metrics.png"

            save_metrics_plot(results, output)

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
