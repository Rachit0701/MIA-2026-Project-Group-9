import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk

from mialab.filtering.postprocessing import remove_small_components, simple_crf_segmentation
from pipeline import _passes_postprocessing_gate, _plot_per_subject_results


class PostprocessingComparisonTests(unittest.TestCase):
    def test_removes_small_components_using_physical_volume(self):
        labels = np.zeros((5, 5, 5), dtype=np.uint8)
        labels[1, 1, 1:4] = 1
        labels[4, 4, 4] = 1
        labels[0, 0, 0] = 2
        image = sitk.GetImageFromArray(labels)
        image.SetSpacing((2.0, 1.0, 1.0))

        result = remove_small_components(image, min_volume_mm3=5.0)
        result_array = sitk.GetArrayFromImage(result)

        self.assertTrue(np.all(result_array[1, 1, 1:4] == 1))
        self.assertEqual(result_array[4, 4, 4], 0)
        self.assertEqual(result_array[0, 0, 0], 2)
        self.assertEqual(result.GetSpacing(), image.GetSpacing())

    def test_uses_26_connectivity_for_diagonal_voxels(self):
        labels = np.zeros((3, 3, 3), dtype=np.uint8)
        labels[0, 0, 0] = 1
        labels[1, 1, 1] = 1
        image = sitk.GetImageFromArray(labels)

        result = remove_small_components(image, min_volume_mm3=2.0)

        self.assertEqual(np.count_nonzero(sitk.GetArrayFromImage(result) == 1), 2)

    def test_preserves_largest_component_if_all_are_below_threshold(self):
        labels = np.zeros((5, 5, 5), dtype=np.uint8)
        labels[0, 0, 0:2] = 1
        labels[4, 4, 4] = 1
        image = sitk.GetImageFromArray(labels)

        result = remove_small_components(image, min_volume_mm3=5.0)
        result_array = sitk.GetArrayFromImage(result)

        self.assertEqual(np.count_nonzero(result_array == 1), 2)
        self.assertEqual(result_array[4, 4, 4], 0)

    def test_rejects_non_positive_component_volume(self):
        image = sitk.GetImageFromArray(np.zeros((2, 2, 2), dtype=np.uint8))

        with self.assertRaisesRegex(ValueError, 'finite positive'):
            remove_small_components(image, min_volume_mm3=0)

    def test_simplified_crf_returns_geometry_matched_label_image(self):
        probability_array = np.empty((3, 3, 3, 2), dtype=np.float32)
        probability_array[..., 0] = 0.1
        probability_array[..., 1] = 0.9
        probability = sitk.GetImageFromArray(probability_array, isVector=True)
        reference = sitk.GetImageFromArray(np.zeros((3, 3, 3), dtype=np.uint8))
        reference.SetSpacing((1.5, 1.0, 2.0))

        result = simple_crf_segmentation(probability, reference)

        np.testing.assert_array_equal(sitk.GetArrayFromImage(result), 1)
        self.assertEqual(result.GetSpacing(), reference.GetSpacing())

    def test_gate_requires_hd95_improvement_and_dice_guardrail(self):
        labels = ('Amygdala', 'GreyMatter')
        baseline = {
            'Amygdala': {'DICE': 0.55, 'HDRFDST': 12.0},
            'GreyMatter': {'DICE': 0.72, 'HDRFDST': 2.0},
        }
        candidate = {
            'Amygdala': {'DICE': 0.546, 'HDRFDST': 11.5},
            'GreyMatter': {'DICE': 0.72, 'HDRFDST': 2.0},
        }

        self.assertTrue(_passes_postprocessing_gate(baseline, candidate, labels))
        candidate['Amygdala']['DICE'] = 0.544
        self.assertFalse(_passes_postprocessing_gate(baseline, candidate, labels))

    def test_plot_writes_per_subject_figure(self):
        labels = ('WhiteMatter', 'GreyMatter', 'Hippocampus', 'Amygdala', 'Thalamus')
        arms = (
            ('subject-1', 0.01),
            ('subject-1-PP', 0.02),
            ('subject-1-PP-components-25mm3', 0.03),
        )
        results = []
        for result_id, offset in arms:
            for label_index, label in enumerate(labels):
                results.append(SimpleNamespace(
                    id_=result_id, label=label, metric='DICE', value=0.7 + offset + label_index * 0.01))
                results.append(SimpleNamespace(
                    id_=result_id, label=label, metric='HDRFDST', value=2.0 + label_index + offset))

        with tempfile.TemporaryDirectory() as temp_dir:
            plot_path = os.path.join(temp_dir, 'per_subject.png')
            _plot_per_subject_results(results, ['subject-1'], plot_path, dense_crf_available=False)

            self.assertTrue(os.path.isfile(plot_path))
            self.assertGreater(os.path.getsize(plot_path), 0)


if __name__ == '__main__':
    unittest.main()
