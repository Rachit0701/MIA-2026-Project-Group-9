"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import datetime
import os
import sys
import timeit
import warnings

import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
import numpy as np
import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer

try:
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]  # the list of data we will load

PREDICTION_CHUNK_SIZE = 250_000
COMPONENT_MIN_VOLUME_MM3 = 25.0


def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - Decision forest classifier model building
        - Segmentation using the decision forest classifier model on unseen images
        - Post-processing of the segmentation
        - Evaluation of the segmentation
    """

    # load atlas images
    putil.load_atlas_images(data_atlas_dir)

    print('-' * 5, 'Training...')

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_train_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())
    pre_process_params = {'skullstrip_pre': True,
                          'normalization_pre': True,
                          'registration_pre': True,
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True}

    # load images for training and pre-process
    images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    # generate feature matrix and label vector
    data_train = np.concatenate([img.feature_matrix[0] for img in images])
    labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()

    warnings.warn('Random forest parameters not properly set.')
    forest = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1],
                                                n_estimators=10,
                                                max_depth=20, random_state=42) #changed + rando state

    start_time = timeit.default_timer()
    forest.fit(data_train, labels_train)
    print(' Time elapsed:', timeit.default_timer() - start_time, 's')
    del images, data_train, labels_train

    # create a result directory with timestamp
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    print('-' * 5, 'Testing...')

    # initialize evaluator
    evaluator = putil.init_evaluator()

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_test_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())

    # Pre-process and evaluate one subject at a time to limit peak memory.
    pre_process_params['training'] = False
    test_subject_ids = []
    dense_crf_available = putil.pydensecrf_available()
    if not dense_crf_available:
        print('DenseCRF comparison unavailable: pydensecrf is not installed.')

    for subject_id, paths in crawler.data.items():
        img = putil.pre_process_batch({subject_id: paths}, pre_process_params, multi_process=False)[0]
        test_subject_ids.append(img.id_)
        print('-' * 10, 'Testing', img.id_)

        start_time = timeit.default_timer()
        predictions, probabilities = _predict_in_chunks(forest, img.feature_matrix[0])
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')
        img.feature_matrix = None

        # convert prediction and probabilities back to SimpleITK images
        image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions, img.image_properties)
        image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)
        del predictions, probabilities

        # evaluate segmentation without post-processing
        evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)
        sitk.WriteImage(image_prediction, os.path.join(result_dir, img.id_ + '_SEG.mha'), True)

        baseline = putil.post_process_batch(
            [img], [image_prediction], [image_probabilities],
            {'simple_crf_post': True}, multi_process=False)[0]
        evaluator.evaluate(baseline, img.images[structure.BrainImageTypes.GroundTruth], img.id_ + '-PP')
        sitk.WriteImage(baseline, os.path.join(result_dir, img.id_ + '_SEG-PP.mha'), True)

        component_result = putil.post_process_batch(
            [img], [baseline], [image_probabilities],
            {'small_components_post': True,
             'component_min_volume_mm3': COMPONENT_MIN_VOLUME_MM3},
            multi_process=False)[0]
        component_id = img.id_ + '-PP-components-25mm3'
        evaluator.evaluate(component_result, img.images[structure.BrainImageTypes.GroundTruth], component_id)
        sitk.WriteImage(
            component_result,
            os.path.join(result_dir, img.id_ + '_SEG-PP-components-25mm3.mha'), True)

        if dense_crf_available:
            post_process_params = {'crf_post': True}
            start_time = timeit.default_timer()
            dense_crf_result = putil.post_process_batch(
                [img], [image_prediction], [image_probabilities],
                post_process_params, multi_process=False)[0]
            print(' DenseCRF time elapsed: {:.3f} s'.format(
                timeit.default_timer() - start_time))
            dense_crf_id = img.id_ + '-PP-pydensecrf'
            evaluator.evaluate(dense_crf_result, img.images[structure.BrainImageTypes.GroundTruth], dense_crf_id)
            sitk.WriteImage(
                dense_crf_result,
                os.path.join(result_dir, img.id_ + '_SEG-PP-pydensecrf.mha'), True)
            del dense_crf_result

        del baseline, component_result
        del img, image_prediction, image_probabilities

    _report_postprocessing_comparison(evaluator.results, test_subject_ids, dense_crf_available)

    # use two writers to report the results
    os.makedirs(result_dir, exist_ok=True)  # generate result directory, if it does not exists
    result_file = os.path.join(result_dir, 'results.csv')
    writer.CSVWriter(result_file).write(evaluator.results)
    plot_file = os.path.join(result_dir, 'postprocessing_per_subject.png')
    _plot_per_subject_results(evaluator.results, test_subject_ids, plot_file, dense_crf_available)
    print('Per-subject plot saved to {}'.format(plot_file))

    print('\nSubject-wise results...')
    writer.ConsoleWriter().write(evaluator.results)

    # report also mean and standard deviation among all subjects
    result_summary_file = os.path.join(result_dir, 'results_summary.csv')
    functions = {'MEAN': np.mean, 'STD': np.std}
    writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
    print('\nAggregated statistic results...')
    writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

    # clear results such that the evaluator is ready for the next evaluation
    evaluator.clear()


def _predict_in_chunks(classifier, features, chunk_size: int = PREDICTION_CHUNK_SIZE):
    """Predict labels and probabilities in bounded batches to control peak memory."""
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')

    sample_count = features.shape[0]
    if sample_count == 0:
        raise ValueError('features must contain at least one sample')

    predictions = np.empty(sample_count, dtype=np.uint8)
    probabilities = None

    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        chunk_probabilities = classifier.predict_proba(features[start:stop])
        if chunk_probabilities.shape[0] != stop - start:
            raise ValueError('classifier returned an unexpected number of probability rows')

        if probabilities is None:
            probabilities = np.empty(
                (sample_count, chunk_probabilities.shape[1]), dtype=np.float32)
        elif chunk_probabilities.shape[1] != probabilities.shape[1]:
            raise ValueError('classifier returned inconsistent probability columns')

        class_indices = np.argmax(chunk_probabilities, axis=1)
        predictions[start:stop] = classifier.classes_[class_indices].astype(np.uint8, copy=False)
        probabilities[start:stop] = chunk_probabilities

    return predictions, probabilities


def _default_data_dir(script_dir: str) -> str:
    """Find a supported dataset layout, preferring the conventional data directory."""
    data_dirs = [
        os.path.normpath(os.path.join(script_dir, '../data')),
        os.path.normpath(os.path.join(script_dir, '../MIAlab_data/MIALab_2020_Dataset')),
    ]
    required_dirs = ('atlas', 'train', 'test')

    for data_dir in data_dirs:
        if all(os.path.isdir(os.path.join(data_dir, name)) for name in required_dirs):
            return data_dir

    return data_dirs[0]


def _passes_postprocessing_gate(baseline, candidate, labels):
    """Apply the declared mean-HD95 improvement and Dice non-inferiority gate."""
    hd95_improved = any(candidate[label]['HDRFDST'] < baseline[label]['HDRFDST'] for label in labels)
    dice_guardrail = all(
        candidate[label]['DICE'] >= baseline[label]['DICE'] - 0.005
        for label in labels)
    return hd95_improved and dice_guardrail


def _report_postprocessing_comparison(results, subject_ids, dense_crf_available):
    """Report mean per-tissue metrics and the predeclared comparison gate."""
    labels = ('WhiteMatter', 'GreyMatter', 'Hippocampus', 'Amygdala', 'Thalamus')
    metrics = ('DICE', 'HDRFDST')
    arm_suffixes = {
        'component-removal': '-PP-components-25mm3',
    }
    if dense_crf_available:
        arm_suffixes['pydensecrf'] = '-PP-pydensecrf'
    result_by_key = {}

    for result in results:
        key = (result.id_, result.label, result.metric)
        if key in result_by_key:
            raise ValueError('Duplicate evaluation result: {}'.format(key))
        result_by_key[key] = float(result.value)

    means = {'baseline': {}, 'component-removal': {}}
    if dense_crf_available:
        means['pydensecrf'] = {}

    for arm, suffix in [('baseline', '-PP')] + list(arm_suffixes.items()):
        means[arm] = {}
        for label in labels:
            means[arm][label] = {}
            for metric in metrics:
                values = []
                for subject_id in subject_ids:
                    key = (subject_id + suffix, label, metric)
                    if key not in result_by_key:
                        raise ValueError('Missing evaluation result: {}'.format(key))
                    values.append(result_by_key[key])
                means[arm][label][metric] = float(np.mean(values))

    print('\nPost-processing comparison (mean across test subjects)')
    for label in labels:
        baseline = means['baseline'][label]
        component = means['component-removal'][label]
        print('{}: baseline Dice {:.4f}, HD95 {:.4f}; component removal '
              'Dice {:.4f} ({:+.4f}), HD95 {:.4f} ({:+.4f})'.format(
                  label, baseline['DICE'], baseline['HDRFDST'],
                  component['DICE'], component['DICE'] - baseline['DICE'],
                  component['HDRFDST'], component['HDRFDST'] - baseline['HDRFDST']))
        if dense_crf_available:
            dense = means['pydensecrf'][label]
            print('  pydensecrf Dice {:.4f} ({:+.4f}), HD95 {:.4f} ({:+.4f})'.format(
                dense['DICE'], dense['DICE'] - baseline['DICE'],
                dense['HDRFDST'], dense['HDRFDST'] - baseline['HDRFDST']))

    for arm in means:
        if arm == 'baseline':
            continue
        passed = _passes_postprocessing_gate(means['baseline'], means[arm], labels)
        print('{} evaluation gate: {}'.format(arm, 'PASS' if passed else 'FAIL'))


def _plot_per_subject_results(results, subject_ids, output_path, dense_crf_available):
    """Save paired per-subject Dice and HD95 plots for each tissue."""
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    labels = ('Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter')
    metrics = (('DICE', 'Dice (higher is better)'), ('HDRFDST', 'HD95 (lower is better)'))
    methods = [
        ('Raw forest', '', '#777777', 'o'),
        ('Simplified CRF (compat=2.0)', '-PP', '#2878b5', 's'),
        ('CRF + remove <25 mm^3', '-PP-components-25mm3', '#e07a1f', '^'),
    ]
    if dense_crf_available:
        methods.append(('DenseCRF', '-PP-pydensecrf', '#2a9d8f', 'D'))

    result_by_key = {}
    for result in results:
        key = (result.id_, result.label, result.metric)
        if key in result_by_key:
            raise ValueError('Duplicate evaluation result: {}'.format(key))
        result_by_key[key] = float(result.value)

    if not subject_ids:
        raise ValueError('At least one test subject is required to create the plot')

    figure, axes = plt.subplots(len(labels), len(metrics), figsize=(17, 15), sharex='col')
    offsets = np.linspace(-0.22, 0.22, len(methods))
    try:
        for row_index, label in enumerate(labels):
            for col_index, (metric, metric_title) in enumerate(metrics):
                axis = axes[row_index, col_index]
                panel_values = []
                for subject_index, subject_id in enumerate(subject_ids):
                    method_points = []
                    for method_index, (method_name, suffix, color, marker) in enumerate(methods):
                        result_id = subject_id + suffix
                        key = (result_id, label, metric)
                        if key not in result_by_key:
                            raise ValueError('Missing evaluation result: {}'.format(key))
                        value = result_by_key[key]
                        x_position = subject_index + offsets[method_index]
                        panel_values.append(value)
                        method_points.append((x_position, value))
                        axis.scatter(x_position, value, color=color, marker=marker, s=35, zorder=3)

                    axis.plot(
                        [method_points[1][0], method_points[2][0]],
                        [method_points[1][1], method_points[2][1]],
                        color='#aaaaaa', linewidth=0.8, zorder=1)

                axis.set_title(metric_title if row_index == 0 else '')
                axis.set_ylabel(label)
                axis.set_xticks(range(len(subject_ids)))
                axis.set_xticklabels(subject_ids, rotation=55, ha='right')
                axis.grid(axis='y', alpha=0.25)
                axis.margins(x=0.04)
                if metric == 'DICE':
                    axis.set_ylim(0, 1)
                else:
                    axis.set_ylim(0, max(panel_values) * 1.12)
                if row_index == len(labels) - 1:
                    axis.set_xlabel('Test subject')

        legend_handles = [
            plt.Line2D([0], [0], color=color, marker=marker, linestyle='None', label=method_name)
            for method_name, _, color, marker in methods
        ]
        figure.legend(handles=legend_handles, loc='upper center', ncol=len(methods), frameon=False,
                      bbox_to_anchor=(0.5, 0.985))
        figure.suptitle('Per-subject segmentation results', y=1.01, fontsize=16)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(output_path, dpi=180, bbox_inches='tight')
    finally:
        plt.close(figure)


if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])
    data_dir = _default_data_dir(script_dir)

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.join(data_dir, 'atlas'),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.join(data_dir, 'train'),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.join(data_dir, 'test'),
        help='Directory with testing data.'
    )

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir)
