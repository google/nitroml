# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
# Lint as: python3
"""Tests for nitroml.automl.ensemble_selection.lib.ensemble_selection."""

import os
import tempfile
from typing import Any, Dict, List

from absl.testing import absltest
from nitroml.automl.ensemble_selection.lib import ensemble_selection
import numpy as np
import tensorflow as tf

from nitroml.protos import problem_statement_pb2 as ps_pb2


def _get_serve_tf_examples_fn(model: tf.keras.Model):
  """Returns a function that parses a serialized tf.Example and applies TFT."""

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples) -> tf.keras.Model:
    """Returns the output to be used in the serving signature."""

    feature_spec = make_raw_feature_spec()
    feature_spec.pop('median_house_value')
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    return model(parsed_features)

  return serve_tf_examples_fn


def get_feature_columns() -> List[Any]:
  column_names = [
      'longitude', 'latitude', 'housing_median_age', 'total_rooms',
      'total_bedrooms', 'population', 'households', 'median_income',
      'median_house_value'
  ]
  return [tf.feature_column.numeric_column(col) for col in column_names]


def make_raw_feature_spec() -> Dict[str, tf.io.FixedLenFeature]:
  return tf.feature_column.make_parse_example_spec(get_feature_columns())


def get_dataset(file_pattern: str) -> tf.data.Dataset:
  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=10,
      features=make_raw_feature_spec(),
      label_key='median_house_value',
      shuffle=False)

  return dataset.cache()


def build_model(seed: int) -> tf.keras.Model:
  """Create and compile a simple linear regression model."""
  feature_layer = tf.keras.layers.DenseFeatures(
      [tf.feature_column.numeric_column('median_income')])
  model = tf.keras.models.Sequential([
      feature_layer,
      tf.keras.layers.Dense(
          units=32,
          activation='relu',
          kernel_initializer=tf.keras.initializers.RandomNormal(seed=seed)),
      tf.keras.layers.Dense(
          units=1,
          input_shape=(1,),
          kernel_initializer=tf.keras.initializers.RandomNormal(seed=seed))
  ])
  model.compile(
      optimizer=tf.keras.optimizers.RMSprop(lr=0.05),
      loss='mse',
      metrics=[
          tf.keras.metrics.RootMeanSquaredError(),
          tf.keras.metrics.MeanSquaredError(),
          tf.keras.metrics.MeanAbsoluteError()
      ])
  return model


def _test_predict_fn(loaded_model: ensemble_selection.LoadedSavedModel,
                     x: tf.Tensor) -> tf.Tensor:
  return loaded_model.signatures['serving_default'](x)['output_0']


class EnsembleSelectionTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.data_path = os.path.join(os.path.dirname(__file__), 'testdata')
    train_path = os.path.join(cls.data_path, 'train_examples.tfrecord')
    export_dir = os.path.join(tempfile.mkdtemp(), 'CA_housing_model/')

    cls.saved_model_paths = {}
    for i in range(5):
      model = build_model(seed=i)
      train_dataset = get_dataset(train_path)
      model.fit(train_dataset, epochs=12, steps_per_epoch=100)

      signatures = {
          'serving_default':
              _get_serve_tf_examples_fn(model).get_concrete_function(
                  tf.TensorSpec(shape=[None], dtype=tf.string,
                                name='examples')),
      }

      cls.saved_model_paths[str(i)] = os.path.join(export_dir, str(i))
      model.save(
          cls.saved_model_paths[str(i)],
          save_format='tf',
          signatures=signatures)

    fitdata_dir = os.path.join(cls.data_path, 'fit_examples.tfrecord')
    fit_dataset = tf.data.TFRecordDataset(fitdata_dir)
    cls.fit_examples = np.asarray(list(fit_dataset.as_numpy_iterator()))

    eval_dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=fitdata_dir,
        batch_size=10,  # size of validation dataset
        features=make_raw_feature_spec(),
        label_key='median_house_value',
        shuffle=False).take(1)
    cls.fit_label = list(eval_dataset.as_numpy_iterator())[0][1]

  def test_get_predictions(self):
    # TODO(liumich): improve test predictions with the following steps
    # - reduce the number of samples to 4-5
    # - manually compute MSE after each iteration for each partial ensemble
    # - also output the ground truth (labels) so that we can verify
    es = ensemble_selection.EnsembleSelection(
        problem_statement=ps_pb2.ProblemStatement(tasks=[
            ps_pb2.Task(
                type=ps_pb2.Type(
                    one_dimensional_regression=ps_pb2.OneDimensionalRegression(
                        label='label')))
        ]),
        saved_model_paths=self.saved_model_paths,
        predict_fn=_test_predict_fn,
        ensemble_size=3,
        metric=tf.keras.metrics.MeanSquaredError(),
        goal='minimize')
    want_predictions = {
        '0':
            np.array([[268520.7], [172055.8], [172840.52], [203374.36],
                      [629715.5], [160393.], [242507.27], [156286.08],
                      [262261.7], [221169.3]]),
        '1':
            np.array([[262822.53], [168104.17], [168874.69], [198855.67],
                      [617477.56], [156652.53], [237280.08], [152620.],
                      [256676.81], [216328.45]]),
        '2':
            np.array([[247936.98], [158487.19], [159214.84], [187528.2],
                      [582864.9], [147672.52], [223815.3], [143864.3],
                      [242133.11], [204029.08]]),
        '3':
            np.array([[268206.75], [171761.81], [172546.36], [203073.86],
                      [629326.7], [160101.38], [242198.66], [155995.36],
                      [261948.98], [220865.14]]),
        '4':
            np.array([[257493.5], [164639.19], [165394.55], [194785.5],
                      [605169.06], [153412.9], [232453.73], [149459.73],
                      [251468.73], [211914.42]])
    }

    predictions = es._get_predictions_dict(self.fit_examples)

    for model_id in predictions.keys():
      np.testing.assert_array_almost_equal(want_predictions[model_id],
                                           predictions[model_id], 1)

  def test_lifecycle(self):
    es = ensemble_selection.EnsembleSelection(
        problem_statement=ps_pb2.ProblemStatement(tasks=[
            ps_pb2.Task(
                type=ps_pb2.Type(
                    one_dimensional_regression=ps_pb2.OneDimensionalRegression(
                        label='label')))
        ]),
        saved_model_paths=self.saved_model_paths,
        predict_fn=_test_predict_fn,
        ensemble_size=3,
        metric=tf.keras.metrics.MeanSquaredError(),
        goal='minimize')
    test_dir = os.path.join(self.data_path, 'test_examples.tfrecord')
    test_examples = np.asarray(
        list(tf.data.TFRecordDataset(test_dir).as_numpy_iterator()))
    test_examples_tensor = tf.convert_to_tensor(test_examples)
    model_predictions = {}
    for model_id, path in self.saved_model_paths.items():
      reloaded_model = tf.saved_model.load(path)
      model_predictions[model_id] = reloaded_model.signatures[
          'serving_default'](test_examples_tensor)['output_0'].numpy()
    want_weights = {'2': 0.3333333333333333, '4': 0.6666666666666666}
    want_prediction = want_weights['2'] * model_predictions['2'] + want_weights[
        '4'] * model_predictions['4']
    mse = tf.keras.metrics.MeanSquaredError()
    mse_scores = []
    for pred in model_predictions.values():
      mse_scores.append(mse(self.fit_label, pred))
      mse.reset_states()
    export_dir = os.path.join(
        tempfile.mkdtemp(dir=absltest.get_default_test_tmpdir()),
        'from_estimator')

    es.fit(self.fit_examples, self.fit_label)
    ensemble_predictions = es.predict(test_examples)
    ensemble_mse = es.evaluate(self.fit_examples, self.fit_label, [mse])[0]
    ensemble_path = es.save(export_dir)
    reloaded_ensemble = tf.saved_model.load(ensemble_path)
    loaded_ensemble_prediction = reloaded_ensemble.signatures[
        'serving_default'](input=test_examples_tensor)['output'].numpy()

    self.assertEqual(want_weights, es.weights)
    self.assertEqual((10, 1), ensemble_predictions.shape)
    np.testing.assert_array_almost_equal(want_prediction, ensemble_predictions,
                                         1)
    self.assertLessEqual(ensemble_mse, min(mse_scores))
    np.testing.assert_array_almost_equal(ensemble_predictions,
                                         loaded_ensemble_prediction, 1)

  def test_calculate_weights(self):
    es = ensemble_selection.EnsembleSelection(
        problem_statement=ps_pb2.ProblemStatement(tasks=[
            ps_pb2.Task(
                type=ps_pb2.Type(
                    one_dimensional_regression=ps_pb2.OneDimensionalRegression(
                        label='label')))
        ]),
        saved_model_paths=self.saved_model_paths,
        predict_fn=_test_predict_fn,
        ensemble_size=4,
        metric=tf.keras.metrics.MeanSquaredError(),
        goal='minimize')
    ensemble_count = {'model_1': 1, 'model_2': 2, 'model_3': 1}
    want_weights = {'model_1': 0.25, 'model_2': 0.5, 'model_3': 0.25}

    es._calculate_weights(ensemble_count)

    self.assertEqual(want_weights, es.weights)

  def test_predict_before_fit(self):
    es = ensemble_selection.EnsembleSelection(
        problem_statement=ps_pb2.ProblemStatement(tasks=[
            ps_pb2.Task(
                type=ps_pb2.Type(
                    one_dimensional_regression=ps_pb2.OneDimensionalRegression(
                        label='label')))
        ]),
        saved_model_paths=self.saved_model_paths,
        predict_fn=_test_predict_fn,
        ensemble_size=3,
        metric=tf.keras.metrics.MeanSquaredError(),
        goal='minimize')

    with self.assertRaisesRegex(
        ValueError,
        'Weights cannot be empty. Must call `fit` before `predict`.'):
      _ = es.predict(self.fit_examples)

  def test_evaluate_metrics(self):
    es = ensemble_selection.EnsembleSelection(
        problem_statement=ps_pb2.ProblemStatement(tasks=[
            ps_pb2.Task(
                type=ps_pb2.Type(
                    one_dimensional_regression=ps_pb2.OneDimensionalRegression(
                        label='label')))
        ]),
        saved_model_paths=self.saved_model_paths,
        predict_fn=_test_predict_fn,
        ensemble_size=3,
        metric=tf.keras.metrics.MeanSquaredError(),
        goal='minimize')
    metrics = [
        tf.keras.metrics.MeanSquaredError(),
        tf.keras.metrics.MeanAbsoluteError(),
        tf.keras.metrics.RootMeanSquaredError()
    ]

    es.fit(self.fit_examples, self.fit_label)
    ensemble_metrics = es.evaluate(self.fit_examples, self.fit_label, metrics)

    self.assertLen(ensemble_metrics, len(metrics))

  def test_evaluate(self):
    es_size_2 = ensemble_selection.EnsembleSelection(
        problem_statement=ps_pb2.ProblemStatement(tasks=[
            ps_pb2.Task(
                type=ps_pb2.Type(
                    one_dimensional_regression=ps_pb2.OneDimensionalRegression(
                        label='label')))
        ]),
        saved_model_paths=self.saved_model_paths,
        predict_fn=_test_predict_fn,
        ensemble_size=2,
        metric=tf.keras.metrics.MeanSquaredError(),
        goal='minimize')
    es_size_4 = ensemble_selection.EnsembleSelection(
        problem_statement=ps_pb2.ProblemStatement(tasks=[
            ps_pb2.Task(
                type=ps_pb2.Type(
                    one_dimensional_regression=ps_pb2.OneDimensionalRegression(
                        label='label')))
        ]),
        saved_model_paths=self.saved_model_paths,
        predict_fn=_test_predict_fn,
        ensemble_size=4,
        metric=tf.keras.metrics.MeanSquaredError(),
        goal='minimize')
    metrics = [tf.keras.metrics.MeanSquaredError()]

    es_size_2.fit(self.fit_examples, self.fit_label)
    es_size_4.fit(self.fit_examples, self.fit_label)
    es_2_mse = es_size_2.evaluate(self.fit_examples, self.fit_label, metrics)[0]
    es_4_mse = es_size_4.evaluate(self.fit_examples, self.fit_label, metrics)[0]

    self.assertLessEqual(es_4_mse, es_2_mse)


if __name__ == '__main__':
  absltest.main()
