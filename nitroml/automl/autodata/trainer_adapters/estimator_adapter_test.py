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
"""Tests for nitroml.automl.autodata.trainer_adapters.estimator_adapter."""

import os

from nitroml.automl.autodata import subpipeline
from nitroml.automl.autodata.preprocessors import basic_preprocessor
from nitroml.automl.autodata.trainer_adapters import estimator_adapter
from nitroml.benchmark.tasks import tfds_task
from nitroml.testing import e2etest
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_model_analysis as tfma
from tfx.utils import path_utils


class EstimatorAdapterTest(e2etest.TestCase):

  # pylint: disable=g-long-lambda
  @e2etest.parameterized.named_parameters(
      {
          'testcase_name':
              'linear',
          'estimator_constructor':
              lambda adapter, config: tf.estimator.LinearEstimator(
                  head=adapter.head,
                  feature_columns=adapter.get_sparse_feature_columns(),
                  config=config)
      }, {
          'testcase_name':
              'dnn',
          'estimator_constructor':
              lambda adapter, config: tf.estimator.DNNEstimator(
                  head=adapter.head,
                  feature_columns=adapter.get_dense_feature_columns(),
                  hidden_units=[3],
                  config=config)
      })
  # pylint: enable=g-long-lambda
  def test_estimator_lifecycle(self, estimator_constructor):
    """Checks that a full estimator lifecycle completes without crashing."""

    # Generate data that the adapter can consume.
    task = tfds_task.TFDSTask(tfds.builder('titanic'))

    autodata = subpipeline.AutoData(
        task.problem_statement,
        examples=task.train_and_eval_examples,
        preprocessor=basic_preprocessor.BasicPreprocessor())

    self.run_pipeline(components=task.components + autodata.components)

    # Create the trainer adapter.
    adapter = estimator_adapter.EstimatorAdapter(
        problem_statement=task.problem_statement,
        transform_graph_dir=self.artifact_dir(
            'Transform.AutoData/transform_graph'))

    config = tf.estimator.RunConfig(
        save_checkpoints_steps=999, keep_checkpoint_max=3)

    # Create the estimator.
    estimator = estimator_constructor(adapter, config)

    # Train.
    estimator.train(
        input_fn=adapter.get_input_fn(
            file_pattern=self.artifact_dir(
                'Transform.AutoData/transformed_examples', 'train/*'),
            batch_size=3),
        max_steps=3)

    # Eval.
    results = estimator.evaluate(
        input_fn=adapter.get_input_fn(
            file_pattern=self.artifact_dir(
                'Transform.AutoData/transformed_examples', 'eval/*'),
            batch_size=3),
        steps=1)
    self.assertNotEmpty(results)

    # Export for TFMA.
    tfma.export.export_eval_savedmodel(
        estimator=estimator,
        export_dir_base=path_utils.eval_model_dir(estimator.model_dir),
        eval_input_receiver_fn=adapter.get_eval_input_receiver_fn())

    # Export for Serving.
    estimator.export_saved_model(
        export_dir_base=os.path.join(estimator.model_dir, 'export'),
        serving_input_receiver_fn=adapter.get_serving_input_receiver_fn())


if __name__ == '__main__':
  e2etest.main()
