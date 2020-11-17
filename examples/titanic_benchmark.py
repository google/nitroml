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
# pyformat: disable
r"""Demos a basic NitroML benchmark on the 'Titanic' dataset from OpenML.

To run in open-source:

  python examples/titanic_benchmark.py

"""  # pylint: disable=line-too-long
# pyformat: enable
# pylint: disable=g-import-not-at-top
import os
import sys

# Required since Python binaries ignore relative paths when importing:
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import nitroml
from nitroml.automl import autodata as ad
from nitroml.automl.autotrainer import subpipeline as at
from nitroml.benchmark.tasks import tfds_task
from examples import config
import tensorflow_datasets as tfds


class TitanicBenchmark(nitroml.Benchmark):
  r"""Demos a NitroML benchmark on the 'Titanic' dataset from OpenML."""

  def benchmark(self,
                data_dir: str = None,
                use_keras: bool = True,
                enable_tuning: bool = True):
    # Use TFDSTask to define the task for the titanic dataset.
    task = tfds_task.TFDSTask(tfds.builder('titanic', data_dir=data_dir))
    self.add(task.components)

    autodata = self.add(
        ad.AutoData(
            task.problem_statement,
            examples=task.train_and_eval_examples,
            preprocessor=ad.BasicPreprocessor()))

    # Define a Trainer to train our model on the given task.
    trainer = self.add(
        at.AutoTrainer(
            problem_statement=task.problem_statement,
            transformed_examples=autodata.outputs.transformed_examples,
            transform_graph=autodata.outputs.transform_graph,
            schema=autodata.outputs.schema,
            train_steps=1000,
            eval_steps=500,
            enable_tuning=enable_tuning))
    # Finally, call evaluate() on the workflow DAG outputs. This will
    # automatically append Evaluators to compute metrics from the given
    # SavedModel and 'eval' TF Examples.
    self.evaluate(task=task, model=trainer.outputs.model)


if __name__ == '__main__':
  if config.USE_KUBEFLOW:
    # We need the string "KubeflowDagRunner" in this file to appease the
    # validator used in `tfx create pipeline`.
    # Validator: https://github.com/tensorflow/tfx/blob/v0.22.0/tfx/tools/cli/handler/base_handler.py#L105
    nitroml.main(
        pipeline_name=config.PIPELINE_NAME + '_titanic',
        pipeline_root=config.PIPELINE_ROOT,
        data_dir=config.TF_DOWNLOAD_DIR,
        tfx_runner=nitroml.get_default_kubeflow_dag_runner())
  else:
    nitroml.main()
