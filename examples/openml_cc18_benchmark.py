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
r"""Demos a basic NitroML benchmark on the 'OpenML-CC18' suite from OpenML.

To run in open-source:

  python examples/openml_cc18_benchmark.py

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
from nitroml.benchmark.suites import openml_cc18
from examples import config


class OpenMLCC18Benchmark(nitroml.Benchmark):
  r"""Demos a NitroML benchmark on the 'OpenML-CC18' classification tasks."""

  def benchmark(self,
                mock_data: bool = False,
                data_dir: str = None,
                use_keras: bool = True,
                enable_tuning: bool = True):

    for i, task in enumerate(
        openml_cc18.OpenMLCC18(data_dir, mock_data=mock_data)):

      if not mock_data and i not in range(20, 40):
        # Use only 20 of the datasets for now.
        # TODO(nikhilmehta): Create subbenchmarks for all 72 tasks.
        # Kubeflow throws a "Max work worflow size error" when pipeline contains
        # too many components.
        # Track issue: https://github.com/kubeflow/pipelines/issues/4170
        continue

      with self.sub_benchmark(task.name):
        # Register running the Task's data preparation components.
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
                train_steps=10,
                eval_steps=10,
                use_keras=use_keras,
                enable_tuning=enable_tuning))
        # Finally, call evaluate() on the workflow DAG outputs, This will
        # automatically append Evaluators to compute metrics from the given
        # SavedModel and 'eval' TF Examples.
        self.evaluate(task=task, model=trainer.outputs.model)


if __name__ == '__main__':

  run_config = dict(
      pipeline_name=config.PIPELINE_NAME + '_openML',
      data_dir=config.OTHER_DOWNLOAD_DIR,
  )

  if config.USE_KUBEFLOW:
    # We need the string "KubeflowDagRunner" in this file to appease the
    # validator used in `tfx create pipeline`.
    # Validator: https://github.com/tensorflow/tfx/blob/v0.22.0/tfx/tools/cli/handler/base_handler.py#L105
    nitroml.main(
        pipeline_root=config.PIPELINE_ROOT,
        tfx_runner=nitroml.get_default_kubeflow_dag_runner(),
        **run_config)
  else:
    # This example has not been tested with engines other than Kubeflow.
    nitroml.main(**run_config)
