# Lint as: python3
"""Tests for nitroml.components.publisher.component."""

from absl.testing import absltest

from nitroml.components.publisher.component import BenchmarkResultPublisher
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class ComponentTest(absltest.TestCase):

  def testMissingBenchmarkResultConstruction(self):
    publisher = BenchmarkResultPublisher(
        'test',
        channel_utils.as_channel([standard_artifacts.ModelEvaluation()]),
        run=1,
        num_runs=2)

    self.assertEqual('NitroML.BenchmarkResult',
                     publisher.outputs['benchmark_result'].type_name)

  def testInvalidBenchmarkNameThrows(self):
    with self.assertRaises(ValueError):
      BenchmarkResultPublisher(
          '',
          channel_utils.as_channel([standard_artifacts.ModelEvaluation()]),
          run=1,
          num_runs=2)


if __name__ == '__main__':
  absltest.main()
