"""Tests for google3.third_party.py.nitroml.analytics.mlmd_analytics."""

from absl import logging
from absl.testing import absltest
from nitroml.analytics import mlmd_analytics
from nitroml.testing import test_mlmd

logging.set_stderrthreshold('error')


class PropertyDictTest(absltest.TestCase):

  def __init__(self, *args, **kwargs):
    super(PropertyDictTest, self).__init__(*args, **kwargs)
    self.test_dict = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    self.test_property_dict = mlmd_analytics.PropertyDictWrapper(self.test_dict)

  def testGetAll(self):
    want = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    got = self.test_property_dict.get_all()
    self.assertEqual(want, got)

  def testKeys(self):
    want = {'key1', 'key2', 'key3'}
    got = self.test_property_dict.keys()
    self.assertSameElements(want, got)

  def testValues(self):
    want = {'value1', 'value2', 'value3'}
    got = self.test_property_dict.values()
    self.assertSameElements(want, got)

  def testItems(self):
    want = [('key1', 'value1'), ('key2', 'value2'), ('key3', 'value3')]
    got = self.test_property_dict.items()
    self.assertSameElements(want, got)

  def testRepExposure(self):
    del self.test_dict['key1']
    want = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    got = self.test_property_dict.get_all()
    self.assertEqual(want, got)


class AnalyticsTest(absltest.TestCase):

  def __init__(self, *args, **kwargs):
    super(AnalyticsTest, self).__init__(*args, **kwargs)
    self.test_mlmd = test_mlmd.TestMLMD()
    self.properties1 = {'pipeline_name': 'taxi1', 'run_id': '1'}
    self.properties2 = {'pipeline_name': 'taxi2', 'run_id': '2'}
    self.test_mlmd.put_context('context1', properties=self.properties1)
    self.test_mlmd.put_context('context2', properties=self.properties2)
    self.run_db = mlmd_analytics.Analytics(store=self.test_mlmd.store)

  def testListRuns(self):
    want = {'1': self.properties1, '2': self.properties2}
    got = self.run_db.list_runs()
    self.assertCountEqual(want, got)

  def testGetRun(self):
    with self.assertRaises(ValueError):
      self.run_db.get_run('3')
    pipeline_run = self.run_db.get_run('2')
    self.assertEqual('2', pipeline_run.run_id)
    self.assertEqual('taxi2', pipeline_run.name)

  def testPipelineComponent(self):
    execution_id = self.test_mlmd.put_execution('1', 'stats_gen')
    pipeline_run = self.run_db.get_run('1')
    self.test_mlmd.put_association(pipeline_run._context_id, execution_id)
    self.assertIn('stats_gen', pipeline_run.components)
    self.assertIsInstance(pipeline_run.components['stats_gen'],
                          mlmd_analytics.ComponentRun)

  def testComponentRun(self):
    properties3 = {'pipeline_name': 'taxi3', 'run_id': '1', 'component_id': '3'}
    context_id = self.test_mlmd.put_context('context3', properties=properties3)
    execution_id = self.test_mlmd.put_execution('3', 'stats_gen')
    artifact_id_output = self.test_mlmd.put_artifact({
        'property': 'value',
        'name': 'test_artifact_output',
        'producer_component': 'stats_gen'
    })
    artifact_id_input = self.test_mlmd.put_artifact({
        'property': 'value',
        'name': 'test_artifact_input',
        'producer_component': 'example_gen'
    })
    pipeline_run = self.run_db.get_run('1')
    self.test_mlmd.put_association(context_id, execution_id)
    self.test_mlmd.put_attribution(context_id, artifact_id_output)
    self.test_mlmd.put_attribution(context_id, artifact_id_input)
    component_run = pipeline_run.components['stats_gen']
    self.assertIn('run_id', component_run.exec_properties)
    self.assertIn('component_id', component_run.exec_properties)
    self.assertIn('test_artifact_output', component_run.outputs)
    self.assertIn('test_artifact_input', component_run.inputs)


if __name__ == '__main__':
  absltest.main()
