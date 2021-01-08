"""Tests for google3.third_party.py.nitroml.analytics.mlmd_analytics."""

from absl.testing import absltest
from absl.testing import parameterized
from nitroml.analytics import mlmd_analytics
from nitroml.testing import test_mlmd

from ml_metadata.proto import metadata_store_pb2


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


PIPELINE_NAME = 'test_pipeline'
RUN_ID = 'test_run_id'
COMPONENT_NAME = 'test_component_name'
INPUT_ARTIFACT_NAME = 'test_input_artifact'
OUTPUT_ARTIFACT_NAME = 'test_output_artifact'

RUN_ID_2 = 'test_run_id_2'
COMPONENT_NAME_2 = 'test_component_name_2'
INPUT_ARTIFACT_NAME_2 = 'test_input_artifact_2'
OUTPUT_ARTIFACT_NAME_2 = 'test_output_artifact_2'


def get_ir_based_analytics():
  tm = test_mlmd.TestMLMD(context_type='pipeline')
  pipeline_ctx_id = tm.put_context(PIPELINE_NAME)
  tm.update_context_type(mlmd_analytics._IR_RUN_CONTEXT_NAME)
  run_context_id = tm.put_context(RUN_ID)
  tm.update_context_type(mlmd_analytics._IR_COMPONENT_NAME)
  component_context_id = tm.put_context(COMPONENT_NAME)
  execution_id = tm.put_execution(RUN_ID, COMPONENT_NAME)
  tm.put_association(pipeline_ctx_id, execution_id)
  tm.put_association(run_context_id, execution_id)
  tm.put_association(component_context_id, execution_id)
  input_artifact_id = tm.put_artifact({'name': INPUT_ARTIFACT_NAME})
  output_artifact_id = tm.put_artifact({'name': OUTPUT_ARTIFACT_NAME})
  tm.put_attribution(component_context_id, input_artifact_id)
  tm.put_attribution(component_context_id, output_artifact_id)
  tm.put_event(input_artifact_id, execution_id, metadata_store_pb2.Event.INPUT)
  tm.put_event(output_artifact_id, execution_id,
               metadata_store_pb2.Event.OUTPUT)

  # Attributed artifacts with unassociated executions. This is an anomoly found
  # in MLMD stores created by IR-Based orchestrators.
  execution_id_2 = tm.put_execution(RUN_ID_2, COMPONENT_NAME_2)
  input_artifact_id_2 = tm.put_artifact({'name': INPUT_ARTIFACT_NAME_2})
  output_artifact_id_2 = tm.put_artifact({'name': OUTPUT_ARTIFACT_NAME_2})
  tm.put_attribution(component_context_id, input_artifact_id_2)
  tm.put_attribution(component_context_id, output_artifact_id_2)
  tm.put_event(input_artifact_id_2, execution_id_2,
               metadata_store_pb2.Event.INPUT)
  tm.put_event(output_artifact_id_2, execution_id_2,
               metadata_store_pb2.Event.OUTPUT)

  return mlmd_analytics.Analytics(store=tm.store)


def get_nonir_based_analytics():
  tm = test_mlmd.TestMLMD(context_type=mlmd_analytics._NONIR_RUN_CONTEXT_NAME)
  run_context_id = tm.put_context(
      RUN_ID, properties={'pipeline_name': PIPELINE_NAME})
  tm.update_context_type(mlmd_analytics._NONIR_COMPONENT_NAME)
  component_context_id = tm.put_context(COMPONENT_NAME)
  execution_id = tm.put_execution(RUN_ID, COMPONENT_NAME)
  tm.put_association(run_context_id, execution_id)
  tm.put_association(component_context_id, execution_id)
  input_artifact_id = tm.put_artifact({'name': INPUT_ARTIFACT_NAME})
  output_artifact_id = tm.put_artifact({'name': OUTPUT_ARTIFACT_NAME})
  tm.put_attribution(component_context_id, input_artifact_id)
  tm.put_attribution(component_context_id, output_artifact_id)
  tm.put_event(input_artifact_id, execution_id, metadata_store_pb2.Event.INPUT)
  tm.put_event(output_artifact_id, execution_id,
               metadata_store_pb2.Event.OUTPUT)
  return mlmd_analytics.Analytics(store=tm.store)


class AnalyticsTest(parameterized.TestCase):

  def _get_analytics(self, ir_analytics_flag: bool) -> mlmd_analytics.Analytics:
    if ir_analytics_flag:
      return get_ir_based_analytics()
    else:
      return get_nonir_based_analytics()

  @parameterized.named_parameters(('IrAnalytics', True),
                                  ('NonIrAnalytics', False))
  def testIRBasedFlag(self, ir_analytics_flag):
    analytics = self._get_analytics(ir_analytics_flag)
    self.assertEqual(ir_analytics_flag, analytics._ir_based_dag_runner)

  @parameterized.named_parameters(('IrAnalytics', True),
                                  ('NonIrAnalytics', False))
  def testListRuns(self, ir_analytics_flag):
    analytics = self._get_analytics(ir_analytics_flag)
    want = {'pipeline_name': PIPELINE_NAME, 'run_id': RUN_ID}
    got = analytics.list_runs()[RUN_ID]
    self.assertContainsSubset(want, got)

  @parameterized.named_parameters(('IrAnalytics', True),
                                  ('NonIrAnalytics', False))
  def testWalkThroughAnalyticsObject(self, ir_analytics_flag):
    analytics = self._get_analytics(ir_analytics_flag)

    # Get Run
    with self.assertRaises(ValueError):
      analytics.get_run('bad_run_id')
    pipeline_run = analytics.get_run(RUN_ID)

    # Check Pipeline Properties
    self.assertEqual(PIPELINE_NAME, pipeline_run.name)
    self.assertEqual(RUN_ID, pipeline_run.run_id)
    self.assertCountEqual([COMPONENT_NAME], pipeline_run.components.keys())
    component_run = pipeline_run.components[COMPONENT_NAME]

    # Check Component Properties
    self.assertEqual(RUN_ID, component_run.run_id)
    self.assertEqual(COMPONENT_NAME, component_run.component_name)
    self.assertCountEqual({INPUT_ARTIFACT_NAME}, component_run.inputs.keys())
    self.assertCountEqual({OUTPUT_ARTIFACT_NAME}, component_run.outputs.keys())
    want = {'component_id': COMPONENT_NAME, 'run_id': RUN_ID}
    got = component_run.exec_properties
    self.assertCountEqual(want, got)


if __name__ == '__main__':
  absltest.main()
