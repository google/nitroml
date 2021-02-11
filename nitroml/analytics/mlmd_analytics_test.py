"""Tests for google3.third_party.py.nitroml.analytics.mlmd_analytics."""

from absl.testing import absltest
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

RUN_ID_3 = 'test_run_id_3'


class AnalyticsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tm = test_mlmd.TestMLMD(context_type='pipeline')
    pipeline_ctx_id = tm.put_context(PIPELINE_NAME)
    tm.update_context_type(mlmd_analytics._IR_RUN_CONTEXT_NAME)
    run_context = metadata_store_pb2.Context(
        name=RUN_ID, create_time_since_epoch=5)
    run_context_2 = metadata_store_pb2.Context(
        name=RUN_ID_3, create_time_since_epoch=0)
    run_context_id = tm.put_context(context=run_context)
    run_context_id_2 = tm.put_context(context=run_context_2)
    tm.update_context_type(mlmd_analytics._IR_COMPONENT_NAME)
    self.component_context_id = tm.put_context(PIPELINE_NAME + '.' +
                                               COMPONENT_NAME)
    execution_id = tm.put_execution(RUN_ID,
                                    PIPELINE_NAME + '.' + COMPONENT_NAME)
    execution_id_2 = tm.put_execution(RUN_ID_3,
                                      PIPELINE_NAME + '.' + COMPONENT_NAME)

    tm.put_association(pipeline_ctx_id, execution_id)
    tm.put_association(run_context_id, execution_id)
    tm.put_association(self.component_context_id, execution_id)

    tm.put_association(pipeline_ctx_id, execution_id_2)
    tm.put_association(run_context_id_2, execution_id_2)

    self.input_artifact_id = tm.put_artifact({'name': INPUT_ARTIFACT_NAME})
    self.output_artifact_id = tm.put_artifact({'name': OUTPUT_ARTIFACT_NAME})
    tm.put_attribution(self.component_context_id, self.input_artifact_id)
    tm.put_attribution(self.component_context_id, self.output_artifact_id)
    tm.put_event(self.input_artifact_id, execution_id,
                 metadata_store_pb2.Event.INPUT)
    tm.put_event(self.output_artifact_id, execution_id,
                 metadata_store_pb2.Event.OUTPUT)

    # Attributed artifacts with unassociated executions. This is an anomoly
    # found in MLMD stores created by IR-Based orchestrators.
    execution_id_3 = tm.put_execution(RUN_ID_2, COMPONENT_NAME_2)
    input_artifact_id_2 = tm.put_artifact({'name': INPUT_ARTIFACT_NAME_2})
    output_artifact_id_2 = tm.put_artifact({'name': OUTPUT_ARTIFACT_NAME_2})
    tm.put_attribution(self.component_context_id, input_artifact_id_2)
    tm.put_attribution(self.component_context_id, output_artifact_id_2)
    tm.put_event(input_artifact_id_2, execution_id_3,
                 metadata_store_pb2.Event.INPUT)
    tm.put_event(output_artifact_id_2, execution_id_3,
                 metadata_store_pb2.Event.OUTPUT)

    self.analytics = mlmd_analytics.Analytics(store=tm.store)

  def testGetPipelineRuns(self):
    # Test invalid input.
    with self.assertRaises(LookupError):
      self.analytics.get_pipeline_run('100')

    self.assertEqual([RUN_ID, RUN_ID_3], self.analytics.list_run_ids())

    pipeline = self.analytics.get_latest_pipeline_run()
    self.assertEqual(PIPELINE_NAME, pipeline.name)
    self.assertEqual(RUN_ID, pipeline.run_id)

    pipeline = self.analytics.get_latest_pipeline_run(COMPONENT_NAME)
    self.assertEqual(PIPELINE_NAME, pipeline.name)
    self.assertEqual(RUN_ID, pipeline.run_id)

    with self.assertRaises(LookupError):
      _ = self.analytics.get_latest_pipeline_run('nonexistent_component_name')

    run_ids = [p.run_id for p in self.analytics.list_pipeline_runs()]
    self.assertEqual([RUN_ID, RUN_ID_3], run_ids)

  def testGetComponentRun(self):
    # Test with non-existent component id.
    with self.assertRaises(LookupError):
      self.analytics.get_component_run(100)

    component_run = self.analytics.get_component_run(self.component_context_id)
    # Check Component Properties
    self.assertEqual(COMPONENT_NAME, component_run.component_name)

    input_artifact = component_run.get_artifact(INPUT_ARTIFACT_NAME)
    output_artifact = component_run.get_artifact(OUTPUT_ARTIFACT_NAME)

    self.assertEqual(INPUT_ARTIFACT_NAME, input_artifact.name)
    self.assertEqual(OUTPUT_ARTIFACT_NAME, output_artifact.name)

    want = {'component_id': COMPONENT_NAME, 'run_id': RUN_ID}
    got = component_run.exec_properties
    self.assertCountEqual(want, got)

  def testGetArtifact(self):
    # Test with non-existent artifact id.
    with self.assertRaises(LookupError):
      self.analytics.get_artifact(100)

    input_artifact = self.analytics.get_artifact(self.input_artifact_id)
    output_artifact = self.analytics.get_artifact(self.output_artifact_id)
    self.assertEqual(INPUT_ARTIFACT_NAME, input_artifact.name)
    self.assertEqual(OUTPUT_ARTIFACT_NAME, output_artifact.name)

  def testWalkThroughAnalyticsObject(self):
    # Test with non-existent run id.
    with self.assertRaises(LookupError):
      self.analytics.get_pipeline_run('bad_run_id')
    pipeline_run = self.analytics.get_pipeline_run(RUN_ID)
    self.assertEqual(str(pipeline_run),
                     'Pipeline Name: %s, Run Id: %s' % (PIPELINE_NAME, RUN_ID))

    # Check Pipeline Properties
    self.assertEqual(PIPELINE_NAME, pipeline_run.name)
    self.assertEqual(RUN_ID, pipeline_run.run_id)
    self.assertCountEqual([COMPONENT_NAME], pipeline_run.components.keys())
    component_run = pipeline_run.get_component_run(COMPONENT_NAME)
    self.assertEqual(component_run.id,
                     pipeline_run.list_component_runs()[0].id)
    self.assertEqual(str(component_run), 'Component Name: %s' % COMPONENT_NAME)

    # Check Component Properties
    self.assertEqual(COMPONENT_NAME, component_run.component_name)

    input_artifact = component_run.get_artifact(INPUT_ARTIFACT_NAME)
    output_artifact = component_run.get_artifact(OUTPUT_ARTIFACT_NAME)

    self.assertEqual(INPUT_ARTIFACT_NAME, input_artifact.name)
    self.assertEqual(OUTPUT_ARTIFACT_NAME, output_artifact.name)

    self.assertEqual(input_artifact.id,
                     component_run.list_input_artifacts()[0].id)
    self.assertEqual(output_artifact.id,
                     component_run.list_output_artifacts()[0].id)

    want = {'component_id': COMPONENT_NAME, 'run_id': RUN_ID}
    got = component_run.exec_properties
    self.assertCountEqual(want, got)


if __name__ == '__main__':
  absltest.main()
