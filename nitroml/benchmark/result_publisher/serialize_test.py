"""Tests for google3.third_party.py.nitroml.benchmark.result_publisher.serialize."""

from absl.testing import absltest
from nitroml.benchmark.result_publisher import serialize


class SerializeTest(absltest.TestCase):

  def testEncodeFunctionTypeCheck(self):
    properties1 = {'list': [], 'int': 1, 'str': 'string'}
    properties2 = {'int': 1, 'dict': dict(), 'float': 0.1}
    properties3 = {'int': 1, 'float': 0.1, 'str': 'string', 'bool': True}
    with self.assertRaises(TypeError):
      serialize.encode(properties1)

    with self.assertRaises(TypeError):
      serialize.encode(properties2)

    want = '{"bool": true, "float": 0.1, "int": 1, "str": "string"}'
    got = serialize.encode(properties3)
    self.assertEqual(want, got)

  def testDecode(self):
    want = {'int': 1, 'float': 0.1, 'str': 'string', 'bool': True}
    encoded_properties = serialize.encode(want)
    got = serialize.decode(encoded_properties)
    self.assertCountEqual(want, got)


if __name__ == '__main__':
  absltest.main()
