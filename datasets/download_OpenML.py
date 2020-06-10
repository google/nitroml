import os

from dataset import OpenMLDataset
from absl import flags
from absl import app

flags.DEFINE_string('root_dir', os.path.join(os.environ['HOME'], 'output'),
                    'Path to output csv files.')
flags.DEFINE_integer('max_threads', 1,
                     'The number of threads to use in parallel.')
flags.DEFINE_boolean('download', False, 'Download OpenML data')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # not being used for now!

  dir_path = FLAGS.root_dir
  da = OpenMLDataset(dir_path)

  # Downloads OpenML data
  if FLAGS.download:
    da._get_data()

  assert len(da.tasks) == len(da.components)

  # check the first task
  task = da.tasks[0]
  print(task.toJSON())


if __name__ == "__main__":
  app.run(main)