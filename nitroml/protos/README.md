# NitroML protos

This directory contains proto stubs for protos from [tensorflow_metadata](https://github.com/tensorflow/metadata). Unfortunately, the stubs for ProblemStatement are not currently included in the current version `v0.22.2`, so we hardcode them here (as opposed to adding a dependency of Bazel).

TODO(weill): Remove this directory once either `tfx` or `tensorflow_metadata` include the generated stubs in their pip packages.
