# MLMD testdata directory

This directory contains MLMD SQLite snapshot instances for testing.

To create an entry in directory, you can run

```shell
$ blaze-py3/bin/third_party/py/nitroml/examples/openml_benchmark  --alsologtostderr --config="$(envsubst < third_party/py/nitroml/examples/config_local.json)" --command=launch --match='.*(vowel|climate).*'
```

Finally you can copy the MLMD instance from `/tmp/nitroml/example/mlmd.sqlite`
to this directory, and date the filename.
