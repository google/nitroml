# MLMD testdata directory

This directory contains MLMD SQLite snapshot instances for testing.

To create an entry in directory, you can run

```shell
alias rube='/google/bin/releases/autolx/rube/rube'
rube run //third_party/py/nitroml/examples/google:openml_benchmark_plx -- --flume_exec_mode=UNOPT --match='.*(vowel|climate).*'
```

Finally you can copy the MLMD instance from `/tmp/nitroml_examples/mlmd.sqlite`
to this directory, and date the filename.
