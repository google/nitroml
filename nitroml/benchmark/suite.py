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
r"""A suite of benchmark tasks."""

import abc
from typing import Iterator

from nitroml.benchmark import task


class BenchmarkSuite(abc.ABC):
  r"""Defines a collection of BenchmarkTasks."""

  @abc.abstractmethod
  def __iter__(self) -> Iterator[task.BenchmarkTask]:
    """Returns the tasks that compose this suite."""
    pass
