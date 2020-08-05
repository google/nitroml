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
"""NitroML Tuner utils."""

import matplotlib.pyplot as plt
import numpy as np


def display_tuner_data(data):

  warmup_tuner_score = data['warmup_trial_data']
  random_tuner_score = data['tuner_trial_data']
  best_score = data['best_cumulative_score']

  num_warmup_trials = len(warmup_tuner_score)
  num_random_trials = len(random_tuner_score)
  total_trials = num_warmup_trials + num_random_trials

  _, axs = plt.subplots(1, 3, figsize=(15, 5))

  ymax = np.max(best_score) + 0.05
  ymin = np.min(best_score) - 0.05

  axs[0].plot(
      np.arange(1, total_trials + 1),
      best_score,
      label=f'best_score ({total_trials} trials)',
      color='orange',
      linewidth=2,
      marker='o')
  axs[0].set_ylim(ymin, ymax)
  axs[0].set_title(f'Best Cumulative Score ({total_trials} trials)')
  axs[0].set_ylabel(data['objective'])
  axs[0].set_xlabel('Trial')

  axs[1].plot(
      np.arange(1, num_warmup_trials + 1),
      warmup_tuner_score,
      label=f'stage_warmup ({num_warmup_trials} trials)',
      color='blue',
      linewidth=2,
      marker='o')
  axs[1].set_ylim(ymin, ymax)
  axs[1].set_title(f'Warmup Tuning ({num_warmup_trials} trials)')
  axs[1].set_xlabel('Trial')

  axs[2].plot(
      np.arange(1, num_random_trials + 1),
      random_tuner_score,
      label=f'stage_final ({num_random_trials} trials)',
      color='blue',
      linewidth=2,
      marker='o')
  axs[2].set_ylim(ymin, ymax)
  axs[2].set_title(f'Random Tuning ({num_random_trials} trials)')
  axs[2].set_xlabel('Trial')

  # plt.savefig('result.png', bbox_inches='tight', pad_inches=0)
  plt.show()
