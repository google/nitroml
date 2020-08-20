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

import itertools
import string
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def aggregate_tuner_data(
    keys: List[str], data_list: List[Dict[str, Any]]) -> Dict[str, List[int]]:
  """Returns the mean and its std_error for a list of dicts with same keys.

  Args:
    keys: List of string keys to aggregate which are common to all dicts.
    data_list: List of dicts to aggregate.
  Returns:
    Aggregate data dict.
  """

  aggregate_data = {}
  for key in keys:

    # Find max number of trials in different runs.
    max_trials = max([len(data[key]) for data in data_list])
    # Different tuners may have different number of trials.
    # For plotting, we extend the list to max_trials by appending the best
    # score achieved by the tuner given by data[key][-1].
    all_data = np.vstack([
        (data[key] + (max_trials - len(data[key])) * [data[key][-1]])
        for data in data_list
    ])
    aggregate_data[f'{key}_mean'] = np.mean(all_data, axis=0)
    aggregate_data[f'{key}_stdev'] = np.std(
        all_data, axis=0) / np.sqrt(all_data.shape[0])

  return aggregate_data


def display_tuner_data_with_error_bars(data_list: List[Dict[str, Any]],
                                       save_plot: bool = False):
  """Plots the tuner data with error bars.

  Args:
    data_list: List of dicts representing tuner data.
    save_plot: If True, saves the plot in local dir.
  """

  keys = ['warmup_trial_data', 'tuner_trial_data', 'best_cumulative_score']
  data = aggregate_tuner_data(keys, data_list)
  data['objective'] = data_list[0]['objective']

  _, axs = plt.subplots(1, len(keys), figsize=(6 * len(keys), 5))
  cycol = itertools.cycle('bgrcmk')

  ymax = 0
  ymin = 1.0

  for ix, key in enumerate(keys):
    tuner_score_mean = data[f'{key}_mean']
    tuner_score_stdev = data[f'{key}_stdev']
    num_trials = len(tuner_score_mean)
    axs[ix].errorbar(
        np.arange(1, num_trials + 1),
        tuner_score_mean,
        yerr=tuner_score_stdev,
        label=f'key ({num_trials} trials)',
        color=next(cycol),
        linewidth=2,
        marker='o')

    alc = metrics.auc(np.arange(1, num_trials + 1), tuner_score_mean)
    title = string.capwords(key.replace(' ', '_'))
    axs[ix].set_title(f'{title} (ALC = {alc:.2f} for {num_trials} trials)')

    ymax = max(np.max(tuner_score_mean), ymax)
    ymin = min(np.min(tuner_score_mean), ymin)

  ymax += 0.05
  ymin -= 0.1

  axs[0].set_ylabel(data['objective'])
  for ax in axs:
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Trial')
  if save_plot:
    plt.savefig('result.png', bbox_inches='tight')
  plt.show()


def display_tuner_data(data, save_plot=True):
  """Plots tuner data."""

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

  if save_plot:
    plt.savefig('display_tuner_data.png', bbox_inches='tight')

  plt.show()
