# DeepSORT for iWildcam2021 (derived from original DeepSORT Code)
# Copyright (C) 2021 Fagner Cunha

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

r"""Tool to track animals on iWildCam sequences using deepsort.

Set the environment variable PYTHONHASHSEED to a reproducible value
before you start the python process to ensure that the model trains
or infers with reproducibility
"""
import json
import random

from absl import app
from absl import flags
import numpy as np
import pandas as pd

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'test_info_json', default=None,
    help=('Path to json file containing the test information json for'
          ' the iWildCam2021 competition'))

flags.DEFINE_string(
    'features_json', default=None,
    help=('Path to json file containing the features for each bounding box'))

flags.DEFINE_float(
    'max_cosine_distance', default=0.2,
    help=('Gating threshold for cosine distance metric (object appearance).'))

flags.DEFINE_integer(
    'nn_budget', default=None,
    help=('Maximum size of the appearance descriptor gallery. If None, no'
          ' budget is enforced.'))

if 'random_seed' not in list(FLAGS):
  flags.DEFINE_integer(
      'random_seed', default=42,
      help=('Random seed for reproductible experiments'))

flags.mark_flag_as_required('test_info_json')
flags.mark_flag_as_required('features_json')

def _load_features():
  with open(FLAGS.features_json, 'r') as json_file:
    json_data = json.load(json_file)
  features = pd.DataFrame(json_data)

  return features

def _load_seq_info():
  with open(FLAGS.test_info_json, 'r') as json_file:
    json_data = json.load(json_file)
  test_set = pd.DataFrame(json_data['images'])

  return test_set

def create_detections(features, img_id):
  features = features[features.img_id == img_id]

  detection_list = []
  for _, row in features.iterrows():
    detection_list.append(Detection(
      row['bbox_tlwh'],
      row['conf'],
      row['features']))

  return detection_list

def run_deepsort_on_seq(detections_list):
  metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", FLAGS.max_cosine_distance, FLAGS.nn_budget)
  tracker = Tracker(metric)
  confirmed_tracks = []
  all_tracks = []

  for frame_idx, detections in enumerate(detections_list):
    tracker.predict()
    tracker.update(detections)

    for track in tracker.tracks:
      if track.is_confirmed():
        confirmed_tracks.append(track.track_id)

      if track.time_since_update > 1:
          continue
      bbox = track.to_tlwh()
      all_tracks.append([
          frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

  confirmed_tracks = set(confirmed_tracks)
  results = [track_bbox for track_bbox in all_tracks
             if track_bbox[1] in confirmed_tracks]

  return results

def track_iwildcam(test_set, features):
  for seq_id in test_set.seq_id.unique()[30:40]:
    seq_info = test_set[test_set.seq_id == seq_id]
    seq_info = seq_info.sort_values(by=['seq_frame_num'])

    detections_list = []
    max_dets = 0
    for _, row in seq_info.iterrows():
      detections = create_detections(features, row['id'])
      detections_list.append(detections)
      max_dets = max(max_dets, len(detections))

    results = run_deepsort_on_seq(detections_list)

def set_random_seeds():
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)

def main(_):
  set_random_seeds()

  test_set = _load_seq_info()
  features = _load_features()
  track_iwildcam(test_set, features)

if __name__ == '__main__':
  app.run(main)
