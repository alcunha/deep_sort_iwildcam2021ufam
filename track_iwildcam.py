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

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'test_info_json', default=None,
    help=('Path to json file containing the test information json for'
          ' the iWildCam2021 competition'))

flags.DEFINE_string(
    'features_json', default=None,
    help=('Path to json file containing the features for each bounding box'))

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

def track_iwildcam(test_set, features):
  for seq_id in test_set.seq_id.unique()[:2]:
    print("seq: ", seq_id)
    images_ids = list(test_set[test_set.seq_id == seq_id].id)
    features_seq = features[features.img_id.isin(images_ids)]
    print('num images: ', len(features_seq))
    for img_id in features_seq.img_id.unique():
      print('image: ', img_id)

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
