# DeepSORT for iWildcam2021 (UFAM Team)

This repository contains code to run DeepSORT on [iWilcam 2021](https://www.kaggle.com/c/iwildcam2021-fgvc8) camera trap image sequences. 

Please refer to the original [DeepSORT repository](https://github.com/nwojke/deep_sort) for more information about Deep SORT or see the [arXiv preprint](https://arxiv.org/abs/1703.07402).

### DeepSORT to track animals

To extract features using EfficientNet-B2, use the script `mot/generate_features.py` from our [main iWildcam repository](https://github.com/alcunha/iwildcam2021ufam). We kept DeepSORT code on a separate repository to avoid GPLv3 licensing conflicts.

To track animals with DeepSORT use the script `track_iwildcam.py`:
```bash
python track_iwildcam.py --test_info_json=PATH_TO_BE_CONFIGURED/iwildcam2021_test_information.json
    --features_json=PATH_TO_BE_CONFIGURED/efficientnet_b2_crop_25mai_features.json
    --tracks_file=PATH_TO_BE_CONFIGURED/efficientnet_b2_crop_25mai_tracks.json
```

Finally, to classify tracks and generate a submission, go back to from our [main iWildcam repository](https://github.com/alcunha/iwildcam2021ufam) and use the script `classification/predict_track.py`.


### Citing DeepSORT

If you find this repo useful in your research, please consider citing the following papers:

    @inproceedings{Wojke2017simple,
      title={Simple Online and Realtime Tracking with a Deep Association Metric},
      author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
      booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
      year={2017},
      pages={3645--3649},
      organization={IEEE},
      doi={10.1109/ICIP.2017.8296962}
    }

    @inproceedings{Wojke2018deep,
      title={Deep Cosine Metric Learning for Person Re-identification},
      author={Wojke, Nicolai and Bewley, Alex},
      booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
      year={2018},
      pages={748--756},
      organization={IEEE},
      doi={10.1109/WACV.2018.00087}
    }

### License

[GPL-3.0 License](LICENSE)