---
pretty_name: HMDB51 Action Recognition Dataset
task_categories:
  - video-classification
language:
  - en
tags:
  - video
  - action-recognition
  - hmdb51
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: metadata_train.csv
      - split: test
        path: metadata_test.csv
      - split: other
        path: metadata_other.csv
---

# HMDB51 Action Recognition Dataset

HMDB51 is a large human motion recognition video database containing
**6,766 clips** across **51 action classes**.

## Dataset Structure

```
hf_dataset/
├── .gitattributes          <- Git LFS rules for video files
├── README.md               <- This file
├── metadata_train.csv      <- 3,570 clips
├── metadata_test.csv       <- 1,530 clips
├── metadata_other.csv      <- 1,666 clips
└── videos/
    ├── brush_hair/
    │   ├── clip1.avi
    │   └── ...
    ├── cartwheel/
    └── ...  (51 class folders total)
```

Each metadata CSV has two columns:

| Column      | Description                            |
|-------------|----------------------------------------|
| `file_name` | Relative path to the video file        |
| `label`     | Action class label string              |

## Splits

| Split  |   Clips |
|--------|---------|
| train  |   3,570 |
| test   |   1,530 |
| other  |   1,666 |
| **Total** | **6,766** |

## Classes (51 total)

```
brush_hair, cartwheel, catch, chew, clap, climb, climb_stairs, dive, draw_sword, dribble, drink, eat, fall_floor, fencing, flic_flac, golf, handstand, hit, hug, jump, kick, kick_ball, kiss, laugh, pick, pour, pullup, punch, push, pushup, ride_bike, ride_horse, run, shake_hands, shoot_ball, shoot_bow, shoot_gun, sit, situp, smile, smoke, somersault, stand, swing_baseball, sword, sword_exercise, talk, throw, turn, walk, wave
```

## Loading with Hugging Face datasets

```python
from datasets import load_dataset

# Load a specific split
ds_train = load_dataset("YOUR_USERNAME/hmdb51", split="train")

# Access a sample
sample = ds_train[0]
print(sample["label"])       # e.g. "brush_hair"
print(sample["file_name"])   # e.g. "videos/brush_hair/foo.avi"

# Stream without downloading everything
ds_stream = load_dataset("YOUR_USERNAME/hmdb51", split="train", streaming=True)
for sample in ds_stream.take(5):
    print(sample["label"])
```
