# Human-made and machine-made labels

Source: SANPO's `frame_segmentation_annotation_type.json` of every walk (chest camera).

## Our copy (every 3rd labeled frame, 5 per second)

| Split | Frames | Human-labeled | Machine-labeled | Not listed |
| --- | --- | --- | --- | --- |
| train | 18,746 | 9,403 (50.2%) | 9,343 (49.8%) | 0 (0.0%) |
| test | 5,107 | 2,561 (50.1%) | 2,546 (49.9%) | 0 (0.0%) |
| total | 23,853 | 11,964 (50.2%) | 11,889 (49.8%) | 0 (0.0%) |

## Native data (all labeled frames, 15 per second)

| Split | Frames | Human-labeled | Machine-labeled |
| --- | --- | --- | --- |
| train | 56,121 | 9,403 (16.8%) | 46,718 (83.2%) |
| test | 15,287 | 2,561 (16.8%) | 12,726 (83.2%) |
| total | 71,408 | 11,964 (16.8%) | 59,444 (83.2%) |

## Alignment check

- Human-labeled frame numbers, by remainder of (frame number / 6): {0: 11964}
- Our frame numbers, by remainder of (frame number / 6): {0: 11966, 3: 11887}
- Walks where our copy has no human-labeled frame: 0
