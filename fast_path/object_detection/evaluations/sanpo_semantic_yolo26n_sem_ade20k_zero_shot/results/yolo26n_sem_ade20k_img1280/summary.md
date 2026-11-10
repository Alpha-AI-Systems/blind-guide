# yolo26n_sem_ade20k_img1280

## 0. RUN

| item | value |
| --- | --- |
| model run | yolo26n_sem_ade20k_img1280 |
| evaluation | sanpo_semantic_yolo26n_sem_ade20k_zero_shot |
| GPU | NVIDIA H100 80GB HBM3 |
| weights | yolo26n-sem-ade20k.pt |
| task | semantic (one class per pixel) |
| classes scored | 20 |
| model input size (long side) | 1280 |
| splits | val (2,297 frames), test (5,107 frames) |

## 1. MAIN SCORES (mIoU = main score)

| split | mIoU | pixel accuracy | mIoU within 10 m | mIoU within 5 m |
| --- | --- | --- | --- | --- |
| val | 0.211 | 0.473 | 0.188 | 0.165 |
| test | 0.249 | 0.438 | 0.222 | 0.204 |

## 2. PER CLASS (sorted by test IoU)

| class | val IoU | test IoU | test precision | test recall | test share of scored pixels |
| --- | --- | --- | --- | --- | --- |
| person | 0.648 | 0.738 | 0.879 | 0.822 | 2.3% |
| sky | 0.493 | 0.710 | 0.731 | 0.962 | 2.7% |
| vehicle | 0.594 | 0.643 | 0.803 | 0.763 | 2.4% |
| building | 0.532 | 0.526 | 0.707 | 0.674 | 12.8% |
| sidewalk | 0.521 | 0.525 | 0.640 | 0.745 | 22.4% |
| stairs | 0.016 | 0.387 | 0.789 | 0.432 | 0.4% |
| terrain | 0.290 | 0.284 | 0.427 | 0.460 | 4.9% |
| road | 0.290 | 0.272 | 0.297 | 0.761 | 4.3% |
| pole | 0.117 | 0.219 | 0.569 | 0.262 | 1.3% |
| wall/fence | 0.191 | 0.210 | 0.405 | 0.305 | 8.8% |
| tree | 0.187 | 0.161 | 0.172 | 0.714 | 2.4% |
| obstacle | 0.075 | 0.080 | 0.757 | 0.082 | 7.0% |
| vegetation | 0.098 | 0.060 | 0.714 | 0.061 | 10.2% |
| traffic sign | 0.057 | 0.041 | 0.044 | 0.358 | 0.0% |
| hand rail | 0.010 | 0.037 | 0.045 | 0.171 | 0.5% |
| traffic light | 0.089 | 0.037 | 0.058 | 0.091 | 0.0% |
| water body | 0.000 | 0.034 | 0.034 | 0.968 | 0.3% |
| paved trail | 0.007 | 0.014 | 0.908 | 0.014 | 17.1% |
| opening-door | 0.002 | 0.003 | 0.028 | 0.003 | 0.2% |
| animal | 0.000 | 0.000 | 0.000 | 0.000 | 0.0% |

## 3. NEAR PIXELS, IoU PER CLASS (test; only pixels within N meters are counted)

| class | all pixels | within 10 m | within 5 m |
| --- | --- | --- | --- |
| person | 0.738 | 0.777 | 0.824 |
| sky | 0.710 | 0.140 | 0.031 |
| vehicle | 0.643 | 0.701 | 0.635 |
| building | 0.526 | 0.445 | 0.377 |
| sidewalk | 0.525 | 0.536 | 0.560 |
| stairs | 0.387 | 0.405 | 0.442 |
| terrain | 0.284 | 0.262 | 0.195 |
| road | 0.272 | 0.221 | 0.133 |
| pole | 0.219 | 0.249 | 0.266 |
| wall/fence | 0.210 | 0.237 | 0.235 |
| tree | 0.161 | 0.149 | 0.067 |
| obstacle | 0.080 | 0.084 | 0.086 |
| vegetation | 0.060 | 0.133 | 0.173 |
| traffic sign | 0.041 | 0.034 | 0.007 |
| hand rail | 0.037 | 0.038 | 0.033 |
| traffic light | 0.037 | 0.009 | 0.007 |
| water body | 0.034 | 0.000 | 0.000 |
| paved trail | 0.014 | 0.014 | 0.016 |
| opening-door | 0.003 | 0.003 | 0.002 |
| animal | 0.000 | 0.000 | 0.000 |

## 4. WHAT THE MODEL SAID ON EACH CLASS'S PIXELS (test; its 3 most common answers)

| class (ground truth) | model classes counted as correct | what the model said |
| --- | --- | --- |
| person | person | person 82%, water 4%, wall 2% |
| sky | sky | sky 96%, tree 2%, building 1% |
| vehicle | car, bus, truck, van, bicycle, minibike | car 70%, building 10%, road 5% |
| building | building, house, skyscraper, hovel | building 67%, wall 17%, sidewalk 4% |
| sidewalk | sidewalk | sidewalk 75%, road 11%, water 7% |
| stairs | stairs, stairway, step | road 36%, stairway 23%, stairs 21% |
| terrain | grass, earth, sand, land, field | water 33%, grass 26%, earth 19% |
| road | road | road 76%, sidewalk 16%, floor 2% |
| pole | pole, streetlight | building 22%, pole 22%, tree 11% |
| wall/fence | wall, fence | wall 25%, railing 17%, building 14% |
| tree | tree, palm | tree 66%, building 8%, sky 6% |
| obstacle | bench, ashcan, box, barrel, rock | water 26%, sidewalk 11%, wall 9% |
| vegetation | plant, flower | tree 67%, building 7%, plant 6% |
| traffic sign | signboard | signboard 36%, building 20%, pole 11% |
| hand rail | railing, bannister | wall 18%, road 18%, railing 15% |
| traffic light | traffic light | building 30%, pole 17%, tree 10% |
| water body | water, sea, river, lake, swimming pool, waterfall | water 65%, sea 31%, bridge 1% |
| paved trail | path | sidewalk 37%, road 21%, floor 17% |
| opening-door | door, screen door | building 77%, wall 14%, tree 1% |
| animal | animal | person 58%, sidewalk 18%, road 9% |

## 5. SPEED AND GPU (milliseconds per frame, model in a batch)

| split | model | preprocess | postprocess | GPU memory peak (GB) |
| --- | --- | --- | --- | --- |
| val | 9.26 | 226.20 | 3.60 | 0.00 |
| test | 0.91 | 131.68 | 1.18 | 3.42 |
