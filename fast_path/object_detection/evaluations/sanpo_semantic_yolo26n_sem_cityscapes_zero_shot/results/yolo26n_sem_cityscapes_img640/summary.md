# yolo26n_sem_cityscapes_img640

## 0. RUN

| item | value |
| --- | --- |
| model run | yolo26n_sem_cityscapes_img640 |
| evaluation | sanpo_semantic_yolo26n_sem_cityscapes_zero_shot |
| GPU | NVIDIA H100 80GB HBM3 |
| weights | yolo26n-sem.pt |
| task | semantic (one class per pixel) |
| classes scored | 13 |
| model input size (long side) | 640 |
| splits | val (2,297 frames), test (5,107 frames) |

## 1. MAIN SCORES (mIoU = main score)

| split | mIoU | pixel accuracy | mIoU within 10 m | mIoU within 5 m |
| --- | --- | --- | --- | --- |
| val | 0.346 | 0.544 | 0.321 | 0.253 |
| test | 0.353 | 0.514 | 0.310 | 0.267 |

## 2. PER CLASS (sorted by test IoU)

| class | val IoU | test IoU | test precision | test recall | test share of scored pixels |
| --- | --- | --- | --- | --- | --- |
| sky | 0.664 | 0.734 | 0.891 | 0.807 | 3.6% |
| tree + vegetation | 0.700 | 0.727 | 0.884 | 0.803 | 16.9% |
| pedestrian | 0.491 | 0.672 | 0.716 | 0.918 | 3.0% |
| building | 0.627 | 0.559 | 0.693 | 0.744 | 17.2% |
| vehicle | 0.490 | 0.414 | 0.449 | 0.840 | 3.2% |
| terrain | 0.246 | 0.387 | 0.717 | 0.457 | 6.6% |
| wall/fence | 0.412 | 0.340 | 0.721 | 0.391 | 11.8% |
| pole | 0.166 | 0.252 | 0.646 | 0.292 | 1.8% |
| rider | 0.143 | 0.131 | 0.272 | 0.202 | 0.0% |
| road | 0.190 | 0.123 | 0.127 | 0.781 | 5.7% |
| sidewalk | 0.140 | 0.123 | 0.629 | 0.132 | 30.0% |
| traffic sign | 0.169 | 0.094 | 0.129 | 0.257 | 0.1% |
| traffic light | 0.059 | 0.032 | 0.125 | 0.041 | 0.0% |

## 3. NEAR PIXELS, IoU PER CLASS (test; only pixels within N meters are counted)

| class | all pixels | within 10 m | within 5 m |
| --- | --- | --- | --- |
| sky | 0.734 | 0.121 | 0.018 |
| tree + vegetation | 0.727 | 0.710 | 0.654 |
| pedestrian | 0.672 | 0.748 | 0.799 |
| building | 0.559 | 0.509 | 0.453 |
| vehicle | 0.414 | 0.432 | 0.333 |
| terrain | 0.387 | 0.379 | 0.313 |
| wall/fence | 0.340 | 0.338 | 0.316 |
| pole | 0.252 | 0.297 | 0.344 |
| rider | 0.131 | 0.150 | 0.081 |
| road | 0.123 | 0.099 | 0.056 |
| sidewalk | 0.123 | 0.121 | 0.101 |
| traffic sign | 0.094 | 0.117 | 0.002 |
| traffic light | 0.032 | 0.009 | 0.000 |

## 4. WHAT THE MODEL SAID ON EACH CLASS'S PIXELS (test; its 3 most common answers)

| class (ground truth) | model classes counted as correct | what the model said |
| --- | --- | --- |
| sky | sky | sky 81%, vegetation 11%, building 8% |
| tree + vegetation | vegetation | vegetation 80%, building 8%, terrain 3% |
| pedestrian | person | person 92%, building 3%, road 2% |
| building | building | building 74%, train 5%, road 4% |
| vehicle | car, truck, bus, train, motorcycle, bicycle | car 74%, building 8%, truck 8% |
| terrain | terrain | terrain 46%, road 34%, sidewalk 12% |
| wall/fence | wall, fence | fence 31%, building 24%, road 17% |
| pole | pole | pole 29%, building 26%, person 10% |
| rider | rider | person 25%, rider 20%, bicycle 20% |
| road | road | road 78%, sidewalk 15%, car 3% |
| sidewalk | sidewalk | road 84%, sidewalk 13%, fence 1% |
| traffic sign | traffic sign | building 34%, traffic sign 26%, vegetation 13% |
| traffic light | traffic light | building 37%, pole 27%, vegetation 15% |

## 5. SPEED AND GPU (milliseconds per frame, model in a batch)

| split | model | preprocess | postprocess | GPU memory peak (GB) |
| --- | --- | --- | --- | --- |
| val | 0.10 | 94.46 | 0.07 | 3.55 |
| test | 0.10 | 76.88 | 0.06 | 3.55 |
