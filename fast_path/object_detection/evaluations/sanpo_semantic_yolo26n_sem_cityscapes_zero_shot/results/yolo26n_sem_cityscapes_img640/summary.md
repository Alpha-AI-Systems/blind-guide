# yolo26n_sem_cityscapes_img640

## 0. RUN

| item | value |
| --- | --- |
| model run | yolo26n_sem_cityscapes_img640 |
| evaluation | sanpo_semantic_yolo26n_sem_cityscapes_zero_shot |
| GPU | Tesla T4 |
| weights | yolo26n-sem.pt |
| task | semantic (one class per pixel) |
| classes scored | 13 |
| model input size (long side) | 640 |
| splits | val (2,297 frames), test (5,107 frames) |

## 1. MAIN SCORES (mIoU = main score; human-labeled frames only = the SANPO paper's protocol)

| split | mIoU | pixel accuracy | mIoU within 10 m | mIoU within 5 m | mIoU, human-labeled frames only | human-labeled frames |
| --- | --- | --- | --- | --- | --- | --- |
| val | 0.348 | 0.532 | 0.325 | 0.257 | 0.344 | 1,151 |
| test | 0.353 | 0.503 | 0.309 | 0.267 | 0.348 | 2,561 |

## 2. PER CLASS (sorted by test IoU)

| class | val IoU | test IoU | test IoU, human-labeled frames only | test precision | test recall | test share of scored pixels |
| --- | --- | --- | --- | --- | --- | --- |
| sky | 0.672 | 0.737 | 0.732 | 0.896 | 0.807 | 3.7% |
| vegetation | 0.669 | 0.701 | 0.695 | 0.860 | 0.792 | 14.2% |
| pedestrian | 0.510 | 0.678 | 0.661 | 0.722 | 0.918 | 3.1% |
| building | 0.639 | 0.566 | 0.561 | 0.702 | 0.744 | 17.8% |
| vehicle | 0.494 | 0.416 | 0.411 | 0.452 | 0.840 | 3.3% |
| terrain | 0.246 | 0.388 | 0.383 | 0.721 | 0.457 | 6.8% |
| wall/fence | 0.413 | 0.340 | 0.345 | 0.725 | 0.391 | 12.2% |
| pole | 0.176 | 0.258 | 0.244 | 0.688 | 0.292 | 1.8% |
| rider | 0.143 | 0.131 | 0.126 | 0.273 | 0.202 | 0.0% |
| road | 0.190 | 0.123 | 0.123 | 0.127 | 0.781 | 5.9% |
| sidewalk | 0.140 | 0.123 | 0.123 | 0.630 | 0.132 | 31.0% |
| traffic sign | 0.173 | 0.095 | 0.090 | 0.131 | 0.257 | 0.1% |
| traffic light | 0.059 | 0.032 | 0.029 | 0.128 | 0.041 | 0.0% |

## 3. NEAR PIXELS, IoU PER CLASS (test; only pixels within N meters are counted)

| class | all pixels | within 10 m | within 5 m |
| --- | --- | --- | --- |
| sky | 0.737 | 0.122 | 0.018 |
| vegetation | 0.701 | 0.682 | 0.643 |
| pedestrian | 0.678 | 0.752 | 0.801 |
| building | 0.566 | 0.511 | 0.454 |
| vehicle | 0.416 | 0.434 | 0.335 |
| terrain | 0.388 | 0.380 | 0.314 |
| wall/fence | 0.340 | 0.338 | 0.316 |
| pole | 0.258 | 0.301 | 0.346 |
| rider | 0.131 | 0.150 | 0.082 |
| road | 0.123 | 0.099 | 0.056 |
| sidewalk | 0.123 | 0.121 | 0.101 |
| traffic sign | 0.095 | 0.118 | 0.002 |
| traffic light | 0.032 | 0.009 | 0.000 |

## 4. WHAT THE MODEL SAID ON EACH CLASS'S PIXELS (test; its 3 most common answers)

| class (ground truth) | model classes counted as correct | what the model said |
| --- | --- | --- |
| sky | sky | sky 81%, vegetation 11%, building 8% |
| vegetation | vegetation | vegetation 79%, building 8%, terrain 3% |
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
| val | 0.96 | 1.67 | 0.42 | 4.05 |
| test | 0.95 | 1.69 | 0.36 | 4.05 |
