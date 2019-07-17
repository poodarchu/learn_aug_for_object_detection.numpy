# learn_aug_for_object_detection.numpy
Learning Data Augmentation Strategies for Object Detection in numpy

## Acknowledge
mainly borrow from https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/autoaugment_utils.py
convert all tf operators to numpy and PIL

## Usage
```python
image, bboxes = distort_image_with_autoaugment(image, bboxes, policy_name)
```
