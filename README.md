Based on [jwyang/fpn.pytorch](https://github.com/jwyang/fpn.pytorch), i change little code to get a more reasonable mAP when training pascal voc 2007 and 07+12.
Pytorch implementation of Feature Pyramid Network (FPN) for Object Detection.



## Introduction

This project inherits the property of our [jwyang/fpn.pytorch](https://github.com/jwyang/fpn.pytorch).Hence, you can see more information about it.The following things are what I did :

* **The stride of Resnet layer4 change 2 from 1 **. The most fundamental reason why mAP is low is that the anchor's position and number of each layer are calculated by stride in this code.The designed FPN_FEAT_STRIDES in config is [4, 8, 16, 32, 64].  When layer4's stride is set to 1, FPN_FEAT_STRIDES should be changed to [4, 8, 16, 16, 32], but FPN_FEAT_STRIDES is still the default value, which results in p5, p6 has about 3/4 of the anchors generated outside the image.
* **Changing $log~e$ to $log~2$ in  _PyramidRoI_Feat**.In original paper, roi pool on pyramid feature maps use $log~2$. It does not seem to affect the training results.

## Benchmarking

I benchmark this code thoroughly on pascal voc2007 and 07+12. Below are the results:

1). PASCAL VOC 2007 (Train/Test: 07trainval/07test, scale=600, ROI Align)

model    | GPUs | Batch Size | lr        | lr_decay | max_epoch     |  Speed/epoch | Memory/GPU | mAP 
---------|-----------|----|-----------|-----|-----|-------|--------|--------
Res-101    | 1  GTX 1080 (Ti) | 2 | 1e-3 | 10  | 12  |  0.22 hr | 6137MB | 75.7 

2). PASCAL VOC 07+12 (Train/Test: 07+12trainval/07test, scale=600, ROI Align)



| model   | GPUs             | Batch Size | lr   | lr_decay | max_epoch | Speed/epoch | Memory/GPU | mAP  |
| ------- | ---------------- | ---------- | ---- | -------- | --------- | ----------- | ---------- | ---- |
| Res-101 | 1  GTX 1080 (Ti) | 1          | 1e-3 | 10       | 12        | \           | 9011MB     | 80.5 |

##**Usage**

train voc2007:

```
CUDA_VISIBLE_DEVICES=3 python3 trainval_net.py exp_name --dataset pascal_voc --net res101 --bs 2 --num_workers 4 --lr 1e-3 --epochs 12 --save_dir weights --cuda --use_tfboard True
```

test voc2007:

```
CUDA_VISIBLE_DEVICES=3 python3 test_net.py exp_name --dataset pascal_voc --net res101 --checksession 1 --checkepoch 7 --checkpoint 5010 --cuda --load_dir weights
```

train voc07+12:

```
CUDA_VISIBLE_DEVICES=3 python3 trainval_net.py exp_name2 --dataset pascal_voc_0712 --net res101 --bs 2 --num_workers 4 --lr 1e-3 --epochs 12 --save_dir weights --cuda --use_tfboard True
```

