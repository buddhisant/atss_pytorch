# atss_pytorch
实现了Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection论文，基于pytorch，并且在实现过程中参考了[maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)和[mmdetection](https://github.com/open-mmlab/mmdetection)。
支持多卡分布式训练

**更多目标检测代码请参见[友好型的object detection代码实现和论文解读](https://blog.csdn.net/gongyi_yf/article/details/109660890)**
**backbone网络基于resnet50**

**请确保已经安装pycocotools以及1.1.0版本以上的pytorch**
