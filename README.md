# atss_pytorch
基于pytorch实现了ATSS模型

实现了Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection论文，并且在实现过程中参考了[maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)和[mmdetection](https://github.com/open-mmlab/mmdetection)。

**支持多卡分布式训练**

**更多目标检测代码请参见[友好型的object detection代码实现和论文解读](https://blog.csdn.net/gongyi_yf/article/details/109660890)**

**backbone网络基于resnet50**

**请确保已经安装pycocotools以及1.1.0版本以上的pytorch**

# 性能
|  | map | 推理速度 |
|:-:|:-:|:-:|
|atss| 39.4 | 17.43 |
|mmdetection对应配置|39.4| 17.9 |

# 使用方法：
- git clone https://github.com/buddhisant/atss_pytorch.git
- cd atss_pytorch
- mkdir pretrained
- cd pretrained
- wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O resnet50_pytorch.pth
- wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth -O resnet101_pytorch.pth
- cd ..
- mkdir data
- cd data
- mkdir coco
- cd ../..
- python setup.py build_ext develop

将coco数据集放在./data/coco下面

# 训练
如果你的设备有多个显卡，请使用分布式训练，例如你的设备有2个显卡，请采用命令
- python -m torch.distributed.launch --nproc_per_node=2 --master_port=$((RANDOM+10000)) train.py

如果你的设备只有一个显卡，请采用普通的训练，请采用命令
- python train.py

# 测试
测试只支持单卡,请采用命令
- python test.py

# 当你不再使用该程序时
可以使用如下命令，恢复你的python环境
- pip uninstall atss-cuda -y

如有技术问题可联系qq 1401997998
