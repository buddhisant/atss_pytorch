import os
import json
import torch
import numpy as np
import config as cfg
import torch.distributed as dist

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from atss_cuda import cuda_nms

def init_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def xyxy2xywh(xyxy):
    """
    将xyxy格式的矩形框转化为xywh格式
    :param xyxy: shape为n*4, 即左上角和右下角
    :return: xywh: shape为n*4, 即左上角和wh
    """
    x1 = xyxy[:, 0]
    y1 = xyxy[:, 1]
    x2 = xyxy[:, 2]
    y2 = xyxy[:, 3]

    w = x2 - x1
    h = y2 - y1

    xywh = torch.stack([x1,y1,w,h],dim=1)
    return xywh

def get_dist_info():
    assert dist.is_initialized(), "还没有初始化分布式!"
    assert dist.is_available(),"分布式在当前设备不可用!"
    rank=dist.get_rank()
    world_size=dist.get_world_size()
    return rank,world_size

def reduce_mean(tensor):
    if not(dist.is_available() and dist.is_initialized()):
        return tensor
    tensor=tensor.clone()
    num_gpus = dist.get_world_size()
    tensor=tensor/num_gpus
    dist.all_reduce(tensor,op=dist.ReduceOp.SUM)
    return tensor

def compute_iou_xyxy(bboxes1,bboxes2):
    bboxes1 = bboxes1[:, None]
    bboxes2 = bboxes2[None, :]
    left = torch.max(bboxes1[..., 0], bboxes2[..., 0])
    top = torch.max(bboxes1[..., 1], bboxes2[..., 1])
    right = torch.min(bboxes1[..., 2], bboxes2[..., 2])
    bottem = torch.min(bboxes1[..., 3], bboxes2[..., 3])

    inter_width = (right - left).clamp(min=1e-6)
    inter_height = (bottem - top).clamp(min=1e-6)
    inter_area = inter_height * inter_width

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    union_area = area1 + area2 - inter_area
    ious = inter_area / (union_area + 1)

    return ious

def synchronize():
    """启用分布式训练时，用于各个进程之间的同步"""
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def compute_anchors(scales,device,dtype):
    all_anchors=[]
    for i,scale in enumerate(scales):
        stride=cfg.fpn_stride[i]
        y = torch.arange(scale[0],device=device,dtype=dtype)
        x = torch.arange(scale[1],device=device,dtype=dtype)
        y,x = torch.meshgrid(y,x)
        y = y.reshape(-1)
        x = x.reshape(-1)

        point = torch.stack([x,y,x,y],dim=-1)*stride

        base_anchors=[cfg.base_anchors[i]]
        base_anchors=torch.tensor(base_anchors,device=device,dtype=dtype)

        anchors=point[:,None,:]+base_anchors[None,:,:]
        anchors=anchors.view(-1,4)

        all_anchors.append(anchors)
    return all_anchors

def reg_encode(anchors,gtbboxes):
    ax=(anchors[:,0]+anchors[:,2])*0.5
    ay=(anchors[:,1]+anchors[:,3])*0.5
    aw=anchors[:,2]-anchors[:,0]
    ah=anchors[:,3]-anchors[:,1]

    gx=(gtbboxes[:,0]+gtbboxes[:,2])*0.5
    gy=(gtbboxes[:,1]+gtbboxes[:,3])*0.5
    gw=gtbboxes[:,2]-gtbboxes[:,0]
    gh=gtbboxes[:,3]-gtbboxes[:,1]

    dx=(gx-ax)/aw
    dy=(gy-ay)/ah
    dw=torch.log(gw/aw)
    dh=torch.log(gh/ah)
    delta=torch.stack([dx,dy,dw,dh],dim=-1)

    mean=delta.new_tensor(cfg.encode_mean)
    std=delta.new_tensor(cfg.encode_std)
    delta=delta.sub_(mean[None]).div_(std[None])
    return delta

def reg_decode(anchors, delta, max_shape=None):
    ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ay = (anchors[:, 1] + anchors[:, 3]) * 0.5
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]

    mean = delta.new_tensor(cfg.encode_mean)
    std = delta.new_tensor(cfg.encode_std)
    delta=delta.mul_(std[None]).add_(mean[None])

    dx = delta[:, 0]
    dy = delta[:, 1]
    dw = delta[:, 2]
    dh = delta[:, 3]
    max_ratio=np.abs(np.log(cfg.wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio,max=max_ratio)
    dh = dh.clamp(min=-max_ratio,max=max_ratio)

    predict_x = dx*aw+ax
    predict_y = dy*ah+ay
    predict_w = aw*torch.exp(dw)
    predict_h = ah*torch.exp(dh)
    x1 = predict_x - predict_w * 0.5
    y1 = predict_y - predict_h * 0.5
    x2 = predict_x + predict_w * 0.5
    y2 = predict_y + predict_h * 0.5
    if(max_shape is not None):
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1,y1,x2,y2],dim=-1)

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def load_model(model, epoch):
    """
    加载指定的checkpoint文件
    :param model:
    :param epoch:
    :return:
    """
    archive_path = os.path.join(cfg.archive_path, cfg.check_prefix+"_"+str(epoch)+".pth")
    check_point = torch.load(archive_path)
    state_dict = check_point["state_dict"]
    model.load_state_dict(state_dict)

def save_model(model,epoch):
    """保存训练好的模型，同时需要保存当前的epoch"""
    if(hasattr(model,"module")):
        model=model.module
    model_state_dict=model.state_dict()
    for key in model_state_dict.keys():
        model_state_dict[key] = model_state_dict[key].cpu()
    checkpoint=dict(state_dict=model_state_dict,epoch=epoch)
    mkdir(cfg.archive_path)
    checkpoint_name=cfg.check_prefix+"_"+str(epoch)+".pth"
    checkpoint_path=os.path.join(cfg.archive_path,checkpoint_name)

    torch.save(checkpoint,checkpoint_path)

def nms(scores, bboxes, labels, nms_thresholds):
    """
    进行nms操作
    :param scores: 每个物体的得分
    :param bboxes: 预测框
    :param labels: 每个物体的预测类别
    :param nms_thresholds: nms阈值
    :return:
    """
    _, inds = torch.sort(scores, descending=True) #首先根据score降序对bboxes和labels进行排列
    bboxes = bboxes[inds]
    labels = labels[inds]
    scores = scores[inds]

    exist_classes = torch.unique(labels) #找到图片中所有预测的类别

    result_scores = []
    result_bboxes=[]
    result_labels=[]

    for classid in exist_classes: #逐类的剔除冗余预测框
        index_per_class = (labels == classid)
        bbox_per_class = bboxes[index_per_class]
        score_per_class = scores[index_per_class]
        label_per_class = labels[index_per_class]

        inds = cuda_nms(bbox_per_class,score_per_class,nms_thresholds)

        result_scores.append(score_per_class[inds])
        result_bboxes.append(bbox_per_class[inds])
        result_labels.append(label_per_class[inds])

    result_scores = torch.cat(result_scores)
    result_bboxes = torch.cat(result_bboxes)
    result_labels = torch.cat(result_labels)
    _, inds = torch.sort(result_scores,descending=True)
    result_scores = result_scores[inds]
    result_bboxes = result_bboxes[inds]
    result_labels = result_labels[inds]

    return result_scores, result_bboxes, result_labels

def ml_nms(scores, bboxes, centerness, pos_thresholds, nms_thresholds):
    """
    对一张图片的原始预测结果进行nms
    :param scores: 分类得分，shape为[N, num_classes], 这里的N为初步筛选出的预测框数量
    :param bboxes: 预测框坐标，shape为[N, 4]。
    :param centerness: 预测框的中心度得分，shape为[N, ]
    :param pos_thresholds: 正样本的得分阈值，默认为0.05
    :param nms_thresholds: nms中的筛选阈值，默认为0.5
    :return:
    """
    num_classes = cfg.num_classes
    bboxes = bboxes[:, None].expand(scores.size(0), num_classes, 4)
    bboxes = bboxes.contiguous()
    valid_mask = scores > pos_thresholds
    bboxes = torch.masked_select(bboxes, torch.stack([valid_mask, valid_mask, valid_mask, valid_mask], -1)).view(-1, 4)

    if centerness is not None:
        scores = scores * centerness[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = bboxes.new_zeros((0, 4))
        labels = bboxes.new_zeros((0, ), dtype=torch.long)

    scores, bboxes, labels = nms(scores, bboxes, labels, nms_thresholds)

    scores = scores[:cfg.max_dets]
    bboxes = bboxes[:cfg.max_dets]
    labels = labels[:cfg.max_dets]

    return scores, bboxes, labels

def evaluate_coco(coco_gt, coco_results, output_path, iou_type="bbox"):
    with open(output_path, "w") as f:
        json.dump(coco_results, f)

    coco_dt = coco_gt.loadRes(output_path) if coco_results else COCO()
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval

"""计量时间和loss的工具"""
class AverageMeter():
    def __init__(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, ncount=1):
        self.val=val
        self.sum+=val*ncount
        self.count+=ncount
        self.avg=self.sum/self.count
