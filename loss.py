import torch
import utils
import math
import config as cfg
import numpy as np

import torch.nn.functional as F

INF = 100000000

class SigmoidFocalLoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidFocalLoss, self).__init__()
        self.weight = cfg.cls_loss_weight
        self.gamma = cfg.focal_loss_gamma
        self.alpha = cfg.focal_loss_alpha

    def forward(self, logits, targets, weight):
        """
        计算分类分支的loss, 即focal loss.
        :param logits: 神经网络分类分支的输出. type为tensor, shape为(cumsum_5(N*ni),80),其中N是batch size, ni为第i层feature map的样本数量
        :param targets: 表示分类分之的targer labels, type为tensor, shape为(cumsum_5(N*ni),), 其中N是batch size, 正样本的label介于[0,79], 负样本的label为-1
        :param weight: 权值，只有正样本和负样本参加分类loss的计算，而ignore样本不会参加
        :return loss: 所有anchor point的loss之和.
        """
        num_classes = logits.shape[1]
        device = targets.device
        dtype = targets.dtype
        class_range = torch.arange(0, num_classes, dtype=dtype, device=device).unsqueeze(0)

        t = targets.unsqueeze(1)
        p = torch.sigmoid(logits)

        term1 = (1-p)**self.gamma*torch.log(p)
        term2 = p**self.gamma*torch.log(1-p)

        loss = -(t == class_range).float()*term1*self.alpha - ((t != class_range)*(t >= 0)).float()*term2*(1-self.alpha)

        weight=weight[:,None]
        loss = loss*weight.float()
        return self.weight*loss.sum()

class GiouLoss(torch.nn.Module):
    def __init__(self):
        super(GiouLoss, self).__init__()
        self.weight = cfg.reg_loss_weight

    def forward(self,preds,targets,weights=None):
        lt = torch.max(preds[:,:2], targets[:,:2])
        rb = torch.min(preds[:,2:], targets[:,2:])
        wh=(rb-lt).clamp(min=0)
        overlap=wh[:,0] * wh[:,1]

        ap = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
        ag = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
        union = ap + ag - overlap + 1

        ious = overlap / union

        enclose_x1y1 = torch.min(preds[:, :2], targets[:, :2])
        enclose_x2y2 = torch.max(preds[:, 2:], targets[:, 2:])
        enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1] + 1

        gious = ious - (enclose_area - union) / enclose_area
        loss = 1 - gious
        if(weights is not None):
            loss = loss*weights
        return self.weight*loss.sum()

class CenternessLoss(torch.nn.Module):
    def __init__(self):
        super(CenternessLoss, self).__init__()
        self.weight=cfg.cen_loss_weight

    def forward(self, centerness_pred, centerness_target):
        loss = F.binary_cross_entropy_with_logits(centerness_pred, centerness_target,reduction='none')
        return self.weight*loss.sum()

class AtssLoss(torch.nn.Module):
    def __init__(self):
        super(AtssLoss, self).__init__()
        self.cls_loss_func = SigmoidFocalLoss()
        self.reg_loss_func = GiouLoss()
        self.cen_loss_func = CenternessLoss()

    def compute_valid_flag(self, fin_img_shape, scales, device):
        valid_flag_per_img=[]
        for i, scale in enumerate(scales):
            stride=cfg.fpn_stride[i]
            h_fpn = scale[0]
            w_fpn = scale[1]
            h_valid = math.ceil(fin_img_shape[0]/stride)
            w_valid = math.ceil(fin_img_shape[1]/stride)

            y_valid = torch.zeros((h_fpn,), device=device, dtype=torch.bool)
            x_valid = torch.zeros((w_fpn,), device=device, dtype=torch.bool)
            x_valid[:w_valid] = 1
            y_valid[:h_valid] = 1

            y_valid,x_valid = torch.meshgrid(y_valid,x_valid)
            y_valid=y_valid.reshape(-1)
            x_valid=x_valid.reshape(-1)
            valid_flag_per_level=y_valid&x_valid

            valid_flag_per_img.append(valid_flag_per_level)
        return valid_flag_per_img

    def unmap(self,data,inds,fill=0):
        if data.dim()==1:
            ret = data.new_full((inds.size(0),),fill)
            ret[inds.type(torch.bool)] = data
        else:
            new_size = (inds.size(0),)+data.size()[1:]
            ret = data.new_full(new_size,fill)
            ret[inds.type(torch.bool),:]=data
        return ret

    def compute_centerness_targets(self, anchors, gt_bboxes):
        if(anchors.dim()==1):
            anchors=anchors[None].contiguous()
        if(gt_bboxes.dim()==1):
            gt_bboxes=gt_bboxes[None].contiguous()
        center_x=(anchors[:,0]+anchors[:,2])*0.5
        center_y=(anchors[:,1]+anchors[:,3])*0.5

        l=center_x-gt_bboxes[:,0]
        t=center_y-gt_bboxes[:,1]
        r=gt_bboxes[:,2]-center_x
        b=gt_bboxes[:,3]-center_y

        return torch.sqrt((torch.min(l,r)/torch.max(l,r))*(torch.min(t,b)/torch.max(t,b)))

    def compute_targets(self, scales, gt_bboxes, gt_labels, fin_img_shape):
        cls_targets=[]
        reg_targets=[]

        dtype = gt_bboxes[0].dtype
        device = gt_bboxes[0].device

        anchors = utils.compute_anchors(scales,device,dtype)
        num_anchors_per_level=[anchor.size(0) for anchor in anchors]
        anchors = torch.cat(anchors,dim=0)

        batch_valid_flags = []
        for shape in fin_img_shape:
            batch_valid_flags.append(self.compute_valid_flag(shape,scales,device))

        for i, gt_bbox in enumerate(gt_bboxes):
            num_gt=gt_bbox.size(0)
            valid_flags_cur_img=batch_valid_flags[i]
            num_valid_cur_img_per_level=[valid_per_level.int().sum().item() for valid_per_level in valid_flags_cur_img]
            valid_flags_cur_img=torch.cat(valid_flags_cur_img)
            valid_anchors_cur_img=anchors[valid_flags_cur_img]
            num_valid_cur_img=valid_anchors_cur_img.size(0)

            anchors_cx = (valid_anchors_cur_img[:,0]+valid_anchors_cur_img[:,2])*0.5
            anchors_cy = (valid_anchors_cur_img[:,1]+valid_anchors_cur_img[:,3])*0.5
            anchors_center = torch.stack([anchors_cx,anchors_cy],dim=-1)

            gt_bbox_cx = (gt_bbox[:,0]+gt_bbox[:,2])*0.5
            gt_bbox_cy = (gt_bbox[:,1]+gt_bbox[:,3])*0.5
            gt_bbox_center = torch.stack([gt_bbox_cx,gt_bbox_cy],dim=-1)

            distance=(anchors_center[:,None,:]-gt_bbox_center[None,:,:]).pow(2).sum(-1).sqrt()
            distance=torch.split(distance,num_valid_cur_img_per_level,dim=0)

            candidate_ids=[]
            start_idx=0
            for distance_per_level in distance:
                num_selected_cur_level=min(distance_per_level.size(0),cfg.num_candidate_per_level)
                _, candidate_ids_cur_level=distance_per_level.topk(num_selected_cur_level,dim=0,largest=False)
                candidate_ids.append(candidate_ids_cur_level+start_idx)
                start_idx+=distance_per_level.size(0)

            candidate_ids=torch.cat(candidate_ids,dim=0)
            overlaps = utils.compute_iou_xyxy(valid_anchors_cur_img, gt_bbox)
            candidate_overlaps=overlaps[candidate_ids,range(num_gt)]
            overlap_mean=candidate_overlaps.mean(0)
            overlap_std=candidate_overlaps.std(0)
            overlap_threshold=overlap_mean+overlap_std

            is_pos = candidate_overlaps >= overlap_threshold[None,:]

            candidate_cx = (anchors_cx[candidate_ids.view(-1)]).reshape(-1,num_gt)
            candidate_cy = (anchors_cy[candidate_ids.view(-1)]).reshape(-1,num_gt)
            l = candidate_cx - gt_bbox[:, 0][None].contiguous()
            t = candidate_cy - gt_bbox[:, 1][None].contiguous()
            r = gt_bbox[:, 2][None].contiguous() - candidate_cx
            b = gt_bbox[:, 3][None].contiguous() - candidate_cy

            distance = torch.stack([l, t, r, b],dim=-1)
            is_inside = distance.min(dim=-1)[0] > 0.01

            is_pos = is_pos & is_inside

            for k in range(num_gt):
                candidate_ids[:, k] = candidate_ids[:, k]+k*num_valid_cur_img
            overlaps_inf = torch.full_like(overlaps, -INF).t().contiguous().view(-1)
            index = candidate_ids.view(-1)[is_pos.view(-1)]
            overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
            overlaps_inf = overlaps_inf.view(num_gt, -1).t().contiguous()

            max_overlaps, argmax_overlaps=overlaps_inf.max(dim=1)
            assigned_gt_inds=overlaps.new_zeros((num_valid_cur_img,),dtype=torch.long)
            assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF]+1
            if gt_labels is not None:
                assigned_labels = assigned_gt_inds.new_full((num_valid_cur_img,),cfg.num_classes)
                pos_inds = torch.nonzero(assigned_gt_inds>0,as_tuple=False).squeeze()
                if(pos_inds.dim()==0):
                    pos_inds=pos_inds[None]
                if pos_inds.numel() >0:
                    gt_label = gt_labels[i]
                    assigned_labels[pos_inds] = gt_label[assigned_gt_inds[pos_inds]-1]

            cls_targets_cur_img = self.unmap(assigned_labels,valid_flags_cur_img,cfg.num_classes)
            cls_targets.append(cls_targets_cur_img)

            pos_gt_bbox_cur_img=gt_bbox[assigned_gt_inds[pos_inds]-1,:]
            pos_anchor_cur_img = valid_anchors_cur_img[pos_inds]
            pos_reg_targets = utils.reg_encode(pos_anchor_cur_img, pos_gt_bbox_cur_img)

            valid_reg_targets = torch.zeros_like(valid_anchors_cur_img)
            valid_reg_targets[pos_inds,:]=pos_reg_targets
            reg_targets.append(self.unmap(valid_reg_targets,valid_flags_cur_img))

            batch_valid_flags[i] = valid_flags_cur_img

        return cls_targets,reg_targets,anchors,batch_valid_flags,num_anchors_per_level

    def forward(self,cls_preds, reg_preds, cen_preds, gt_bboxes, gt_labels, fin_img_shape):
        batch_size=cls_preds[0].size(0)
        scales = [cls_pred.shape[-2:] for cls_pred in cls_preds]
        num_levels = len(scales)

        cls_targets, reg_targets,anchors,batch_valid_flags,num_anchors_per_level = self.compute_targets(scales, gt_bboxes, gt_labels, fin_img_shape)

        anchors = [anchors for _ in range(batch_size)]
        anchors = [anchor.split(num_anchors_per_level) for anchor in anchors]
        cls_targets = [cls_target.split(num_anchors_per_level) for cls_target in cls_targets]
        reg_targets = [reg_target.split(num_anchors_per_level) for reg_target in reg_targets]
        batch_valid_flags = [batch_valid_flag.split(num_anchors_per_level) for batch_valid_flag in batch_valid_flags]

        anchors_level_first = []
        cls_targets_level_first = []
        reg_targets_level_first = []
        valid_flags_level_first = []
        for i in range(num_levels):
            anchors_level_first.append(torch.cat([anchor[i] for anchor in anchors]))
            cls_targets_level_first.append(torch.cat([cls_target[i] for cls_target in cls_targets]))
            reg_targets_level_first.append(torch.cat([reg_target[i] for reg_target in reg_targets]))
            valid_flags_level_first.append(torch.cat([batch_valid_flag[i] for batch_valid_flag in batch_valid_flags]))

        anchors_level_first = torch.cat(anchors_level_first)
        cls_targets_level_first = torch.cat(cls_targets_level_first)
        reg_targets_level_first = torch.cat(reg_targets_level_first)
        valid_flags_level_first = torch.cat(valid_flags_level_first)

        cls_preds = [cls_pred.permute(0, 2, 3, 1).reshape(-1, cfg.num_classes) for cls_pred in cls_preds]
        reg_preds = [reg_pred.permute(0, 2, 3, 1).reshape(-1, 4) for reg_pred in reg_preds]
        cen_preds = [cen_pred.permute(0, 2, 3, 1).reshape(-1) for cen_pred in cen_preds]

        cls_preds_all = torch.cat(cls_preds)
        reg_preds_all = torch.cat(reg_preds)
        cen_preds_all = torch.cat(cen_preds)

        pos_inds = torch.nonzero((cls_targets_level_first >= 0) & (cls_targets_level_first != cfg.num_classes), as_tuple=False).reshape(-1)
        num_pos = utils.reduce_mean(torch.tensor(pos_inds.size(0)).float().cuda()).item()
        cls_loss = self.cls_loss_func(cls_preds_all,cls_targets_level_first,valid_flags_level_first) / num_pos

        anchors_pos = anchors_level_first[pos_inds]
        reg_preds_pos=reg_preds_all[pos_inds]
        cen_preds_pos=cen_preds_all[pos_inds]
        reg_targets_pos=reg_targets_level_first[pos_inds]


        if(num_pos>0):
            bboxes_predict = utils.reg_decode(anchors_pos, reg_preds_pos)
            bboxes_target = utils.reg_decode(anchors_pos,reg_targets_pos)
            cen_targets_pos = self.compute_centerness_targets(anchors_pos,bboxes_target)

            sum_cen = utils.reduce_mean(cen_targets_pos.sum()).item()

            reg_loss=self.reg_loss_func(bboxes_predict,bboxes_target,cen_targets_pos)/sum_cen
            cen_loss=self.cen_loss_func(cen_preds_pos,cen_targets_pos)/num_pos
        else:
            reg_loss=cls_loss.new_tensor(0,requires_grad=True)
            cen_loss=cls_loss.new_tensor(0,requires_grad=True)

        return dict(cls_loss=cls_loss, reg_loss=reg_loss, cen_loss=cen_loss)

