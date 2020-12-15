import torch
import numpy as np
from atss_cuda import ops

class NMSop(torch.autograd.Function):

    @staticmethod
    def forward(ctx, bboxes, scores, iou_threshold):

        inds = ops.nms(bboxes, scores, iou_threshold=float(iou_threshold))

        return inds

def cuda_nms(boxes, scores, iou_threshold):
    inds = NMSop.apply(boxes, scores, iou_threshold)
    return inds

if __name__=="__main__":
    np_boxes = np.array([[6.0, 3.0, 8.0, 7.0], [3.0, 6.0, 9.0, 11.0],
                         [3.0, 7.0, 10.0, 12.0], [1.0, 4.0, 13.0, 7.0]],
                        dtype=np.float32)
    np_scores = np.array([0.6, 0.9, 0.7, 0.2], dtype=np.float32)
    np_inds = np.array([1, 0, 3])
    boxes = torch.from_numpy(np_boxes).cuda()
    scores = torch.from_numpy(np_scores).cuda()

    inds = cuda_nms(boxes, scores,iou_threshold=0.3)
    print(inds)