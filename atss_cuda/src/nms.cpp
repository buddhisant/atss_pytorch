#include <torch/extension.h>
#include <vector>

using namespace at;

Tensor NMSCUDAKernelLauncher(Tensor boxes, Tensor scores, float iou_threshold);

Tensor nms(Tensor boxes, Tensor scores, float iou_threshold)
{
    return NMSCUDAKernelLauncher(boxes, scores, iou_threshold);
}