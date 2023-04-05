import torch
import numpy as np

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)

def confusion_matrix(detections, ground_truth, iou_thresh):
	# print('Ground Truth boxes :', ground_truth)
	# print('Detections : ', detections)
	detections = torch.Tensor(detections)
	ground_truth = torch.Tensor(ground_truth)
	"""
    Returns True Positive Count, False negative count, False Positive count of the detections and ground truth for a given IoU threshold
    Arguments:
        detections (Tensor[N,4])
        ground_truth (Tensor[M, 4])
		IoU threshold, generally 0.5
    Returns:
        Returns True Positive Count, False negative count, False Positive count
    """
	# print('Ground Truth boxes :', ground_truth)
	# print('Detections : ', detections)
	if len(ground_truth) == 0 and len(detections) != 0:
		return (0,len(detections),0)
	elif len(ground_truth) != 0 and len(detections) == 0:
		return (0,0,len(ground_truth))
	else:
		iou = box_iou(ground_truth, detections) #Getting an NXM matrix of IoU of all possible detections and ground truth annotations
		potential_matches = torch.where(iou > iou_thresh) #indices where iou is greater than the threshold
		if potential_matches[0].shape[0]: #If there are more than 1 posible matches (atleast 1 signficant detection)
			matches = torch.cat((torch.stack(potential_matches, 1), iou[potential_matches[0], potential_matches[1]][:, None]), 1).cpu().numpy() #indices where the IoU is greater than the threshold along with the IoU values
			if potential_matches[0].shape[0] > 1:
				matches = matches[matches[:, 2].argsort()[::-1]]# matches after sorting in decreasing order by IoU
				matches = matches[np.unique(matches[:, 1], return_index=True)[1]] # matches after getting the unique values along the detections (columns)
				matches = matches[matches[:, 2].argsort()[::-1]] # matches after sorting in decreasing order by IoU
				matches = matches[np.unique(matches[:, 0], return_index=True)[1]] # matches after getting the unique values along the ground truth (rows)
		else:
			matches = np.zeros((0, 3))
		true_positives_count = len(matches)
		false_postives_count = len(detections) - len(matches)
		false_negatives_count = len(ground_truth) - len(matches)
		return (true_positives_count,false_postives_count,false_negatives_count)
	

# detections = torch.Tensor([[10,20,30,40],[10,20,30,50]])
# ground_truth = torch.Tensor([[10,20,30,40],[50,60,70,80]])
# print(confusion_matrix(detections,ground_truth,0.5))

