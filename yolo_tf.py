#-*-coding:utf-8-*-
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
import os
import numpy as np

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def postprocess(out_layers, anchors, classes, input_shape, image_shape, score_threshold=0.4, iou_threshold=0.4):
	num_layers = len(out_layers)
	anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
	
	boxes = []
	box_scores = []	

	for mask in range(num_layers):
		_boxes, _box_scores = yolo_boxes_and_scores(out_layers[mask],
													anchors[anchor_mask[mask]],
													len(classes), 
													input_shape,
													image_shape)
		boxes.append(_boxes)
		box_scores.append(_box_scores)


	boxes = np.concatenate(boxes, axis=0)  # [3 *None*13*13*3, 4]
	box_scores = np.concatenate(box_scores, axis=0)  # [3 *None*13*13*3, 80]

	mask = box_scores >= score_threshold  # False & True in [3*None*13*13*3, 80] based on box_scores

	boxes_ = []
	scores_ = []
	classes_ = []
	
	for cl in range(len(classes)):
		class_boxes = []
		class_box_scores = []
		for i in range(len(mask[:,cl])):
			if(mask[i,cl] == True):
				class_boxes.append(boxes[i])
				class_box_scores.append(box_scores[i, cl])

		bounding_boxes = np.asarray(class_boxes, dtype=np.float32)
		confidence_score = np.asarray(class_box_scores, dtype=np.float32)
		
		picked_boxes, picked_score = nms(bounding_boxes, confidence_score, iou_threshold)
		class_ = np.ones_like(picked_score) * cl

		if picked_boxes != []:
			boxes_.append(picked_boxes)
			scores_.append(picked_score)
			classes_.append(class_)
	

	boxes_ = np.concatenate(boxes_, axis=0)
	scores_ = np.concatenate(scores_, axis=0)
	classes_ = np.concatenate(classes_, axis=0)
	
	return boxes_, scores_, classes_


def yolo_head(feature_maps, anchors, num_classes, input_shape):
	num_anchors = len(anchors)
	anchors_tensor = anchors.astype(feature_maps.dtype)
	anchors_tensor = np.reshape(anchors_tensor, [1, 1, 1, num_anchors, 2])

	grid_shape = np.shape(feature_maps)[1:3]
	grid_y = range(0, grid_shape[0])
	grid_x = range(0, grid_shape[1])

	grid_y = np.reshape(grid_y, [-1, 1, 1, 1])
	grid_x = np.reshape(grid_x, [1, -1, 1, 1])
	
	grid_y = np.tile(grid_y, [1, grid_shape[0], 1, 1])
	grid_x = np.tile(grid_x, [grid_shape[1], 1, 1, 1])

	grid = np.concatenate((grid_x, grid_y), axis=-1)
	grid = grid.astype(feature_maps.dtype)

	feature_maps_reshape = np.reshape(feature_maps, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

	box_xy = sigmoid(feature_maps_reshape[..., :2])
	box_wh = np.exp(feature_maps_reshape[..., 2:4])

	box_confidence = sigmoid(feature_maps_reshape[..., 4:5])  # [None, 13, 13, 3, 1]
	box_class_probs = sigmoid(feature_maps_reshape[..., 5:]) 

	box_xy = (box_xy + grid) / grid_shape[::-1]
	box_wh = box_wh * anchors_tensor / input_shape[::-1]

	return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
	box_yx = box_xy[..., ::-1]  # (None, 13, 13, 3, 2) => ex: , x,y --> y,x
	box_hw = box_wh[..., ::-1]

	input_shape = input_shape.astype(box_yx.dtype)
	image_shape = image_shape.astype(box_yx.dtype)

	constant = (input_shape / image_shape)
	min_ = np.minimum(constant[0], constant[1])

	new_shape = image_shape * min_  # 640*(416/640), 480*(416/640)
	new_shape = np.round(new_shape)  # lam tron ---> (416, 312)

	offset = (input_shape - new_shape) / (input_shape*2.)  # 0,  (416-312)/2/416=0.125
	scale = input_shape / new_shape  # (1, 416/312)

	# box in scale
	box_yx = (box_yx - offset) * scale  # (x-0)*1, (y-0.125)*416/312
	box_hw *= scale  # h*1, w*1.333
	box_mins = box_yx - (box_hw / 2.)  # (x-0)*1-h*1/2 = y_min, (y-0.125)*(416/312)-w*(416/312)/2 = x_min
	box_maxes = box_yx + (box_hw / 2.)  # (x-0)*1+h*1/2 = y_max, (y-0.125)*(416/312)+w*(416/312)/2 = x_max
	
	boxes = np.concatenate([box_mins[..., 0:1],  # y_min
                           box_mins[..., 1:2],  # x_min
                           box_maxes[..., 0:1],  # y_max
                           box_maxes[..., 1:2]],  # x_max
                           axis=-1)

	boxes = np.multiply(boxes,np.concatenate([image_shape, image_shape], axis=-1))
	return boxes


def yolo_boxes_and_scores(feats,anchors, num_classes, input_shape, image_shape):
	box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
	boxes = yolo_correct_boxes(box_xy, box_wh, input_shape,image_shape)

	boxes = np.reshape(boxes, [-1, 4])

	box_scores = box_confidence * box_class_probs 
	box_scores = np.reshape(box_scores, [-1, num_classes])

	return boxes, box_scores


def nms(bounding_boxes, score, threshold):
	if len(bounding_boxes) == 0:
		return [], []

	# coordinates of bounding boxes
	start_x = bounding_boxes[:, 0]
	start_y = bounding_boxes[:, 1]
	end_x = bounding_boxes[:, 2]
	end_y = bounding_boxes[:, 3]

	# Picked bounding boxes
	picked_boxes = []
	picked_score = []

	# Compute areas of bounding boxes
	areas = (end_x - start_x + 1) * (end_y - start_y + 1)

	# Sort by confidence score of bounding boxes
	order = np.argsort(score)

	# Iterate bounding boxes
	while order.size > 0:
		# The index of largest confidence score
		index = order[-1]
		# Pick the bounding box with largest confidence score
		picked_boxes.append(bounding_boxes[index])
		picked_score.append(score[index])

		# Compute ordinates of intersection-over-union(IOU)
		x1 = np.maximum(start_x[index], start_x[order[:-1]])
		x2 = np.minimum(end_x[index], end_x[order[:-1]])
		y1 = np.maximum(start_y[index], start_y[order[:-1]])
		y2 = np.minimum(end_y[index], end_y[order[:-1]])

		# Compute areas of intersection-over-union
		w = np.maximum(0.0, x2 - x1 + 1)
		h = np.maximum(0.0, y2 - y1 + 1)
		intersection = w * h

		# Compute the ratio between intersection and union
		ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

		left = np.where(ratio < threshold)
		order = order[left]

	return picked_boxes, picked_score
