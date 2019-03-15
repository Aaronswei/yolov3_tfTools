#-coding:utf-8-*-

from keras.models import load_model
from keras.layers import Input
from keras import backend as K
import tensorflow as tf
import os
import numpy as np
from tensorflow.python.framework import graph_util,graph_io
from tensorflow.python.tools import import_pb_to_tensorboard
from yolo3.model import yolo_body

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def get_anchors(anchors_path):
	anchors_path_ = os.path.expanduser(anchors_path)
	with open(anchors_path_) as f:
		anchors = f.readline()
	anchors = [float(x) for x in anchors.split(',')]
	return np.array(anchors).reshape(-1, 2)

def get_classes(classes_path):
	classes_path_ = os.path.expanduser(classes_path)
	with open(classes_path_) as f:
		class_names = f.readlines()
	class_names = [c.strip() for c in class_names]
	return class_names

def getLayersNames(h5_model):
    out_layers = h5_model.output
    num_layers = len(h5_model.output)
    print(num_layers)
    print(out_layers)
    for l in range(num_layers):
        grid_shape = K.shape(out_layers[l])[1:3]




if __name__ == '__main__':
	weight_file = "kerasToTF/ts_ep108.h5"

	anchors_path = 'model_data/yolo_anchors.txt' 
	classes_path = 'model_data/trafficSign_classes.txt'

	num_anchors = len(get_anchors(anchors_path))
	print(get_anchors(anchors_path))
	num_classes = len(get_classes(classes_path))
	
	yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
	yolo_model.load_weights(weight_file)


	getLayersNames(yolo_model)
