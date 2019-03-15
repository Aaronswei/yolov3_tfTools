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

def h5_to_pb(h5_model, output_dir, model_name, out_prefix="output_", log_tensorboard=False):
    if os.path.exists(output_dir) == False:
    	os.mkdir(output_dir)

    out_nodes = []
    for i in range(len(h5_model.outputs)):
    	out_nodes.append(out_prefix + str(i + 1))
    	tf.identity(h5_model.output[i], out_prefix + str(i + 1))

    sess = K.get_session()
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    for out_node in out_nodes:
        print(out_node)
    graph_io.write_graph(main_graph, output_dir, name = model_name, as_text = False)
    if log_tensorboard:
    	import_pb_to_tensorboard.import_to_tensorboard(os.path.join(output_dir,model_name),output_dir)


if __name__ == '__main__':
	cur_path = os.getcwd()

	weight_file = "ts_ep108.h5"
	output_graph_name = weight_file[:-3] + '.pb'

	weight_file_path = os.path.join(cur_path, 'kerasToTF/' + weight_file)
	assert weight_file_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

	output_dir = os.path.join(cur_path, "kerasToTF")
	
	anchors_path = 'model_data/yolo_anchors.txt' 
	classes_path = 'model_data/trafficSign_classes.txt'

	num_anchors = len(get_anchors(anchors_path))
	num_classes = len(get_classes(classes_path))
	
	yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
	yolo_model.load_weights(weight_file_path)


	h5_to_pb(yolo_model, output_dir=output_dir, model_name= output_graph_name)
	#print('model saved')
