#-*-coding:utf-8-*-
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def load_pb(pb_file_path):
	sess = tf.Session()
	with gfile.FastGFile(pb_file_path, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		sess.graph.as_default()
		tf.import_graph_def(graph_def, name='')

	constant_values = {}
	constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
	for constant_op in constant_ops:
		print(constant_op.name)



output_graph_path = 'kerasToTF/ts_ep087.pb'
load_pb(output_graph_path)
