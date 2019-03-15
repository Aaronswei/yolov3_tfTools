#-*-coding:utf-8-*-
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
import os
import colorsys

from yolo_tf import postprocess
from PIL import Image, ImageFont, ImageDraw

import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


class YOLO:

    def __init__(self,  
                model_path='../models/trafficSign.pb', 
                anchors_path='../models/yolo_anchors.txt', 
                classes_path='../models/trafficSign_classes.txt',
                score = 0.3,
                iou = 0.45,
                model_image_size = [416,416]):
        self.model_path = model_path
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.score = score
        self.iou = iou
        self.model_image_size = np.array(model_image_size)

        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.load_model()
        self.use_colors()


    def use_colors(self):
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.


    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def load_model(self):
        with gfile.FastGFile(self.model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            self.tensors = tf.import_graph_def(graph_def, name='')
            print(self.tensors)

            self.input_x = self.sess.graph.get_tensor_by_name('input_1:0')
            self.out1 = self.sess.graph.get_tensor_by_name('conv2d_59/BiasAdd:0')
            self.out2 = self.sess.graph.get_tensor_by_name('conv2d_67/BiasAdd:0')
            self.out3 = self.sess.graph.get_tensor_by_name('conv2d_75/BiasAdd:0')

    def detect_image(self, image):

        assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
        assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
        
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        image_w, image_h = image.size
        image_shape = np.array([image_h, image_w])
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        outs = self.sess.run(
            [self.out1, self.out2, self.out3],
            feed_dict={self.input_x: image_data})


        self.out_boxes, self.out_scores, self.out_classes = postprocess(outs, self.anchors, self.class_names, self.model_image_size, image_shape, self.score, self.iou)

        print('Found {} boxes for {}'.format(len(self.out_boxes), 'img'))

        self.font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        self.thickness = (image.size[0] + image.size[1]) // 300

        postImage = self.drawBox(image)

        return postImage

    
    def drawBox(self, image):
        for i, c in reversed(list(enumerate(self.out_classes))):
            predicted_class = self.class_names[int(c)]
            box = self.out_boxes[i]
            score = self.out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, self.font)

            top, left, bottom, right = box  # y_min, x_min, y_max, x_max
            top = max(0, np.floor(top + 0.5).astype(np.int32))
            left = max(0, np.floor(left + 0.5).astype(np.int32))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype(np.int32))
            right = min(image.size[0], np.floor(right + 0.5).astype(np.int32))
            print(label, (left, top), (right, bottom))  # (x_min, y_min), (x_max, y_max)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for j in range(self.thickness):
                draw.rectangle([left + j, top + j, right - j, bottom - j], outline=self.colors[int(c)])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[int(c)])
            draw.text(text_origin, label, fill=(0, 0, 0), font=self.font)
            del draw

        return image


if __name__ == "__main__":
    #image_file = "../models/09160.png"
    image_file = "../models/timg2.jpg"

    image = Image.open(image_file)

    yolo = YOLO('../models/trafficSign.pb',
                '../models/yolo_anchors.txt',
                '../models/trafficSign_classes.txt')

    r_image = yolo.detect_image(image)
    r_image.show()