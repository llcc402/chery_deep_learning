import json
import xml.etree.ElementTree as ET 
import os 
import matplotlib.pyplot as plt 
import cv2 
import numpy as np 
import tensorflow as tf 

#----------------------------- process annotations ----------------------------
annotation_folder='E:/datasets/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations'
# labels from texts to integers
voc_labels = ['aeroplane','bicycle','bird','boat','bottle',
		'bus','car','cat','dog','horse','motorbike',
		'person','pottedplant','sheep','sofa','chair',
		'cow','diningtable','train','tvmonitor']
label_map = {v:k for (k,v) in enumerate(voc_labels)}
re_label_map = {k:v for (v,k) in label_map.items()}

def parse_annotation(anno_path):
	'''
		This function parse the annotations of the images. The original annotations record
	the coordinates of the upper left and lowe right. We need to change them to 
	[upper left x, upper left y, width, height] 
		These coordinates should be rescaled to be in the interval [0,1]. The first 2 values
	are set to be with respect to the upper left corner of the image and the width and height 
	should be set to be with respect to the width and height of the image.
	'''

	tree = ET.parse(anno_path)
	root = tree.getroot()

	boxes = []
	labels = []

	img_size = root.find('size')
	img_width = float(img_size.find('width').text)
	img_height = float(img_size.find('height').text)

	for obj in root.iter('object'):
		label = obj.find('name').text.lower().strip()
		
		if label in voc_labels:
			bbox = obj.find('bndbox')
			xmin = float(bbox.find('xmin').text) - 1 
			ymin = float(bbox.find('ymin').text) - 1
			xmax = float(bbox.find('xmax').text) - 1
			ymax = float(bbox.find('ymax').text) - 1 

			w = (xmax - xmin) / img_width
			h = (ymax - ymin) / img_height

			xmin = xmin / img_width
			ymin = ymin / img_height
			xmax = xmax / img_width
			ymax = ymax / img_height

			cx = (xmin + xmax) / 2 
			cy = (ymin + ymax) / 2 

			labels.append(label_map[label])
			boxes.append([cx,cy,w,h])
	return labels, boxes

#------------------------- process images --------------------------
img_folder = 'E:/datasets/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'


def resize_and_plot(img, boxes):
	'''
	INPUT 
		boxes    [xmin, ymin, w, h], all the values are in [0,1]
	'''
	img_resized = cv2.resize(img, [448,448])
	for box in boxes:
		cx, cy, w, h = box
		xmin = np.clip(round((cx - w/2) * 448), 0, 447)
		ymin = np.clip(round((cy - h/2) * 448), 0, 447)
		xmax = np.clip(round((cx + w/2) * 448), 0, 447)
		ymax = np.clip(round((cy + h/2) * 448), 0, 447)

		cv2.rectangle(img_resized, (xmin,ymin), (xmax,ymax), [0,255,0], 1)
	plt.imshow(img_resized)
	plt.show()

def load_data(img_folder, annotation_folder):

	# get image file_dirs
	imgs = os.listdir(img_folder)
	imgs = [os.path.join(img_folder, i) for i in imgs]

	# get annotation filr_dirs
	annos = os.listdir(annotation_folder)
	annos = [i for i in annos if i.endswith('xml')]
	annos = [os.path.join(annotation_folder,i) for i in annos]

	# get all labels
	y = []
	for anno in annos:
		labels, boxes = parse_annotation(anno)
		y.append((labels, boxes))

	x = []
	for img_dir in imgs:
		x.append(cv2.imread(img_dir))

	return x,y

def get_target_one_obj(coord_upper,box):
	'''
	INPUT
		c              int, which class, class 0 ~ class 19, 20 classes in total
		box            [cx, cy, w, h]
		coord_upper    [7,]
	OUTPUT
		[49,5]    one hot for confidence of the first col

	'''
	cx, cy, w, h = box
	idx = 0
	for i in range(len(coord_upper)):
		if cx > coord_upper[i]:
			idx += 7
		else:
			for j in range(len(coord_upper)):
				if cy > coord_upper[j]:
					idx += 1
				else:
					break
			break

	return idx 
			
def get_target_one_fig(coord_upper, c, boxes):
	'''
	INPUT
		boxes    a list of boxes
		c        a list of class labels, class 0 ~ class 19, 20 classes in total
	'''
	outputs = np.zeros([len(coord_upper) * len(coord_upper), 25])
	for i in range(len(boxes)):
		idx = get_target_one_obj(coord_upper, boxes[i])
		outputs[idx, c[i]] = 1 # one hot class label
		outputs[idx, 20] = 1 # confidence
		outputs[idx, -4:] = boxes[i]
	return outputs

def get_target_all_figs(coord_upper,y):
	'''
	INPUT
		y     a list of figure informations
			  y[i][0]   lables
			  y[i][1]   boxes
	OUTPUT
		[num_boxes, 1], boxes should run over all figures and all boxes
	'''
	outputs = []
	for i in range(len(y)):
		outputs.append(get_target_one_fig(coord_upper, y[i][0], y[i][1]))
	return np.concatenate(outputs, axis=0)


if __name__ == "__main__":
	x,y = load_data(img_folder, annotation_folder)
	idx = 0 
	# resize_and_plot(x[idx], y[idx][1])
	coord_upper = np.arange(0, 1, 1/7) + 1/7
	target = get_target_all_figs(coord_upper,y)
