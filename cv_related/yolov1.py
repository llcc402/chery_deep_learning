import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from preprocess_pascal_voc import * 
import numpy as np 
import cv2 

batch_size = 1
lambda_noobj = 0.5
lambda_obj = 1
lambda_coord = 5

def get_iou(bbox1, bbox2):
	'''
	NOTE
		for the ease of the computation, cx, cy, w, h are rescaled to [0,1], and they are
		set to be with respect to the whole image. We can use the computation x1 = cx - w/2 
		only in this setting
	INPUT
		bbox1    [..., 4], the last dimension denotes cx, cy, w, h
	RETURN
		iou      [...], the last dimesion is removed 
	'''
	b1_x1 = bbox1[...,0] - bbox1[...,2] / 2
	b1_y1 = bbox1[...,1] - bbox1[...,3] / 2
	b1_x2 = bbox1[...,0] + bbox1[...,2] / 2
	b1_y2 = bbox1[...,1] + bbox1[...,3] / 2

	b2_x1 = bbox2[...,0] - bbox2[...,2] / 2
	b2_y1 = bbox2[...,1] - bbox2[...,3] / 2
	b2_x2 = bbox2[...,0] + bbox2[...,2] / 2
	b2_y2 = bbox2[...,1] + bbox2[...,3] / 2

	inter_ul_x = tf.maximum(b1_x1, b2_x1)
	inter_ul_y = tf.maximum(b1_y1, b2_y1)
	inter_lr_x = tf.minimum(b1_x2, b2_x2)
	inter_lr_y = tf.minimum(b1_y2, b2_y2)

	# when the bboxes are not intersected, left upper may be even larger than right lower
	inter_area = tf.maximum(0,inter_lr_x - inter_ul_x) * tf.maximum(0,inter_lr_y - inter_ul_y)

	union_area = bbox1[...,2] * bbox1[...,3] + bbox2[...,2] * bbox2[...,3] - inter_area

	# remove small denominator
	iou = inter_area / tf.maximum(union_area, 1e-10)

	# restrict iou between 0 and 1. Setting to be 1 if > 1 and setting to be 0 if < 0
	return tf.clip_by_value(iou, 0, 1)

def loss_func(y_true, y_pred):
	'''
	INPUT
		y_true   [batch_size, num_grid, num_grid, num_class + 1 + 4] 
				 the last dimension denote whether an object exist and the position
		y_pred   [batch_size, num_grid, num_grid, (num_class + num_bbox * (1+4))] 
	'''

	pred_classes = y_pred[...,:20] # [batch_size,7,7,20]
	pred_confidences = y_pred[...,20:22] # [batch_size, 7,7,2]
	pred_bboxes = y_pred[...,22:]
	pred_bboxes = tf.reshape(pred_bboxes, (-1, 7,7,2,4))

	true_class = y_true[...,:20] # [batch_size,7,7,20]
	true_confidence = y_true[...,20:21] # [batch_size, 7,7,1]
	true_bbox = y_true[...,21:] # [batch_size, 7,7,4]
	true_bbox = tf.stack([true_bbox, true_bbox], axis=-1) # [batch_size, 7,7,4,2]
	true_bbox = tf.transpose(true_bbox, [0,1,2,4,3]) # [batch_size,7,7,2,4]
	
	pred_iou = get_iou(pred_bboxes, true_bbox) #[batch_size,7,7,2]

	# which bbox is responsible for prediction when a true object presents
	obj_bbox_mask = tf.reduce_max(pred_iou, axis=-1, keepdims=True) # [batch_size, 7,7,1]
	obj_bbox_mask = tf.cast(pred_iou >= obj_bbox_mask, tf.float32) # [batch_size, 7,7,2]
	obj_bbox_mask = obj_bbox_mask * true_confidence # [batch_size,7,7,2]

	# if there is no object in the grid cell or if the bbox is not responsible for prediction
	noobj_mask = tf.ones_like(obj_bbox_mask) - obj_bbox_mask # [batch_size, 7,7,2]

	# noobj confidence loss, when there is no object, pred confidence should approach 0
	noobj_loss = tf.reduce_sum(tf.square(pred_confidences * noobj_mask))
	noobj_loss = lambda_noobj * noobj_loss / batch_size / 49 / 2 

	# obj confidence loss, when there is some obj in the grid cell in the ground truth
	# and the bbox is responsible for prediction
	obj_loss = obj_bbox_mask * tf.square(pred_confidences - pred_iou) # [batch_size, 7,7,2]
	obj_loss = lambda_obj * tf.reduce_sum(obj_loss) / batch_size / 49 / 2

	# position loss, only bbox which is responsible for prediction is punished
	position_loss = obj_bbox_mask[...,tf.newaxis] * \
	                tf.square(pred_bboxes[...,:2] - true_bbox[...,:2])
	position_loss += obj_bbox_mask[...,tf.newaxis] * \
	                 tf.square(tf.sqrt(pred_bboxes[...,2:]) - tf.sqrt(true_bbox[...,2:])) 
	position_loss = lambda_coord * tf.reduce_sum(position_loss) / batch_size / 49 / 2

	# class loss, only grid cells with true object are punished
	class_loss = true_confidence * tf.square(true_class - pred_classes)
	class_loss = tf.reduce_sum(class_loss) / batch_size / 49 / 2

	return position_loss + noobj_loss + obj_loss + class_loss


def yolo_conv_net():
	inputs = tf.keras.Input(shape=(448,448,3))

	x = Conv2D(kernel_size=7, filters=64, strides=2, padding='same',activation='relu')(inputs) # 224
	x = MaxPool2D(pool_size=2, strides=2)(x) # 112

	x = Conv2D(filters=192, kernel_size=3,padding='same',activation='relu')(x) # 112
	x = MaxPool2D(pool_size=2, strides=2)(x) # 56

	x = Conv2D(filters=128, kernel_size=1, padding='same',activation='relu')(x) # 56
	x = Conv2D(filters=256, kernel_size=3, padding='same',activation='relu')(x) # 56
	x = Conv2D(filters=256, kernel_size=1, padding='same',activation='relu')(x) # 56
	x = Conv2D(filters=512, kernel_size=3, padding='same',activation='relu')(x) # 56
	x = MaxPool2D(pool_size=2, strides=2)(x) # 28

	for i in range(4):
		x = Conv2D(filters=256, kernel_size=1, padding='same',activation='relu')(x) # 28
		x = Conv2D(filters=512, kernel_size=3, padding='same',activation='relu')(x) # 28
	x = Conv2D(filters=512, kernel_size=1, padding='same',activation='relu')(x) # 28
	x = Conv2D(filters=1024, kernel_size=3, padding='same',activation='relu')(x) # 28
	x = MaxPool2D(pool_size=2, strides=2)(x) # 14

	for i in range(2):
		x = Conv2D(filters=512, kernel_size=1, padding='same',activation='relu')(x) # 14
		x = Conv2D(filters=1024, kernel_size=3, padding='same',activation='relu')(x) # 14
	x = Conv2D(filters=1024, kernel_size=3, padding='same',activation='relu')(x) # 14
	x = Conv2D(filters=1024, kernel_size=3, strides=2, padding='same',activation='relu')(x) # 7

	x = Conv2D(filters=1024, kernel_size=3, padding='same',activation='relu')(x) # 7
	x = Conv2D(filters=1024, kernel_size=3, padding='same',activation='relu')(x) # 7

	x = Flatten()(x) 
	x = Dense(4096,activation='relu')(x)
	x = Dense(1470, activation='relu')(x)
	outputs = tf.reshape(x, (-1, 7,7,30))

	return tf.keras.Model(inputs=inputs, outputs=outputs)

def prepro(x,y):
	x = tf.cast(x,tf.float32) / 255.0 * 2 - 1
	y = tf.cast(y, tf.float32)
	return x,y

x,y = load_data(img_folder, annotation_folder)
coord_upper = np.arange(0, 1, 1/7) + 1/7
target = get_target_all_figs(coord_upper,y)	
target = np.reshape(target, [-1,7,7,25]).astype(np.float32)
x = [cv2.resize(i,[448,448])[np.newaxis,...] for i in x]
x = np.concatenate(x, axis=0)

data = tf.data.Dataset.from_tensor_slices((x,target))
data = data.map(prepro)
data = data.shuffle(5011).batch(10)

model = yolo_conv_net()
model.compile(loss = loss_func, optimizer=tf.keras.optimizers.Adam(1e-3))
model.fit(data, epochs=5)