import numpy as np 
import pickle

# read data
def read_cifar():
	with open('E:/datasets/cifar-10-python/cifar-10-batches-py/data_batch_1', 'rb') as f:
		train_dict_1 = pickle.load(f, encoding='bytes')

	with open('E:/datasets/cifar-10-python/cifar-10-batches-py/data_batch_2', 'rb') as f:
		train_dict_2 = pickle.load(f, encoding='bytes')

	with open('E:/datasets/cifar-10-python/cifar-10-batches-py/data_batch_3', 'rb') as f:
		train_dict_3 = pickle.load(f, encoding='bytes')

	with open('E:/datasets/cifar-10-python/cifar-10-batches-py/data_batch_4', 'rb') as f:
		train_dict_4 = pickle.load(f, encoding='bytes')

	with open('E:/datasets/cifar-10-python/cifar-10-batches-py/data_batch_5', 'rb') as f:
		train_dict_5 = pickle.load(f, encoding='bytes')

	with open('E:/datasets/cifar-10-python/cifar-10-batches-py/test_batch', 'rb') as f:
		test_dict = pickle.load(f, encoding='bytes')

	train_x = np.concatenate([train_dict_1[b'data'], train_dict_2[b'data'], train_dict_3[b'data'],
		train_dict_4[b'data'], train_dict_5[b'data']], axis=0)
	train_y = np.concatenate([train_dict_1[b'labels'], train_dict_2[b'labels'], 
		train_dict_3[b'labels'], train_dict_4[b'labels'], train_dict_5[b'labels']], axis=0)

	test_x = test_dict[b'data']
	test_y = np.array(test_dict[b'labels'])

	train_y = train_y.astype(np.int64)
	test_y = test_y.astype(np.int64)

	return train_x, train_y, test_x, test_y

def normalize_cifar(train_x, test_x):
	cifar_mean = np.mean(train_x,axis=0)
	#cifar_std = np.std(train_x, axis=0)

	train_x = (train_x - cifar_mean) #/ cifar_std
	test_x = (test_x - cifar_mean) #/ cifar_std

	return train_x, test_x

def reshape_cifar(train_x, test_x):
	train_x = np.reshape(train_x, [-1, 3, 32,32])
	train_x = np.transpose(train_x, [0, 2,3,1])
	test_x = np.reshape(test_x, [-1,3,32,32])
	test_x = np.transpose(test_x, [0,2,3,1])

	return train_x, test_x