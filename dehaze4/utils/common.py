import numpy as np
import cv2


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class ListAverageMeter(object):
	def __init__(self):
		self.len = 10000
		self.reset()

	def reset(self):
		self.val = [0] * self.len
		self.avg = [0] * self.len
		self.sum = [0] * self.len
		self.count = 0

	def set_len(self, n):
		self.len = n
		self.reset()

	def update(self, vals, n=1):
		assert len(vals) == self.len, 'length of vals not equal to self.len'
		self.val = vals
		for i in range(self.len):
			self.sum[i] += self.val[i] * n
		self.count += n
		for i in range(self.len):
			self.avg[i] = self.sum[i] / self.count
			

def read_img(filename):
	img = cv2.imread(filename)
	return img[:, :, ::-1].astype('float32') / 255.0


def write_img(filename, img):
	img = np.round((img[:, :, ::-1].copy() * 255.0)).astype('uint8')
	cv2.imwrite(filename, img)


def write_img2(filename, img):
	# 检查图像通道数
	if len(img.shape) == 3 and img.shape[2] == 3:  # 3通道RGB图像
		img = np.round((img[:, :, ::-1].copy() * 255.0)).astype('uint8')
	elif len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):  # 单通道图像
		# 如果是3维单通道，转换为2维
		if len(img.shape) == 3:
			img = img.squeeze()
		img = np.round((img.copy() * 255.0)).astype('uint8')
	else:
		raise ValueError(f"Unsupported image shape: {img.shape}")

	cv2.imwrite(filename, img)


def hwc_to_chw(img):
	return np.transpose(img, axes=[2, 0, 1]).copy()


def chw_to_hwc(img):
	return np.transpose(img, axes=[1, 2, 0]).copy()
