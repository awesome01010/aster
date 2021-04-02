import numpy as np
import cv2

def gamma_transform(img, gamma):
	gamma_table = [np.power(x/255.0, gamma) * 255.0 for x in range(256)]
	gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
	return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
	log_gamma_vari = np.log(gamma_vari)
	alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
	gamma = np.exp(alpha)
	return gamma_transform(img, gamma)

def rotate(xb, yb, angle):
	img_w, img_h = xb.shape[0], xb.shape[1]
	M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
	xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
	yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
	return xb, yb

def blur(img):
	k_size = np.random.choice([3, 5, 7])
	blurs = ['cv2.blur(img, (k_size, k_size))', 'cv2.GaussianBlur(img, (k_size, k_size), 0)',
			 'cv2.medianBlur(img, k_size)', 'cv2.bilateralFilter(img, k_size, 31, 31)']
	img = eval(np.random.choice(blurs))
	return img

def add_nosie(img):
	for i in range(200):
		tmp_x = np.random.randint(0, img.shape[0])
		tmp_y = np.random.randint(0, img.shape[1])
		img[tmp_x][tmp_y] = 255
	return img

def data_augment(xb):
	"""
	if np.random.random() < 0.25:
		xb, yb = rotate(xb, yb, 90)
	if np.random.random() < 0.25:
		xb, yb = rotate(xb, yb,180)
	if np.random.random() < 0.25:
		xb, yb = rotate(xb, yb, 270)
	if np.random.random() < 0.25:
		xb = cv2.flip(xb, 1)
		yb = cv2.flip(yb, 1)
	if np.random.random() < 0.25:
		xb = cv2.flip(xb, 0)
		yb = cv2.flip(yb, 0)
	"""
	if np.random.random() < 0.25:
		xb = random_gamma_transform(xb, 1.0)

	if np.random.random() < 0.25:
		xb = blur(xb)

	if np.random.random() < 0.25:
		xb = add_nosie(xb)

	return xb
