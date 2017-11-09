from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import csv

def plot_vals(img_1, img_2, img_3):

	fig = plt.figure()

	# show first image
	ax = fig.add_subplot(1, 3, 1)
	ax.text(180, -10, 'INPUT', style='normal', horizontalalignment='left')
	plt.imshow(img_1, cmap = plt.cm.gray)
	plt.axis("off")
	
	# show the second image
	ax = fig.add_subplot(1, 3, 2)
	ax.text(180, -10, 'TARGET', style='normal', horizontalalignment='left')
	plt.imshow(img_2, cmap = plt.cm.gray)
	plt.axis("off")

	# show the second image
	ax = fig.add_subplot(1, 3, 3)
	ax.text(180, -10, 'EMBEDDED', style='normal', horizontalalignment='left')
	plt.imshow(img_3, cmap = plt.cm.gray)
	plt.axis("off")
	
	# show the images
	plt.show()


inp = cv2.imread('plotting_samples/input2.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
tg = cv2.imread('plotting_samples/target2.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
emb = cv2.imread('plotting_samples/emb2.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
plot_vals(inp, tg, emb)