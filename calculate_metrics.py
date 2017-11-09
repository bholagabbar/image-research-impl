from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import csv

"""
Mean Squared Error' between the two same-dimension images is
- The sum of the squared difference between the two images
- The lower the error, the more similar the two images are
"""
def mse(img_1, img_2):
	
	err = np.sum((img_1.astype("float") - img_2.astype("float")) ** 2)
	err /= float(img_1.shape[0] * img_1.shape[1])
	return err
 
# Peak Signal To Noise Ratio
# formula - https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
def psnr(img_1, img_2, mse_v):
	maxi = np.amax(img_1)
	psnr_val = (20 * math.log10(maxi)) - (10 * math.log10(mse_v))
	return psnr_val

# Mean Absolute Error
# https://stackoverflow.com/questions/33359411/mean-absolute-error-python
def mae(img_1, img_2):
	mae_val = np.sum(np.absolute((img_2.astype("float") - img_1.astype("float"))))
	mae_val /= (float(img_1.shape[0] * img_1.shape[1] * 255))
	return mae_val	


def plot_vals(img_1, img_2, title, m, s, p, mae_val):
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f, PSNR: %.2f, MAE: %.2f" % (m, s, p, mae_val))
	
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	ax.text(180, -10, 'TARGET', style='normal', horizontalalignment='left')
	plt.imshow(img_1, cmap = plt.cm.gray)
	plt.axis("off")
	
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	ax.text(180, -10, 'EMBEDDED', style='normal', horizontalalignment='left')
	plt.imshow(img_2, cmap = plt.cm.gray)
	plt.axis("off")
	
	# show the images
	plt.show()

def compare_images(img_1, img_2, title):
	m = round(mse(img_1, img_2), 4)
	s = round(ssim(img_1, img_2), 4)
	p = round(psnr(img_1, img_2, m), 4)
	mae_val = round(mae(img_1, img_2), 4)

	plot_vals(img_1, img_2, title, m, s, p, mae_val)	

	return m, s, p, mae_val


table = [['','0.1P','','','','','0.2P', '','','','','0.4P', '','','','','0.5P', '','','','','0.6P', '','','','','0.8P', '','','','']]
table.append(['','MSE', 'SSIM', 'PSNR', 'MAE','','MSE', 'SSIM', 'PSNR', 'MAE','','MSE', 'SSIM', 'PSNR', 'MAE','','MSE', 'SSIM', 'PSNR', 'MAE','','MSE', 'SSIM', 'PSNR', 'MAE','','MSE', 'SSIM', 'PSNR', 'MAE'])


def main():
	pairing_prefixes = ['1_32', '2_12', '5_88', '20_47', '23_13', '24_26', '25_71', '28_15', '34_36', '37_21', '38_51', '40_31', '58_55', '72_39', '73_48', '75_45', '86_76', '96_35', '104_105','106_102']
	payload_dir_suffix = ['10', '20', '40', '50', '60', '80']
	with open('metrics_calc_result.csv', 'w') as csvfile:
		writer = csv.writer(csvfile)
		for i in range(1, 21):
			arr = [str(i)]
			for j in payload_dir_suffix:
				path = 'calc_results/' + str(i) + '/payload_' + str(j) + '/'
				
				tg_img = cv2.imread(path + 'target_img.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)

				em_img = cv2.imread(path + 'encoded_img.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)

				vals = compare_images(tg_img, em_img, "Target vs. Embedded")
				arr.append(vals[0])
				arr.append(vals[1])
				arr.append(vals[2])
				arr.append(vals[3])
				arr.append('')
				
			table.append(arr)
		[writer.writerow(r) for r in table]

# def plot_samples():
# 	path = 'final_image_pairings/'
# 	one_tg = cv2.imread(path + '20_47_initial_target.pgm', cv2.CV_LOAD_IMAGE_GRAYSCALE)
# 	one_em = cv2.imread(path + '20_47_final_embedded.pgm', cv2.CV_LOAD_IMAGE_GRAYSCALE)
# 	compare_images(one_tg, one_em, 'Target vs Embedded')




# main()
plot_samples()