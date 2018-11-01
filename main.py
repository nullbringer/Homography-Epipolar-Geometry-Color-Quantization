UBIT = 'amlangup'
import numpy as np
np.random.seed(sum([ord(c) for c in UBIT]))

import cv2
import matplotlib.pyplot as plt

SOURCE_FOLDER = 'data/'
OUTPUT_FOLDER = 'output/'


def print_image(img, image_name):
	cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
	cv2.imshow(image_name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	

def write_image(img, image_name):
	cv2.imwrite(OUTPUT_FOLDER + image_name,img)


def image_feature_and_homography():

	#### Mark Keypoints ####

	mountain_1_img = cv2.imread(SOURCE_FOLDER + "mountain1.jpg", 0)
	mountain_2_img = cv2.imread(SOURCE_FOLDER + "mountain2.jpg", 0)

	kp_mountain_1_img = np.zeros(mountain_1_img.shape)


	sift = cv2.xfeatures2d.SIFT_create()
	
	kp_1, desc_1 = sift.detectAndCompute(mountain_1_img, None)
	kp_mountain_1_img = cv2.drawKeypoints(mountain_1_img, kp_1, kp_mountain_1_img)

	write_image(kp_mountain_1_img, 'task1_sift1.jpg')

	kp_mountain_2_img = np.zeros(mountain_2_img.shape)

	kp_2, desc_2 = sift.detectAndCompute(mountain_2_img, None)
	kp_mountain_2_img = cv2.drawKeypoints(mountain_2_img, kp_2, kp_mountain_2_img)

	write_image(kp_mountain_2_img, 'task1_sift2.jpg')



	### Draw matches using k nearest neighbors ##
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(desc_1,desc_2, k=2)

	# Apply ratio test
	good_matches = []
	for m,n in matches:
	    if m.distance < 0.75*n.distance:
	        good_matches.append(m)


	knn_matched_img = np.zeros(mountain_1_img.shape)
	knn_matched_img = cv2.drawMatches(mountain_1_img, kp_1, mountain_2_img, kp_2, good_matches, knn_matched_img, flags=2)

	write_image(knn_matched_img, 'task1_matches_knn.jpg')

	### Calculate H matrix ###

	src_pts = np.float32([ kp_1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
	dst_pts = np.float32([ kp_2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
	matchesMask = mask.ravel().tolist()

	print(M)

	draw_params = dict(matchColor = (0,255,0),
	           singlePointColor = None,
	           matchesMask = matchesMask,
	           flags = 2)


	task1_matches = cv2.drawMatches(mountain_1_img, kp_1, mountain_2_img, kp_2, good_matches, None, **draw_params)

	
	write_image(task1_matches, 'task1_matches.jpg')


	im_out = cv2.warpPerspective(mountain_1_img, M, (mountain_2_img.shape[1],mountain_2_img.shape[0]))
	task1_pano = cv2.addWeighted(im_out, 0.2, mountain_2_img, 0.8, 1)

	write_image(task1_pano, 'task1_pano.jpg')



	################

	# mountain_1_canvas = np.zeros([mountain_1_img.shape[0]*2,mountain_1_img.shape[1]*2])

	# x_offset= 300
	# y_offset=200
	# mountain_1_canvas[y_offset : y_offset + mountain_1_img.shape[0], x_offset : x_offset + mountain_1_img.shape[1]] = mountain_1_img

	# # write_image(mountain_1_canvas,'jjjhjh.jpg')

	# mountain_2_canvas = np.zeros([mountain_2_img.shape[0]*2,mountain_2_img.shape[1]*2])

	# x_offset= 500
	# y_offset=200
	# mountain_2_canvas[y_offset : y_offset + mountain_2_img.shape[0], x_offset : x_offset + mountain_2_img.shape[1]] = mountain_2_img


	# im_out = cv2.warpPerspective(mountain_1_canvas, M, (mountain_2_canvas.shape[1],mountain_2_canvas.shape[0]))
	# task1_pano_ddfdf = cv2.addWeighted(im_out, 1, mountain_2_canvas, 1, 1)

	# write_image(task1_pano_ddfdf, 'task1_panosafsafsd.jpg')






def main():
	image_feature_and_homography()
	print('Done!!!!')

main()