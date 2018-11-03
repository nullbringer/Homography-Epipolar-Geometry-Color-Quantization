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


def perform_sift(img):

	sift_img = np.zeros(img.shape)


	sift = cv2.xfeatures2d.SIFT_create()
	
	kp, desc = sift.detectAndCompute(img, None)
	sift_img = cv2.drawKeypoints(img, kp, sift_img)

	return sift_img, kp, desc

def find_knn_match(img_1, img_2, kp_1, kp_2, desc_1, desc_2):

	bf = cv2.BFMatcher()
	matches = bf.knnMatch(desc_1,desc_2, k=2)

	# Apply ratio test
	good_matches = []
	for m,n in matches:
	    if m.distance < 0.75*n.distance:
	        good_matches.append(m)


	knn_matched_img = np.zeros(img_1.shape)
	knn_matched_img = cv2.drawMatches(img_1, kp_1, img_2, kp_2, good_matches, knn_matched_img, flags=2)
	return knn_matched_img, good_matches


def find_homography_and_match_images(good_matches, kp_1, kp_2, img_1, img_2):

	src_pts = np.float32([ kp_1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
	dst_pts = np.float32([ kp_2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
	matchesMask = mask.ravel().tolist()

	draw_params = dict(matchColor = (0,255,0),
	           singlePointColor = None,
	           matchesMask = matchesMask,
	           flags = 2)


	matches_img = cv2.drawMatches(img_1, kp_1, img_2, kp_2, good_matches, None, **draw_params)
	return M, matches_img



def do_image_stitching(img_1, img_2, M):

	(h1, w1) = img_1.shape[:2]
	(h2, w2) = img_2.shape[:2]

	#remap the coordinates of the projected image onto the panorama image space
	top_left = np.dot(M,np.asarray([0,0,1]))
	top_right = np.dot(M,np.asarray([w2,0,1]))
	bottom_left = np.dot(M,np.asarray([0,h2,1]))
	bottom_right = np.dot(M,np.asarray([w2,h2,1]))

	#normalize
	top_left = top_left/top_left[2]
	top_right = top_right/top_right[2]
	bottom_left = bottom_left/bottom_left[2]
	bottom_right = bottom_right/bottom_right[2]

	

	pano_left = int(min(top_left[0], bottom_left[0], 0))
	pano_right = int(max(top_right[0], bottom_right[0], w1))
	W = pano_right - pano_left

	pano_top = int(min(top_left[1], top_right[1], 0))
	pano_bottom = int(max(bottom_left[1], bottom_right[1], h1))
	H = pano_bottom - pano_top

	size = (W, H)

	# offset of first image relative to panorama
	X = int(min(top_left[0], bottom_left[0], 0))
	Y = int(min(top_left[1], top_right[1], 0))
	offset = (-X, -Y)

	panorama = np.zeros((size[1], size[0]), np.uint8)

	(ox, oy) = offset

	translation = np.matrix([
					[1.0, 0.0, ox],
					[0, 1.0, oy],
					[0.0, 0.0, 1.0]
					])


	M = translation * M

	cv2.warpPerspective(img_1, M, size, panorama)

	panorama[oy:h1+oy, ox:ox+w1] = img_2

	return panorama  






def image_feature_and_homography(mountain_1_img, mountain_2_img):

	# task 1.1

	kp_mountain_1_img, kp_1, desc_1 = perform_sift(mountain_1_img)
	write_image(kp_mountain_1_img, 'task1_sift1.jpg')

	
	kp_mountain_2_img, kp_2, desc_2 = perform_sift(mountain_2_img)
	write_image(kp_mountain_2_img, 'task1_sift2.jpg')


	# task 1.2
	
	knn_matched_img, good_matches = find_knn_match(mountain_1_img, mountain_2_img, kp_1, kp_2, desc_1, desc_2)
	write_image(knn_matched_img, 'task1_matches_knn.jpg')

	
	#task 1.3, task 1.4
	
	h_matrix, task1_matches = find_homography_and_match_images(good_matches, kp_1, kp_2, mountain_1_img, mountain_2_img)
	print(h_matrix)
	write_image(task1_matches, 'task1_matches.jpg')

	# task 1.5
	panorama = do_image_stitching(mountain_1_img, mountain_2_img, h_matrix)
	write_image(panorama,'task1_pano.jpg')


def epipolar_geometry(tsucuba_left_img, tsucuba_right_img):

	# task 2.1

	kp_tsucuba_left_img, kp_1, desc_1 = perform_sift(tsucuba_left_img)
	write_image(kp_tsucuba_left_img, 'task2_sift1.jpg')

	kp_tsucuba_right_img, kp_2, desc_2 = perform_sift(tsucuba_right_img)
	write_image(kp_tsucuba_right_img, 'task2_sift2.jpg')
	
	knn_matched_img, good_matches = find_knn_match(tsucuba_left_img, tsucuba_right_img, kp_1, kp_2, desc_1, desc_2)
	write_image(knn_matched_img, 'task2_matches_knn.jpg')





def main():

	mountain_1_img = cv2.imread(SOURCE_FOLDER + "mountain1.jpg", 0)
	mountain_2_img = cv2.imread(SOURCE_FOLDER + "mountain2.jpg", 0)
	
	image_feature_and_homography(mountain_1_img, mountain_2_img)

	tsucuba_left_img = cv2.imread(SOURCE_FOLDER + "tsucuba_left.png", 0)
	tsucuba_right_img = cv2.imread(SOURCE_FOLDER + "tsucuba_right.png", 0)

	epipolar_geometry(tsucuba_left_img, tsucuba_right_img)
	



main()