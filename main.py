UBIT = 'amlangup'
import numpy as np
np.random.seed(sum([ord(c) for c in UBIT]))

import cv2
import matplotlib.pyplot as plt
import random

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
	pts1 = []
	pts2 = []
	for m,n in matches:
	    if m.distance < 0.75*n.distance:
	        good_matches.append(m)
	        pts2.append(kp_2[m.trainIdx].pt)
	        pts1.append(kp_1[m.queryIdx].pt)


	knn_matched_img = np.zeros(img_1.shape)
	knn_matched_img = cv2.drawMatches(img_1, kp_1, img_2, kp_2, good_matches, knn_matched_img, flags=2)
	return knn_matched_img, good_matches, pts1, pts2


def find_homography_and_match_images(good_matches, kp_1, kp_2, img_1, img_2):

	src_pts = np.float32([ kp_1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
	dst_pts = np.float32([ kp_2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
	matchesMask = mask.ravel().tolist()

	matchesMask_in = []
	good_matches_in = []
	for mm, gm in zip(matchesMask,good_matches):
		if mm == 1:
			matchesMask_in.append(mm)
			good_matches_in.append(gm)



	print(len(good_matches_in))
	print(len(matchesMask_in))

	
	# randIndx = np.random.randint(low=0, high=len(good_matches_in), size=10)
	# good_matches_in = good_matches_in[randIndx]
	# matchesMask_in = matchesMask_in[ran]

	# matchesMask_in = matchesMask_in[:10]
	# good_matches_in_10 = random.sample(range(len(good_matches_in)), 10)


	draw_params = dict(matchColor = (0,255,0),
	           singlePointColor = None,
	           matchesMask = matchesMask_in,
	           flags = 2)


	matches_img = cv2.drawMatches(img_1, kp_1, img_2, kp_2, good_matches_in_10, None, **draw_params)
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
	
	knn_matched_img, good_matches, _, _ = find_knn_match(mountain_1_img, mountain_2_img, kp_1, kp_2, desc_1, desc_2)
	write_image(knn_matched_img, 'task1_matches_knn.jpg')

	
	#task 1.3, task 1.4
	
	h_matrix, task1_matches = find_homography_and_match_images(good_matches, kp_1, kp_2, mountain_1_img, mountain_2_img)
	# print(h_matrix)
	write_image(task1_matches, 'task1_matches.jpg')

	# task 1.5
	panorama = do_image_stitching(mountain_1_img, mountain_2_img, h_matrix)
	write_image(panorama,'task1_pano.jpg')


def drawlines(img1,img2,lines,pts1,pts2, color):
	r,c = img1.shape
	img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
	img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
	for r,pt1,pt2,colr in zip(lines,pts1,pts2,color):
		x0,y0 = map(int, [0, -r[2]/r[1] ])
		x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
		img1 = cv2.line(img1, (x0,y0), (x1,y1), tuple(colr),1)
		img1 = cv2.circle(img1,tuple(pt1),5,tuple(colr),-1)
		img2 = cv2.circle(img2,tuple(pt2),5,tuple(colr),-1)
	return img1,img2


def epipolar_geometry(tsucuba_left_img, tsucuba_right_img):

	# task 2.1

	kp_tsucuba_left_img, kp_1, desc_1 = perform_sift(tsucuba_left_img)
	write_image(kp_tsucuba_left_img, 'task2_sift1.jpg')

	kp_tsucuba_right_img, kp_2, desc_2 = perform_sift(tsucuba_right_img)
	write_image(kp_tsucuba_right_img, 'task2_sift2.jpg')
	
	# task 2.2, 2.3

	knn_matched_img, good_matches, pts1, pts2 = find_knn_match(tsucuba_left_img, tsucuba_right_img, kp_1, kp_2, desc_1, desc_2)
	
	pts1 = np.int32(pts1)
	pts2 = np.int32(pts2)
	F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
	pts1 = pts1[mask.ravel()==1]
	pts2 = pts2[mask.ravel()==1]
	print(F)


	randIndx = np.random.randint(low=0, high=pts1.shape[0], size=10)
	pts1 = pts1[randIndx]
	pts2 = pts2[randIndx]

	# color = tuple(np.random.randint(0,255,3).tolist())
	

	color = np.random.randint(0,255, size=(10, 3)).tolist()

	lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
	lines1 = lines1.reshape(-1,3)
	tsucuba_left_ep , _ = drawlines(tsucuba_left_img, tsucuba_right_img, lines1, pts1, pts2, color)

	write_image(tsucuba_left_ep, 'task2_epi_left.jpg')

	lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F)
	lines2 = lines2.reshape(-1,3)
	tsucuba_right_ep, _ = drawlines(tsucuba_right_img,tsucuba_left_img,lines2, pts2, pts1, color)

	write_image(tsucuba_right_ep, 'task2_epi_right.jpg')

	






def main():

	mountain_1_img = cv2.imread(SOURCE_FOLDER + "mountain1.jpg", 0)
	mountain_2_img = cv2.imread(SOURCE_FOLDER + "mountain2.jpg", 0)
	
	image_feature_and_homography(mountain_1_img, mountain_2_img)

	tsucuba_left_img = cv2.imread(SOURCE_FOLDER + "tsucuba_left.png", 0)
	tsucuba_right_img = cv2.imread(SOURCE_FOLDER + "tsucuba_right.png", 0)

	epipolar_geometry(tsucuba_left_img, tsucuba_right_img)
	



main()