import cv2 as cv
import numpy as np


class Matcher:
	def __init__(self):
		self.ratio = 0.65
		self.refineF = True
		self.confidence = 0.99
		self.distance = 3.0
		self.detector = cv.xfeatures2d_SURF.create()
		self.fundamental_matrix = None

	def get_fundamental_matrix(self):
		return self.fundamental_matrix

	def set_detector(self, detector):
		self.detector = detector

	def set_confidence_level(self, confidence_level):
		self.confidence = confidence_level

	def set_min_distance_to_epipolar(self, distance):
		self.distance = distance

	def set_ratio(self, ratio):
		self.ratio = ratio

	def match(self, image1, image2):
		enough_matches = False

		kp1, desc1 = self.detector.detectAndCompute(image1, None)
		kp2, desc2 = self.detector.detectAndCompute(image2, None)

		# TODO try cv.DIST_L2 instead
		matcher = cv.BFMatcher(cv.NORM_L2)

		matches1 = matcher.knnMatch(desc1, desc2, k=2)
		matches2 = matcher.knnMatch(desc2, desc1, k=2)

		_, matches1 = self.ratio_test(matches1)
		_, matches2 = self.ratio_test(matches2)

		sym_matches = self.symmetry_test(matches1, matches2)
		matches = None

		if len(sym_matches) < 30:
			enough_matches = False
		else:
			enough_matches = True
			self.fundamental_matrix, matches = self.ransac_test(sym_matches, kp1, kp2)
			if len(matches) < 25:
				enough_matches = False

		return enough_matches, matches, kp1, kp2

	def ratio_test(self, matches):
		removed = 0
		for match in matches:
			if len(match) > 1:
				if match[0].distance/match[1].distance > self.ratio:
					match.clear()
					removed += 1
			else:
				match.clear()
				removed += 1

		return removed, matches

	def symmetry_test(self, matches1, matches2):
		symmetric_matches = []
		for match1 in matches1:
			if len(match1) < 2:
				continue
			for match2 in matches2:
				if len(match2) < 2:
					continue

				# Symmetry test
				if match1[0].queryIdx == match2[0].trainIdx and match2[0].queryIdx == match1[0].trainIdx:
					symmetric_matches.append(cv.DMatch(match1[0].queryIdx, match1[0].trainIdx, match1[0].distance))
					break

		return symmetric_matches

	def ransac_test(self, matches, kp1, kp2):
		ret_matches = []
		points1, points2 = [], []
		for match in matches:
			points1.append(kp1[match.queryIdx].pt)
			points2.append(kp2[match.trainIdx].pt)

		fundamental, inliers = cv.findFundamentalMat(np.float32(points1), np.float32(points2), method=cv.FM_RANSAC, param1=self.distance, param2=self.confidence)

		for inlier, match in zip(inliers, matches):
			if inlier[0] > 0:
				ret_matches.append(match)

		if self.refineF:
			points1 = []
			points2 = []
			for match in ret_matches:
				points1.append(kp1[match.queryIdx].pt)
				points2.append(kp2[match.trainIdx].pt)

			if len(points1) > 0 and len(points2) > 0:
				fundamental, _ = cv.findFundamentalMat(np.float32(points1), np.float32(points2), method=cv.FM_8POINT)

		return fundamental, ret_matches
