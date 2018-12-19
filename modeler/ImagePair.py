import cv2 as cv
import numpy as np
from .Matcher import Matcher


class ImagePair:
	def __init__(self, image1, image2, debug_mode=False):
		self.image1 = image1
		self.image2 = image2
		self.debug_mode = debug_mode

		good_matcher = Matcher()
		good_matcher.set_confidence_level(0.98)
		good_matcher.set_min_distance_to_epipolar(1.0)
		good_matcher.set_ratio(0.65)
		surf = cv.xfeatures2d_SURF.create()
		good_matcher.set_detector(surf)

		self.kp1 = []
		self.kp2 = []
		self.colors = []
		self.fundamental_matrix = np.float32([])

		self.matches = []
		self.full_kp1 = []
		self.full_kp2 = []
		self.enough_matches = False

		self.enough_matches, self.matches, self.full_kp1, self.full_kp2 = good_matcher.match(image1, image2)
		self.fundamental_matrix = good_matcher.get_fundamental_matrix()

		for match in self.matches:
			self.kp1.append(self.full_kp1[match.queryIdx])
			self.kp2.append(self.full_kp2[match.trainIdx])

	def get_colors(self, image):
		print('Image dimensions: ', image.shape[1], ', ', image.shape[0])

		colors = []

		for kp in self.kp1:
			y = int(kp.pt[0] + 0.5)
			x = int(kp.pt[1] + 0.5)

			# TODO change (everywhere in project) the way to deal with colors and make it BGR not RGB or BRG
			blue = image[x, y, 0]
			green = image[x, y, 1]
			red = image[x, y, 2]

			colors.append((blue, green, red))
		self.colors = colors
		return colors

	def display_and_save_matches_image(self):
		out_img = 0
		out_img = cv.drawMatches(self.image1, self.full_kp1, self.image2, self.full_kp2, self.matches, outImg=out_img)
		cv.imwrite('output.jpg', out_img)

		if self.debug_mode:
			cv.imshow('Matches', out_img)
			cv.waitKey(0)
			cv.destroyAllWindows()

	def has_enough_matches(self):
		return self.enough_matches

	# Getters
	def get_fundamental_matrix(self):
		return self.fundamental_matrix

	def get_keypoints_image1(self):
		return self.kp1

	def get_keypoints_image2(self):
		return self.kp2

