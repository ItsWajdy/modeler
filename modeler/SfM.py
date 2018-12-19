import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from .VideoToImages import VideoToImages as Converter
from .ImagePair import ImagePair
from .PLY_Manip import PLY_Manip
from .Triangulation import Triangulation
from .PointCloudTable import PointCloudTable


class SfM:
	def __init__(self, results_dir, video_already_converted, video_path=None, video_sampling_rate=None, debug_mode=False):
		self.images_dir = 'input_images/'
		self.results_dir = results_dir

		try:
			os.mkdir(self.results_dir)
		except FileExistsError:
			pass

		if video_already_converted:
			self.number_of_images = len(glob.glob1(self.images_dir, "*.jpg"))
		else:
			converter = Converter(video_path, self.images_dir, video_sampling_rate, debug_mode)
			self.number_of_images = converter.convert()

		self.debug_mode = debug_mode

		self.triangulation = Triangulation()
		self.ply = PLY_Manip(self.results_dir)

		self.table1 = PointCloudTable()
		self.table2 = PointCloudTable()

		self.table1.init()
		self.table2.init()

		self.current = self.table1.copy()
		self.prev = self.table2.copy()

	@staticmethod
	def downsample(image):
		max_rows = 1800
		max_cols = 1600
		modify_image = image.copy()
		height = modify_image.shape[0]
		width = modify_image.shape[1]

		if height % 2 != 0:
			height -= 1
		if width % 2 != 0:
			width -= 1

		down_size = modify_image.copy()

		while True:
			tmp_height = down_size.shape[0]
			tmp_width = down_size.shape[1]

			if tmp_height % 2 != 0:
				tmp_height -= 1
			if tmp_width % 2 != 0:
				tmp_width -= 1

			even_size = down_size[0:tmp_height, 0:tmp_width]
			down_size = cv.pyrDown(even_size, dst=down_size, dstsize=(int(tmp_width / 2), int(tmp_height / 2)))

			if tmp_width * tmp_height <= max_cols * max_rows:
				break

		return down_size

	@staticmethod
	def find_second_camera_matrix(p1, new_kp, old_kp, current, prev, K):
		found_points2D = []
		found_points3D = []

		for i in range(len(old_kp)):
			found = prev.find_3d(old_kp[i].pt)
			if found is not None:
				new_point = (found[0], found[1], found[2])
				new_point2 = (new_kp[i].pt[0], new_kp[i].pt[1])

				found_points3D.append(new_point)
				found_points2D.append(new_point2)
				current.add_entry(new_point, new_point2)

		print('Matches found in table: ' + str(len(found_points2D)))

		size = len(found_points3D)

		found3d_points = np.zeros([size, 3], dtype=np.float32)
		found2d_points = np.zeros([size, 2], dtype=np.float32)

		for i in range(size):
			found3d_points[i, 0] = found_points3D[i][0]
			found3d_points[i, 1] = found_points3D[i][1]
			found3d_points[i, 2] = found_points3D[i][2]

			found2d_points[i, 0] = found_points2D[i][0]
			found2d_points[i, 1] = found_points2D[i][1]

		p_tmp = p1.copy()

		r = np.float32(p_tmp[0:3, 0:3])
		t = np.float32(p_tmp[0:3, 3:4])

		r_rog, _ = cv.Rodrigues(r)

		_dc = np.float32([0, 0, 0, 0])

		_, r_rog, t = cv.solvePnP(found3d_points, found2d_points, K, _dc, useExtrinsicGuess=False)
		t1 = np.float32(t)

		R1, _ = cv.Rodrigues(r_rog)

		camera = np.float32([
			[R1[0, 0], R1[0, 1], R1[0, 2], t1[0]],
			[R1[1, 0], R1[1, 1], R1[1, 2], t1[1]],
			[R1[2, 0], R1[2, 1], R1[2, 2], t1[2]]
		])

		return camera

	def find_structure_from_motion(self):
		file_number = 0

		picture_number1 = 0
		picture_number2 = 1

		image_name1 = self.images_dir + 'im0.jpg'
		image_name2 = self.images_dir + 'im1.jpg'

		# TODO check what -1 means in OpenCV
		frame1 = cv.imread(image_name1)
		frame2 = cv.imread(image_name2)

		point_cloud = []
		p1 = np.zeros([3, 4], dtype=np.float32)
		p2 = np.zeros([3, 4], dtype=np.float32)

		prev_number_of_points_added = 0
		initial_3d_model = True

		factor = 1
		count = 0

		while file_number < self.number_of_images - 1:
			frame1 = SfM.downsample(frame1)
			frame2 = SfM.downsample(frame2)

			print('Using ' + str(image_name1) + ' and ' + str(image_name2))

			if self.debug_mode:
				plt.subplot('121')
				plt.imshow(frame1)

				plt.subplot('122')
				plt.imshow(frame2)
				plt.show()

			print('Matching...')

			robust_matcher = ImagePair(frame1, frame2)
			kp1 = robust_matcher.get_keypoints_image1()
			kp2 = robust_matcher.get_keypoints_image2()
			colors = robust_matcher.get_colors(frame1)

			if robust_matcher.has_enough_matches():
				robust_matcher.display_and_save_matches_image()
				print('Enough Matches!')

				K = self.triangulation.find_matrix_K(frame1)
				if initial_3d_model:
					print('Calculating initial camera matrices...')
					p1, p2 = self.triangulation.find_camera_matrices(robust_matcher.get_fundamental_matrix())

					print('Creating initial 3D model...')
					point_cloud = self.triangulation.triangulate(kp1, kp2, K, p1, p2, point_cloud)
					self.current.add_all_entries(kp2, point_cloud)

					if self.debug_mode:
						print('Initial lookup table size is: ' + str(self.current.table_size()))
					initial_3d_model = False
				else:
					self.prev.init()
					# TODO one might have to call .copy() on current to eliminate unwanted behaviours
					self.prev = self.current.copy()

					if self.current == self.table2:
						self.current = self.table1.copy()
					elif self.current == self.table1:
						self.current = self.table2.copy()

					if self.debug_mode:
						print('LookupTable size is: ' + str(self.prev.table_size()))
						print('New Table size is: ' + str(self.current.table_size()))

					p1 = p2.copy()
					p2 = SfM.find_second_camera_matrix(p2, kp2, kp1, self.current, self.prev, K)

					if self.debug_mode:
						print('New table size after adding known 3D points: ' + str(self.current.table_size()))

					print('Triangulating...')
					point_cloud = self.triangulation.triangulate(kp1, kp2, K, p1, p2, point_cloud)
					self.current.add_all_entries(kp2, point_cloud)

				number_of_points_added = len(kp1)

				print('Start writing points to file...')
				self.ply.insert_header(len(point_cloud), file_number)

				for i in range(prev_number_of_points_added):
					point = point_cloud[i]
					blue = point_cloud[i].b
					red = point_cloud[i].r
					green = point_cloud[i].g
					self.ply.insert_point(point.x, point.y, point.z, red, green, blue, file_number)

				for i in range(number_of_points_added):
					point = point_cloud[i + prev_number_of_points_added]
					point_color = colors[i]
					point_cloud[i + prev_number_of_points_added].b = point_color[0]
					point_cloud[i + prev_number_of_points_added].g = point_color[1]
					point_cloud[i + prev_number_of_points_added].r = point_color[2]

					self.ply.insert_point(point.x, point.y, point.z, point_color[0], point_color[1], point_color[2],
											 file_number)

				file_number += 1
				prev_number_of_points_added = number_of_points_added + prev_number_of_points_added

			else:
				print('Not enough matches')

			picture_number1 = picture_number2 % self.number_of_images
			picture_number2 = (picture_number2 + factor) % self.number_of_images

			count += 1
			if count % self.number_of_images == self.number_of_images - 1:
				picture_number2 += 1
				factor += 1

			image_name1 = self.images_dir + 'im' + str(picture_number1) + '.jpg'
			image_name2 = self.images_dir + 'im' + str(picture_number2) + '.jpg'
			frame1 = cv.imread(image_name1)
			frame2 = cv.imread(image_name2)
			
			print('\n\n')

		print('Done')

