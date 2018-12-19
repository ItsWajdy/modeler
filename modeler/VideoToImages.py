import cv2 as cv
import os


class VideoToImages:
	def __init__(self, video_path, images_dir, sampling_rate, debug_mode=False):
		self.video_path = video_path
		self.images_dir = images_dir
		self.sampling_rate = sampling_rate
		self.debug_mode = debug_mode

		try:
			os.mkdir(self.images_dir)
		except FileExistsError:
			pass

	def convert(self):
		print('Converting video to images...\n')
		cap = cv.VideoCapture(self.video_path)
		counter = 0
		name_counter = 0

		while True:
			ret, frame = cap.read()
			if not ret:
				break

			if counter % self.sampling_rate == 0:
				if self.debug_mode:
					print('Writing image #' + str(name_counter))
				cv.imwrite(self.images_dir + 'im' + str(name_counter) + '.jpg', frame)
				name_counter += 1

			counter += 1

		cap.release()

		print('Done converting!\n\n')
		return name_counter
