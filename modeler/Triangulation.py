import cv2 as cv
import numpy as np

from .SpacePoint import SpacePoint


class Triangulation:
	def __init__(self):
		self. K = np.float32([])

	def calculate_essential_matrix(self, F):
		return np.matmul(np.matmul(np.transpose(self.K), F), self.K)

	def decompose_E_to_R_and_T(self, E):
		# TODO recheck for correctness
		_, u, vt = cv.SVDecomp(E, flags=cv.SVD_MODIFY_A)

		W = np.float32([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
		Wt = np.float32([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

		R1 = np.matmul(np.matmul(u, W), vt)
		R2 = np.matmul(np.matmul(u, Wt), vt)
		t1 = u[:, 2]
		t2 = -u[:, 2]
		return R1, R2, t1, t2

	def find_matrix_K(self, image):
		px = image.shape[1] / 2
		py = image.shape[0] / 2

		self.K = np.float32([[1000, 0, px], [0, 1000, py], [0, 0, 1]])
		return self.K

	def find_camera_matrices(self, F):
		E = self.calculate_essential_matrix(F)
		R1, R2, t1, t2 = self.decompose_E_to_R_and_T(E)

		p_temp = np.float32([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
		p1 = p_temp.copy()

		p2 = np.float32([
			[R1[0, 0], R1[0, 1], R1[0, 2], t2[0]],
			[R1[1, 0], R1[1, 1], R1[1, 2], t2[1]],
			[R1[2, 0], R1[2, 1], R1[2, 2], t2[2]]
		])

		return p1, p2

	# TODO try cv.triangulatePoints instead
	def triangulate(self, kp1, kp2, K, p1, p2, point_cloud):
		inv_k = np.linalg.inv(K)

		temp_cloud = point_cloud.copy()

		for k1, k2 in zip(kp1, kp2):
			# TODO check if getting right x and y from kp (sometimes they need to be flipped)
			x = k1.pt[0]
			y = k1.pt[1]
			tmp = np.float32([[x], [y], [1]])
			mapping1 = np.matmul(inv_k, tmp)
			point3D1 = [mapping1[0], mapping1[1], mapping1[2]]

			x = k2.pt[0]
			y = k2.pt[1]
			tmp = np.float32([[x], [y], [1]])
			mapping2 = np.matmul(inv_k, tmp)
			point3D2 = [mapping2[0], mapping2[1], mapping2[2]]

			X = self.iterative_triangulation(point3D1, p1, point3D2, p2)

			temp_cloud.append(SpacePoint(x=X[0], y=X[1], z=X[2]))

		# point_cloud = temp_cloud.copy()
		return temp_cloud

	def linear_ls_triangulation(self, u, p, u1, p1):
		tmpA = [
			[u[0]*p[2, 0]-p[0, 0], u[0]*p[2, 1]-p[0, 1], u[0]*p[2, 2]-p[0, 2]],
			[u[1]*p[2, 0]-p[1, 0], u[1]*p[2, 1]-p[1, 1], u[1]*p[2, 2]-p[1, 2]],
			[u1[0]*p1[2, 0]-p1[0, 0], u1[0]*p1[2, 1]-p1[0, 1], u1[0]*p1[2, 2]-p1[0, 2]],
			[u1[1]*p1[2, 0]-p1[1, 0], u1[1]*p1[2, 1]-p1[1, 1], u1[1]*p1[2, 2]-p1[1, 2]]
		]
		A = np.float32(tmpA)
		A = np.reshape(A, [4, 3])

		B = np.float32([
			[-(u[0]*p[2, 3] - p[0, 3])],
			[-(u[1]*p[2, 3] - p[1, 3])],
			[-(u1[0]*p1[2, 3] - p1[0, 3])],
			[-(u1[1]*p1[2, 3] - p1[1, 3])]
		])
		B = np.reshape(B, [4, 1])

		X = np.zeros([])
		_, X = cv.solve(A, B, dst=X, flags=cv.DECOMP_SVD)
		return X

	def iterative_triangulation(self, u, p, u1, p1):
		wi = 1
		wi1 = 1
		X = np.zeros([4, 1], dtype=np.float32)

		iterations = 10

		for i in range(iterations):
			X_ = self.linear_ls_triangulation(u, p, u1, p1)
			X[0] = X_[0]
			X[1] = X_[1]
			X[2] = X_[2]
			X[3] = 1.0

			p2x = (np.matmul(np.float32(p)[2, :], X))[0]
			p2x1 = (np.matmul(np.float32(p1)[2, :], X))[0]

			wi = p2x
			wi1 = p2x1

			A = np.float32([
				[(u[0]*p[2, 0]-p[0, 0])/wi, (u[0]*p[2, 1]-p[0, 1])/wi, (u[0]*p[2, 2]-p[0, 2])/wi],
				[(u[1]*p[2, 0]-p[1, 0])/wi, (u[1]*p[2, 1]-p[1, 1])/wi, (u[1]*p[2, 2]-p[1, 2])/wi],
				[(u1[0]*p1[2, 0]-p1[0, 0])/wi1, (u1[0]*p1[2, 1]-p1[0, 1])/wi1, (u1[0]*p1[2, 2]-p1[0, 2])/wi1],
				[(u1[1]*p1[2, 0]-p1[1, 0])/wi1, (u1[1]*p1[2, 1]-p1[1, 1])/wi1, (u1[1]*p1[2, 2]-p1[1, 2])/wi1]
			])
			A = np.reshape(A, [4, 3])

			B = np.float32([
				[-(u[0]*p[2, 3] - p[0, 3])/wi],
				[-(u[1]*p[2, 3] - p[1, 3])/wi],
				[-(u1[0]*p1[2, 3] - p1[0, 3])/wi1],
				[-(u1[1]*p1[2, 3] - p1[1, 3])/wi1]
			])
			B = np.reshape(B, [4, 1])

			_, X_ = cv.solve(A, B, flags=cv.DECOMP_SVD)
			X[0] = X_[0]
			X[1] = X_[1]
			X[2] = X_[2]
			X[3] = 1.0

		return X
