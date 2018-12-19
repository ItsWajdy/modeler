ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


class PLY_Manip:
	def __init__(self, results_dir):
		self.dir = results_dir

	def insert_header(self, point_cloud_size, index):
		number = str(index)
		name = self.dir + 'out' + number + '.ply'

		with open(name, 'wb') as file:
			file.write((ply_header % dict(vert_num=point_cloud_size+1)).encode('utf-8'))
			file.write('0 0 0 255 0 0\n'.encode('utf-8'))

	def insert_point(self, x, y, z, b, g, r, index):
		number = str(index)
		name = self.dir + 'out' + number + '.ply'

		with open(name, 'ab') as file:
			file.write((str(x[0]) + ' ').encode('utf-8'))
			file.write((str(y[0]) + ' ').encode('utf-8'))
			file.write((str(z[0]) + ' ').encode('utf-8'))
			file.write((str(b) + ' ').encode('utf-8'))
			file.write((str(g) + ' ').encode('utf-8'))
			file.write((str(r) + '\n').encode('utf-8'))
