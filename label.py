"""
code by Xinan Zhang
for interview questions--Deep learning on traffic sign detection using public data.
"""
"""
This file is used for resizing the raw images for training and test
Input: raw gt.txt
Output: new gt for each image
"""
import os

root = os.getcwd()
dataset = root
test = dataset + '/test/'

def get_gt(filename):
	
	with open(root+'/'+filename,'r') as f:
		gt_lines = f.readlines()

	return gt_lines

def process_gt(filename):
	# <object-class> <x_center> <y_center> <width> <height>
	gt_lines = get_gt(filename)
	for gt_line in gt_lines:
		line = gt_line.strip()
		element = line.split(';')
		# x coordinate of top left corner, y coordinate of the top left corner, width and height
		#ImgNo#.ppm;#leftCol#;##topRow#;#rightCol#;#bottomRow#;#ClassID#
		x = int(element[1])
		y = int(element[2])
		w = int(element[3])-int(element[1])
		h = int(element[4])-int(element[2])
		txt_name = element[0].split('.')[0] + '.txt'
		cla = element[-1]
		label = ''
		for key in mapping.keys():
			if int(cla) in mapping[key]:
				label = key
				break
		if label != '':
			# if txt_name == '00340.txt':
			# 	print(gt_line)
			string = f'{label} {x} {y} {w} {h}\n'
			# print(string)
			create_txt(txt_name, string)
			# break

def create_txt(txt_name, string):
	image_index = int(txt_name.split('.')[0])
	with open(test+txt_name, 'a+') as f:
		f.write(string)

# create folders
if os.path.exists(test):
	print(test,' exists!\n')
else:
	os.makedirs(test)

# create mapping
mapping = {'RedRoundSign':[0,1,2,3,4,5,6,7,8,9,10,15,16],
		   'pg':[13],
		   'ps':[14],
		   'pne':[17],
		   'pn':[43,44] }

# create txt for each image
for i in range(900):
    file_name = '{:05d}'.format(i) + '.txt'
    with open(test+file_name, 'a+') as f:
    	pass
process_gt('gt.txt')
