"""
code by Xinan Zhang
for interview questions--Deep learning on traffic sign detection using public data.
"""
"""
This file is used for classifying outcome of the pre-train model into new classes
Input: json file
Output: new records will appear in gt.txt
"""
import os
import json
import cv2
import easygui

def get_GT(path):
	with open(path, 'r') as f:
		gt = f.readlines()
	return gt

def write_record(record):
	with open('gt.txt','a+') as f:
		f.write(record+'\n')

def read_json(file):
	frames = list()
	with open(file, 'r') as f:
		json_str = f.read()
		json_dict = json.loads(json_str)
		frames = json_dict['output']['frames']
	for frame in frames:
		signs = frame['signs']
		for sign in signs:
			if sign['class'] == 'pn':
				print(frame['frame_number'])
				img = cv2.imread(frame['frame_number'])
				coor = sign['coordinates']
				cv2.rectangle(img, (coor[0],coor[1]),(coor[0]+coor[2],coor[1]+coor[3]), (0, 0, 255), 2)
				cv2.imwrite('temp.png',img)
				choise = easygui.ccbox(msg='test', title='PN annotation', choices=('43 X', '44 \\'), image='temp.png')
				cla = '43' if choise else '44'
				new_record = frame['frame_number']+';'+str(coor[0])+';'+str(coor[1])+';'+str(coor[0]+coor[2])+';'+str(coor[1]+coor[3])+';'+cla
				write_record(new_record)
				print(new_record)

read_json('GTSDB.json')

