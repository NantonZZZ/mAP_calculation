"""
code by Xinan Zhang
for interview questions--Deep learning on traffic sign detection using public data.
"""
"""
This file is used for calculating AP for each class
Input: gt file for each image; prediction results
Output: APs for each class; csv files
"""
import pandas as pd
import json
import copy
import cv2
import os

gt_dir = 'test/'
classes = ['RedRoundSign', 'pg', 'ps', 'pne', 'pn']
records = pd.DataFrame([],columns=['Image', 'Detection', 'Confidence', 'TP or FP'])

def get_cla_gt(gts, cla):
	cla_gts = list()
	for gt in gts:
		elements = gt.strip().split()
		# print(elements)
		if elements[0] == cla:
			cla_gts.append(gt)
	return cla_gts

def get_cla_sign(signs, cla):
	cla_sign = list()
	for sign in signs:
		if sign['class'] == cla:
			cla_sign.append(sign)
	return cla_sign

def overlap(x1, len1, x2, len2):

	left = max(x1, x2)
	right = min(x1 + len1, x2 + len2)

	return right - left

def box_intersection(a, b):
	w = overlap(a[0], a[2], b[0], b[2])
	h = overlap(a[1], a[3], b[1], b[3])
	if w < 0 or h < 0:
		return 0

	area = w * h
	return area

def box_union(a, b):
	i = box_intersection(a, b)
	u = a[2] * a[3] + b[2] * b[3] - i
	return u

def get_iou(a, b):
	a = [int(i) for i in a]
	b = [int(i) for i in b]
	return box_intersection(a, b) / box_union(a, b)

def tp_or_fp(sign, cla_gt):
	elements = cla_gt.strip().split()
	IoU = get_iou(sign['coordinates'], elements[1:])
	if IoU >= 0.5:
		# TP
		return True
	else:
		# FP
		return False

def add_record(img,det,conf,result):
	# ['Image', 'Detection', 'Confidence', 'TP or FP']
	global records
	new=pd.DataFrame({'Image':img,
				  'Detection':det,
				  'Confidence':conf,
				  'TP or FP':result},index=[1])
	records = records.append(new,ignore_index=True)

def cal_ap(total_gt):
	global records
	records['Acc TP'] = 0
	records['Acc FP'] = 0
	records['Precision'] = 0
	records['Recall'] = 0
	acc_tp = 0
	acc_fp = 0

	for i in range(len(records)):
		if records['TP or FP'][i] == 'TP':
			acc_tp += 1
		else:
			acc_fp += 1
		records.loc[i,'Acc TP'] = acc_tp
		records.loc[i,'Acc FP'] = acc_fp
		records.loc[i,'Precision'] = acc_tp/(acc_tp+acc_fp)
		records.loc[i,'Recall'] = acc_tp/total_gt
		# records['Acc TP'][i] = i

	pre = records['Precision']
	# print(pre)
	recallxpre = dict()
	index = 0
	local_max = 0
	while True:
		# index = 0
		for i in range(index,len(pre)):
			if pre[i]>=local_max:
				index = i
				local_max = pre[i]
		# print(index, local_max)
		if recallxpre.get(records['Recall'][index],0) == 0:
			recallxpre[records['Recall'][index]] = local_max
		index += 1
		local_max = 0
		if index == len(pre): break

	recallxpre[1] = recallxpre.get(1,0)
	recallxpre[0] = recallxpre[list(recallxpre.keys())[0]]
	# print(recallxpre)

	points = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	keys = list(recallxpre.keys())
	keys.sort()
	values = list()
	
	for point in points:
		current = 0
		for i in range(len(keys)-1,-1,-1):
			# print(i)
			if point <= keys[i]: current = recallxpre[keys[i]]
			else: break
		values.append(current)

	# print(values)
	return sum(values)/len(values)

def save_fp_img(cla, fp_list, raw_img):
	folder = 'FP&FN/'+cla+'_fp'
	if not os.path.exists(folder):
		os.makedirs(folder)
	for i,fp in enumerate(fp_list):
		# print(fp)
		img = cv2.imread(raw_img)
		coor = fp['coordinates']
		cv2.rectangle(img, (coor[0],coor[1]),(coor[0]+coor[2],coor[1]+coor[3]), (0, 0, 255), 2)
		cv2.putText(img, fp['class'],(coor[0],coor[1]-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2)
		cv2.imwrite(folder+'/'+raw_img[:-3]+'_'+str(i)+'.jpg',img)
	# print(fp_list)
	# assert 1==2
	# 39,219,90

def save_fn_img(cla, fn_list, raw_img):
	folder = 'FP&FN/'+cla+'_fn'
	if not os.path.exists(folder):
		os.makedirs(folder)
	for i,fn in enumerate(fn_list):
		f = fn.strip().split()
		coor = [int(f[j]) for j in range(1,len(f))]
		img = cv2.imread(raw_img)
		# print(f,coor)
		cv2.rectangle(img, (coor[0],coor[1]),(coor[0]+coor[2],coor[1]+coor[3]), (39, 219, 90), 2)
		cv2.putText(img, f[0],(coor[0],coor[1]-10),cv2.FONT_HERSHEY_SIMPLEX,1,(39, 219, 90),2)
		cv2.imwrite(folder+'/'+raw_img[:-3]+'_'+str(i)+'.jpg',img)

def get_class_records(cla, file):
	frames = list()
	total_gt = 0
	with open(file, 'r') as f:
		json_str = f.read()
		json_dict = json.loads(json_str)
		frames = json_dict['output']['frames']
	for frame in frames:
		gt_file = gt_dir + frame['frame_number'][:-3]+'txt'
		with open(gt_file,'r+') as f:
			gts = f.readlines()
		# print(gts)

		cla_gts = get_cla_gt(gts, cla)
		cla_signs = get_cla_sign(frame['signs'], cla)
		temp_gts = copy.deepcopy(cla_gts)
		temp_signs = copy.deepcopy(cla_signs)
		# print(temp_gts)
		# print(temp_signs)
		# ['RedRoundSign 1091 393 42 43\n']
		# [{'coordinates': [1096, 395, 32, 35], 'detection_confidence': 0.775431, 'class': 'RedRoundSign'}]

		total_gt += len(cla_gts)
		
		for i,sign in enumerate(cla_signs):
			for j,cla_gt in enumerate(cla_gts):
				result = tp_or_fp(sign, cla_gt)
				if result:
					# print(temp_signs.index(sign), temp_gts.index(cla_gt))
					temp_signs.pop(temp_signs.index(sign))
					temp_gts.pop(temp_gts.index(cla_gt))
					add_record(frame['frame_number'],cla,sign['detection_confidence'],'TP')
				# else:
				# 	add_record(frame['frame_number'],cla,sign['detection_confidence'],'FP')

		for sign in temp_signs:
			add_record(frame['frame_number'],cla,sign['detection_confidence'],'FP')
		if temp_signs != []:
			save_fp_img(cla,temp_signs,frame['frame_number'])
		if temp_gts != []:
			save_fn_img(cla,temp_gts,frame['frame_number'])
	global records
	records = records.sort_values(by = ['Confidence'], ascending = [False])
	records = records.reset_index(drop=True)
	# print(records)
	# print(total_gt)
	AP = cal_ap(total_gt)
	print(f'AP of {cla}: {AP} , total gt label: {total_gt}\n')

	records.to_csv(cla+'_records.csv')
	records = records.drop(index=records.index)
		
if __name__ == '__main__':
	for cla in classes:
		get_class_records(cla,'GTSDB.json')
