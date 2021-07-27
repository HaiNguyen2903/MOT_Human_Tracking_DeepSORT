import os 
import time 
import argparse

parser = argparse.ArgumentParser(description='Get directions')
parser.add_argument('--pred_file', type=str, help='Path to prediction for postprocessing', required=True)
parser.add_argument('--mode', type=str, help='Filter boxes by rightmost or center point', default="center")
args = parser.parse_args()
    
def get_invalid_area(file="invalid_area.txt"):
	rects = []
	with open(file, "rt") as f:
		for line in f:
			x1, y1, x2, y2 = line.strip().split("_")
			rects.append([float(x1), float(y1), float(x2), float(y2)])
	return rects

def check_in_rect(point, rect):
	x1, y1, x2, y2 = rect
	if(point[0] > min(x1, x2) and point[0] < max(x1, x2) and point[1] > min(y1, y2) and point[1] < max(y1, y2)):
		return True
	return False

def get_center_point(rect):
	x1, y1, w, h = rect 
	c_x, c_y = x1 + w/2, y1 + h/2
	return c_x, c_y

def is_in_invalid_area(target_rect, rects):
	c_x, c_y = get_center_point(target_rect)
	for rect in rects:
		if check_in_rect((c_x, c_y), rect):
			return True 
	return False


def filter_boxes(pred_file, area_file):
	rects = get_invalid_area(area_file)
	pp_out = []
	with open(pred_file, "rt") as f:
		for line in f:
			p = line.strip().split(",")
			x1, y1, w, h = list(map(float, p[2:6]))
			if(is_in_invalid_area([x1, y1, w, h], rects)):
				print(x1, y1, w, h)
				continue
			pp_out.append(line)
	with open("pp_" + pred_file, "wt") as f:
		for line in pp_out:
			f.write(line + "\n")

if __name__ == '__main__':
	filter_boxes(args.pred_file, "invalid_area.txt")
