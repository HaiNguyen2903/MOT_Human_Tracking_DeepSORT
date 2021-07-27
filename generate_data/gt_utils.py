import cv2 
from collections import defaultdict
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

class Human():
  def __init__(self, frame, track_id, coord):
    self.coord = list(map(float, coord))
    self.frame = int(frame)
    self.track_id = int(track_id)
  def xywh_to_xyxy(self):
    x, y, w, h = self.coord
    xmin, ymin, xmax, ymax = x, y, x+w, y+h 
    return (xmin, ymin, xmax, ymax)

def get_object_frame(file):
    with open(file, "rt") as f:
        lines = f.readlines()     
        
    lines = list(map(lambda x:x.strip().split(",")[:6], lines))

    humans = []
    for line in lines:
        humans.append(Human(line[0], line[1], line[2:]))

    group_frame = defaultdict(list)
    for v in humans:
        group_frame[v.frame].append(v)
    persons_inf = group_frame[1]
    bbox_gt_xyxy = list(map(lambda x: x.xywh_to_xyxy(), persons_inf))
    identities_gt = list(map(lambda x: x.track_id, persons_inf))
    print("bbox: ", bbox_gt_xyxy)
    print("id: ", identities_gt)
    print("gf: ", group_frame[1])
    return group_frame

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img
if __name__ == "__main__":
    get_object_frame("gt.txt")

    

