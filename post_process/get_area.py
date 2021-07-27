import cv2 
from collections import defaultdict
import argparse

refPt = []
cropping = False
def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get directions')
    parser.add_argument('--image_path', type=str, help='Path to image', required=True)
    args = parser.parse_args()
    
    # print("You can mark {} points".format(args.num_points))
    
    # coordinate of points you have marked

    vidcap = cv2.VideoCapture(args.image_path)        
                
    cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("image", click_and_crop)

    success,image = vidcap.read()        

    if(success):
        clone = image.copy()
    
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
            refPt = []
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break

    if(len(refPt) % 2 == 0):
        with open("invalid_area.txt", "wt") as f:
            for i in range(len(refPt) - 1):
                f.write(str(refPt[i][0]) + "_" +str(refPt[i][1]) + "_" + str(refPt[i+1][0]) + "_" + str(refPt[i+1][1]) + "\n")



    #vidcap.released()
