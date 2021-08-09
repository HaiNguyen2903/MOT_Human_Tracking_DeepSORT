import cv2
 
 
cap = cv2.VideoCapture('/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/track_dataset/NVR-CH01_S20210608-084648_E20210608-084709.mp4')
 
 
#check if the video capture is open
if(cap.isOpened() == False):
    print("Error Opening Video Stream Or File")
 
 
while(cap.isOpened()):
    ret, frame =cap.read()
 
    if ret == True:
        cv2.imshow('frame', frame)
 
        if cv2.waitKey(25)  == ord('q'):
            break
 
    else:
        break
 
 
cap.release()