import cv2
from IPython import embed

vid_path = '/data.local/hangd/data_vtx/testing/NVR-CH07_S20210609-084936_E20210609-085153.mp4'

# Python program to save a
# video using OpenCV

# Create an object to read
# from camera
video = cv2.VideoCapture(vid_path)

# We need to check if camera
# is opened previously or not
if (video.isOpened() == False):
	print("Error reading video file")

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(video.get(3))
frame_height = int(video.get(4))

size = (frame_width, frame_height)


# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
# embed(header='before write')
result = cv2.VideoWriter('filename.mp4',
						cv2.VideoWriter_fourcc(*'mp4v'),
						10, size)


# embed(header='after write')

while(True):

    ret, frame = video.read()
    # embed()

    if ret == True:

        # Write the frame into the
        # file 'filename.avi'
        # embed(header='before write frame')


        result.write(frame)
        print(frame.shape)
        exit()

        # embed(header='before after frame')
        # Display the frame
        # saved in the file
        # cv2.imshow('Frame', frame)
        # embed(header='debug imshow')

        # Press S on keyboard
        # to stop the process

        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture and video
# write objects
video.release()
result.release()
	
# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")
