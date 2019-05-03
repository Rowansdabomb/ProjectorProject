# import the necessary packages
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import argparse
import imutils
import time
import cv2
from helpers import initializeColor, captureColor, detectFaces
 
####################### HANDLE ARGUMENTS ##########################

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

###################################################################

################### OBJECT DETECTION SETUP ########################

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

####################################################################

########################### VIDEO SETUP ############################

camera = PiCamera()
resolution = (320, 240)
camera.resolution = resolution
rawCapture = PiRGBArray(camera, size=resolution)
stream = camera.capture_continuous(rawCapture, format="bgr", use_video_port=True)

# allow the camera or video file to warm up
time.sleep(2.0)
fps = FPS().start()

####################################################################

########################## GLOBAL VARIABLES ########################

initialized = False
mean = None

frame = None
overlay = None
personBox = None
faceMask = None

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
colorLower = (29, 86, 6)
colorUpper = (64, 255, 255)
screenWidth = 900

####################################################################

# loop over some frames
for (i, f) in enumerate(stream):
  # grab the frame from the stream and resize it to have a maximum
  # width of 400 pixels
  frame = f.array
  frame = imutils.resize(frame, width=400)

##  faceMask = detectFaces(net, frame, args["confidence"])

  if not initialized:
    mean, frame = initializeColor(mean, frame)

  # blur the frame, and convert it to the HSV
  # color space
  blurred = cv2.GaussianBlur(frame, (11, 11), 0)
  hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
  # construct a mask for the color, then perform
  # a series of dilations and erosions to remove any small
  # blobs left in the mask
  mask = cv2.inRange(hsv, colorLower, colorUpper)
  mask = cv2.erode(mask, None, iterations=2)
  mask = cv2.dilate(mask, None, iterations=2)

  # find contours in the mask and initialize the current
  # (x, y) center of the ball
  cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)

  center = None
  radius = 0

  # only proceed if at least one contour was found
  if len(cnts) > 0:
    # find the largest contour in the mask, then use
    # it to compute the minimum enclosing circle and
    # centroid
    c = max(cnts, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    # only proceed if the radius is larger than 1/200 of the screen width
    if radius > screenWidth / 100:
      # draw the circle and centroid on the frame and overlayMask,
      # then update the list of tracked points
      currentMask = np.ones((frame.shape[0], frame.shape[1]), np.uint8)
      cv2.circle(currentMask, (int(x), int(y)), int(radius), 255, -1)

  if initialized:
    if overlay is None:
      overlay = mask
    elif currentMask is not None:
      # create a historical mask
      overlay = cv2.bitwise_or(overlay, currentMask)

    ##    temp = cv2.bitwise_or(cv2.bitwise_not(overlay), faceMask)
    temp = cv2.bitwise_not(overlay)
    final = cv2.bitwise_not(cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR))
    frame = cv2.addWeighted(final, 1, frame, 1, -255)

  cv2.imshow("Frame", frame)
  key = cv2.waitKey(1) & 0xFF

  if key == ord("q"):
    break
  if key == ord("r"):
    init_second = 0
    overlay = None
    initialized = False
  if key == ord("s"):
    mean, colorUpper, colorLower = captureColor(mean, colorUpper, colorLower)
    initialized = True

  # clear the stream in preparation for the next frame and update
  # the FPS counter
  rawCapture.truncate(0)
  fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# close all windows
cv2.destroyAllWindows()

stream.close()
rawCapture.close()
camera.close()
