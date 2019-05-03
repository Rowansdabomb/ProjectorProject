# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
 
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

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()
 
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
 
# allow the camera or video file to warm up
time.sleep(.25)
fps = FPS().start()

####################################################################

########################## GLOBAL VARIABLES ########################

initialized = False
mean = None

font = cv2.FONT_HERSHEY_PLAIN
lineType = cv2.LINE_AA

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

def initializeColor():
  global mean
  global frame
  # left, top, right, bottom
  size = 75
  c1 = int((frame.shape[1] / 2 - size/2))
  r1 = int((frame.shape[0] / 2 - size/2))

  roi = frame[r1:c1, r1+size:c1+size]

  cv2.rectangle(frame, (c1, r1), (c1+size, r1 + size), (255, 255, 255), 1)
  mean = cv2.mean(roi)
  cv2.rectangle(frame, (frame.shape[1] - 200, frame.shape[0]-200), (frame.shape[1], frame.shape[0]), mean, -1)

  text = "Press 's' key to capture colour" 
  # get boundary of this text
  textsize = cv2.getTextSize(text, font, 1, 2)[0]

  # get coords based on boundary
  textX = (frame.shape[1] - textsize[0]) // 2
  textY = (size + 20 + frame.shape[0] + textsize[1]) // 2

  cv2.putText(frame, text, (textX, textY), font, 1, (255, 255, 255), 1, lineType)
  
def captureColor():
  global colorLower
  global colorUpper
  global mean
  
  rgb = np.uint8([[mean]])
  hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)[0][0]

  threshold = 10

  minHue = hsv[0] - threshold
  if (minHue < 0): minHue = 0
  minSat = hsv[1] - 3*threshold
  if (minSat < 0): minSat = 0
  minVal = hsv[2] - 3*threshold
  if (minVal < 0): minVal = 0
  
  maxHue = hsv[0] + threshold
  if (maxHue > 180): maxHue = 180
  maxSat = hsv[1] + 10*threshold
  if (maxSat > 255): maxSat = 255
  maxVal = hsv[2] + 10*threshold
  if (maxVal > 255): maxValue = 255

  colorLower = np.array([minHue, minSat, minVal])
  colorUpper = np.array([maxHue, maxSat, maxVal])
  time.sleep(1.0)

def detectFaces():
  global frame
  global net
  global faceMask
  # grab the frame dimensions and convert it to a blob
  (h, w) = frame.shape[:2]
  blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
    (300, 300), (104.0, 177.0, 123.0))

  # pass the blob through the network and obtain the detections and
  # predictions
  net.setInput(blob)
  detections = net.forward()

  # loop over the detections
  for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence < args["confidence"]:
      continue

    # compute the (x, y)-coordinates of the bounding box for the
    # object
    face = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = face.astype("int")
    center = (int(startX + (endX - startX)/2), int(startY + (endY - startY)/2))
    
    faceMask = np.ones((frame.shape[0], frame.shape[1]), np.uint8)
    # draw a circle around the persons head
    cv2.circle(faceMask, center, int((endX - startX) / 2), 255, -1)

# keep looping
while True:
  # grab the current frame
  frame = vs.read()

  # handle the frame from VideoCapture or VideoStream
  frame = frame[1] if args.get("video", False) else frame

  # if we are viewing a video and we did not grab a frame,
  # then we have reached the end of the video
  if frame is None:
    break

  frame = cv2.flip(frame, 1)
  # resize the frame
  frame = imutils.resize(frame, width=screenWidth)

  detectFaces()

  if not initialized:
    initializeColor()

  # blur the frame, and convert it to the HSV
  # color space
  blurred = cv2.GaussianBlur(frame, (11, 11), 0)
  hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
  # construct a mask for the color, then perform
  # a series of dilations and erosions to remove any small
  # blobs left in the mask
  mask = cv2.inRange(hsv, colorLower, colorUpper)
  cv2.imshow("inrange: ", mask)
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

    # # c2.imshow("mask", overlay)
    temp = cv2.bitwise_or(cv2.bitwise_not(overlay), faceMask)
    final = cv2.bitwise_not(cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR))
    cv2.imshow("overlay", final)
    frame = cv2.addWeighted(final, 1, frame, 1, -255)
    

  cv2.putText(frame, str(int(radius)), (10, 30), font, 2, (0, 0, 255), 2, lineType)
  # show the frame to our screen
  cv2.imshow("Frame", frame)
  key = cv2.waitKey(1) & 0xFF

  # if the 'q' key is pressed, stop the loop
  if key == ord("q"):
    break
  if key == ord("r"):
    init_second = 0
    overlay = None
    initialized = False
  if key == ord("s"):
    captureColor()
    initialized = True

  fps.update()
    
 
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()
 
# otherwise, release the camera
else:
	vs.release()
 
# close all windows
cv2.destroyAllWindows()