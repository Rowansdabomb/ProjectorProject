def initializeColor(mean, frame):
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

  return mean, frame

# takes in the global mean color, upper color and lower color threshold values
def captureColor(mean, colorUpper, colorLower):
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

  return mean, colorUpper, colorLower

def detectFaces(net):
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
  
  return faceMask