# Hand Gesture Recognition
#  
## Description
#### Recognize the hand gestures that correspond to the digits 0-5.
#  
## Algorithm
#### 1. Face detection
* Find the bounding box around the user's face (Viola-Jones algorithm).
* Calculate the bounding box's histogram of the Cr channel (after converting the image to the YCrCb colorspace), its size and position. It is assumed that a fair amount of the face's skin is exposed.
#### 2. Skin based hand detection
* Use a sliding window at the space on the right of the face to find the window with the most similar histogram to that of the face's.
* It is assumed that during that phase the user will raise their right hand, preferably with the fingers extended and not too spread apart. The best matched window should contain the right palm.
#### 3. Hand tracking
* The hand's window Cr channel is thresholded and a mask of the hand is obtained, which is used to calculate a more specific histogram for the hand.
* The hand is tracked from that point using the Mean Shift algorithm.
#### 4. Gesture recognition
* Recognition is based on observing the convexity defects of the hand's mask and counting the points of it's contour that form an angle less than 90&deg; with the point before and after it. This formation represents the triangle formed by the tips of two consecutive fingers and the point in the middle of the knuckles. When there is no point far enough from the hand's centroid, the hand is assumed to be closed (the corresponding gesture is 0).
#
## How to use
Follow the instructions that appear on the camera's window. After your face is detected, raise your raise your right hand to initiate the gesture recognition. For better results, try to keep the palm not too far away from the camera and the fingers inside the tracking window.
#
## Required compilers/libraries
* Python 2.7
* OpenCV 3.0
* Numpy
