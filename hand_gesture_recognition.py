import cv2
import numpy as np
import math


class HandGedtureRecognition:
    def __init__(self, cam_index=-1):
        self.cam_index = cam_index
        (self.opencv_ver, _, _) = cv2.__version__.split('.')
        self.frame = None
        self.frame_clone = None
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        self.mean_shift_term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3, 1)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.cam = cv2.VideoCapture(cam_index)
        self.str_el = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5, 5))
        self.face_hist = None
        self.face_rect = None
        self.hand_rect = None
        self.hand_hist = None
        self.hand_found = False

    def find_hand(self):
        # Loop over the sliding window.
        hist_distances = []
        window_rectangles = []
        for window_rect in self.sliding_window(
                self.frame.shape,
                step_size=32,
                face_rect=(self.face_rect[0], self.face_rect[1], self.face_rect[2], self.face_rect[3])
        ):
            win_x, win_y, win_w, win_h = window_rect
            window_ycrcb = self.frame_ycrcb[win_y: win_y + win_h, win_x: win_x + win_w]
            window_hist = cv2.calcHist(
                [window_ycrcb],
                channels=[1],
                mask=None,
                histSize=[128],
                ranges=[0, 256]
            )
            d = cv2.compareHist(self.face_hist, window_hist, cv2.HISTCMP_CORREL)
            if 0.6 < d < 0.9:
                hist_distances.append(d)
                window_rectangles.append(window_rect)
        if hist_distances:
            max_d, max_d_idx = max((val, idx) for (idx, val) in enumerate(hist_distances))
            self.hand_rect = window_rectangles[max_d_idx]
            self.hand_found = True

            # Threshold the window to get the hand's binary mask.
            hand_x, hand_y, hand_w, hand_h = self.hand_rect
            hand_x = int(0.6 * hand_x)
            hand_y = int(0.8 * hand_y)
            hand_h = int(1.6 * hand_h)
            hand_w = int(2 * hand_w)
            self.hand_rect = (hand_x, hand_y, hand_w, hand_h)
            hand_image = self.frame_ycrcb[hand_y: hand_y + hand_h, hand_x: hand_x + hand_w, 1]
            retval, hand_mask = cv2.threshold(hand_image, 0, 255, cv2.THRESH_OTSU)
            # Calculate hand's histogram.
            self.hand_hist = cv2.calcHist(
                [hand_image],
                channels=[0],
                mask=hand_mask,
                histSize=[128],
                ranges=[0, 256]
            )
        else:
            self.hand_found = False
            self.hand_rect = None
            self.hand_hist = None

    def angle_rad(self, v1, v2):
        return np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))

    def deg2rad(self, angle_deg):
        return angle_deg / 180.0 * np.pi

    def recognize_gesture(self):
        # Threshold the window to get the hand's binary mask.
        hand_x, hand_y, hand_w, hand_h = self.hand_rect
        hand_image = self.frame_ycrcb[hand_y: hand_y + hand_h, hand_x: hand_x + hand_w, 1]
        hand_image = cv2.GaussianBlur(hand_image, ksize=(5, 5), sigmaY=1, sigmaX=1)
        retval, hand_mask = cv2.threshold(hand_image, 0, 255, cv2.THRESH_OTSU)
        hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, self.str_el)
        hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_OPEN, self.str_el)
        hand_mask_in_frame = np.zeros(self.frame.shape[:2], np.uint8)
        hand_mask_in_frame[hand_y: hand_y + hand_h, hand_x: hand_x + hand_w] = hand_mask

        thresh_deg = 80.0
        # Convexity hull based gesture recognition.
        contours = None
        if self.opencv_ver == '3':
            contours_image, contours, contours_hierarchy = cv2.findContours(
                hand_mask_in_frame.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        elif self.opencv_ver == '2':
            contours, contours_hierarchy = cv2.findContours(
                hand_mask_in_frame.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Find the contour with max area.
        hand_contour = max(contours, key=lambda c: cv2.contourArea(c))
        hand_contour = np.array(hand_contour)
        # Convex hull and convexity defects.
        hull = cv2.convexHull(hand_contour, returnPoints=False)
        defects = cv2.convexityDefects(hand_contour, hull)
        # Calculate the centroid to center the result's visualization and detect 0 gesture (closed fist).
        centroid = (hand_x + int(0.5 * hand_w), hand_y + int(0.6 * hand_h))
        valid_angles_count = 0
        possibly_zero = True
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(hand_contour[s][0])
            end = tuple(hand_contour[e][0])
            far = tuple(hand_contour[f][0])
            dist = math.sqrt((centroid[0] - far[0]) ** 2 + (centroid[1] - far[1]) ** 2)
            # Usually in cases of closed fist.
            if far[1] < centroid[1] and dist > 0.4 * hand_h:
                possibly_zero = False
            # Point
            if self.angle_rad(np.subtract(start, far), np.subtract(end, far)) < self.deg2rad(thresh_deg):
                valid_angles_count += 1
        gesture = min([5, valid_angles_count])
        if gesture <= 1 and possibly_zero:
            gesture = 0
        return hand_contour, defects, tuple(centroid), gesture

    def sliding_window(self, frame_shape, step_size, face_rect):
        (face_x, face_y, face_w, face_h) = face_rect
        # Slide a window across the image.
        for y in range(face_y - int(face_h / 2), frame_shape[0] - int(1.5 * face_h), step_size):
            for x in range(face_x + int(2 * face_w), frame_shape[1] - face_w / 2, step_size):
                # Yield the current window.
                window_rect = [x, y, face_w, face_h]
                yield (window_rect)

    def find_faces(self):
        # Face detection.
        faces = self.face_cascade.detectMultiScale(self.frame, 1.3, 5)
        if len(faces) == 1:
            for (x, y, w, h) in faces:
                x += w / 6
                y -= h / 10
                h += h / 10
                w -= w / 3
                face_mask = np.zeros(self.frame.shape[:2], np.uint8)
                face_mask[y: y + h, x: x + w] = 255
                self.face_hist = cv2.calcHist(
                    [self.frame_ycrcb],
                    channels=[1],
                    mask=face_mask,
                    histSize=[128],
                    ranges=[0, 256]
                )
                self.face_rect = [x, y, w, h]
        else:
            self.face_hist = None
            self.face_rect = None

    def run(self):
        key = 0
        # Press 'Esc' to quit.
        while key != 27:
            ret_val, self.frame = self.cam.read()
            if not ret_val:
                continue
            self.frame = cv2.flip(self.frame, 1)
            self.frame_ycrcb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2YCrCb)

            if not self.hand_found:
                self.find_faces()
                if self.face_hist is None or self.face_rect is None:
                    continue

                self.frame_clone = self.frame.copy()
                cv2.putText(
                    self.frame_clone,
                    'Face detected. Raise your right hand.',
                    (30, 30),
                    self.font,
                    1,
                    (255, 255, 255),
                    1
                )
                cv2.rectangle(
                    self.frame_clone,
                    (self.face_rect[0], self.face_rect[1]),
                    (self.face_rect[0] + self.face_rect[2], self.face_rect[1] + self.face_rect[3]),
                    (0, 255, 0),
                    2
                )
                cv2.imshow('Camera', self.frame_clone)
                key = cv2.waitKey(10)

                # Use the face's color, size and position to search for a window that contains the right hand.
                self.find_hand()
            else:
                # Hand tracking.
                dst = cv2.calcBackProject(
                    images=[self.frame_ycrcb],
                    channels=[1],
                    hist=self.hand_hist,
                    ranges=[0, 256],
                    scale=1
                )
                ret, self.hand_rect = cv2.meanShift(dst, tuple(self.hand_rect), self.mean_shift_term_crit)

                # Refine the hand's mask.
                hand_x, hand_y, hand_w, hand_h = self.hand_rect
                hand_y = int(0.9 * hand_y)
                self.hand_rect = [hand_x, hand_y, hand_w, hand_h]

                # Gesture recognition.
                hand_contour, defects, centroid, gesture = self.recognize_gesture()

                # Visualization.
                clone = self.frame.copy()
                cv2.putText(clone, 'The answer is in your hand ;)', (30, 30), self.font, 1, (255, 255, 255), 1)
                cv2.rectangle(
                    clone,
                    (hand_x, hand_y),
                    (hand_x + hand_w, hand_y + hand_h),
                    (255, 0, 0),
                    2
                )
                cv2.drawContours(clone, [hand_contour], 0, (0, 255, 0), 0)
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(hand_contour[s][0])
                    end = tuple(hand_contour[e][0])
                    far = tuple(hand_contour[f][0])
                    cv2.line(clone, start, end, [0, 255, 0], 2)
                    cv2.circle(clone, far, 5, [0, 0, 255], -1)
                cv2.putText(clone, str(gesture), centroid, self.font, 1, (255, 255, 255), 3)
                cv2.imshow('Camera', clone)
                key = cv2.waitKey(10)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    hand_gesture_recognition = HandGedtureRecognition(cam_index=1)
    hand_gesture_recognition.run()
