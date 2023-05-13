import cv2
import numpy as np


class ColorDescriptor:
    def __init__(self, bins):
        self.bins = bins

    def describe(self, image):
        # 이미지를 HSV 색 공간으로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []

        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))

        # top-left, top-right, bottom-right, bottom-left, corner
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]

        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipseMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipseMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        for(startX, endX, startY, endY) in segments:
            # construct a mask for each corner
            corner_mask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(corner_mask, (startX, startY), (endX, endY), 255, -1)
            corner_mask = cv2.subtract(corner_mask, ellipseMask)

            histogram = self.histogram(image, corner_mask)

            features.extend(histogram)

        histogram = self.histogram(image, ellipseMask)
        features.extend(histogram)
        return features

    def histogram(self, image, mask):
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        return hist

