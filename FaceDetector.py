import cv2
import cv2.face
import numpy as np
import os
import math
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)

    def detect(self, image, biggest_only=True):
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30, 30)
        biggest_only = True
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
            cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
            cv2.CASCADE_SCALE_IMAGE
        face_coord = self.classifier.detectMultiScale(image,
        scaleFactor = scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=flags)

        return face_coord
