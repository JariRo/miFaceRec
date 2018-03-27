import cv2
import cv2.face
import numpy as np
import os
import math
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from IPython.display import clear_output
import argparse
import imutils
import time

from imutils.video import VideoStream
from FaceDetector import FaceDetector


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())



def runMenu():
    print "*" * 10 + "Menu" + "*" * 10

    print "1: Collect and Train models"
    print "2: Live Recognition"
    print "3: Exit"

    usrInput = raw_input("Select menu option: ")

    if usrInput == "1":
        collectImages()
    elif usrInput == "2":
        live_recognition()
    elif usrInput == "3":
        exit()
    else:
        print "Please select one of the options listed"
        runMenu()

def normalize_intensity(images):
    images_norm= []

    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_norm.append(image)
    return images_norm

def cut_faces(image, faces_coord):
    faces = []

    for (x,y,w,h) in faces_coord:
        w_rm = int(0.2 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])

    return faces

def resize(images, size=(50, 50)):
    images_norm = []

    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, interpolation = cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, interpolation = cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm

def normalize_faces(frame, faces_coord):
    faces = cut_faces(frame, faces_coord)
    faces = normalize_intensity(faces)
    #faces = resize(faces)
    return faces

def draw_rectangle(frame, faces_coord):
    for(x,y,w,h) in faces_coord:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (150, 150, 0), 8)

def train_models():
    images, labels, labels_dic = collect_dataset()

    rec_eig = cv2.face.EigenFaceRecognizer_create()
    #rec_eig.train(images, labels)

    rec_fisher = cv2.face.FisherFaceRecognizer_create()
    #rec_fisher.train(images, labels)

    rec_lbph = cv2.face.LBPHFaceRecognizer_create()
    rec_lbph.train(images, labels)

    print('Models Trained Successfully')
    return [rec_eig, rec_fisher, rec_lbph, labels_dic]

def make_prediction():
    themodels = train_models()
    #rec_eig = themodels[0]
    #fish = themodels[1]
    lbph = themodels[2]

    labels_dic = themodels[3]

    vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
    detector = FaceDetector("haarcascade_frontalface_default.xml")
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    faces_coord = detector.detect(frame)
    faces = normalize_faces(frame, faces_coord)
    face = faces[0]

    #prediction, confidence = rec_eig.predict(face)
    #print ('Eigen faces -> prediction: ' + labels_dic.get(prediction).capitalize() + " Confidence: " + str(round(confidence)))

    #prediction, confidence = fish.predict(face)
    #print ('Fisher Faces -> prediction: ' + labels_dic.get(prediction).capitalize() + " Confidence: " + str(round(confidence)))

    prediction, confidence = lbph.predict(face)
    print ('LBPH -> prediction: ' + labels_dic.get(prediction).capitalize() + " Confidence: " + str(round(confidence)))

def collectImages():
    folder = "people/" + raw_input('Person: ').lower()
    cv2.namedWindow('Frame', cv2.WINDOW_AUTOSIZE)

    if not os.path.exists(folder):
        detector = FaceDetector('haarcascade_frontalface_default.xml')
        vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
        time.sleep(2.0)
        os.makedirs(folder)
        counter = 0
        timer = 0
        while counter < 13:
            frame = vs.read()
            #frame = imutils.resize(frame, width=600)
            faces_coord = detector.detect(frame)
            if len(faces_coord) and timer % 700 == 50:
                faces = normalize_faces(frame, faces_coord)
                cv2.imwrite(folder + '/' + str(counter) + '.jpg', faces[0])
                print ("Images saved: " + str(counter))
                counter += 1
            draw_rectangle(frame, faces_coord)
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1) & 0xFF
            timer += 50

            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        vs.stop()
        runMenu()
    else:
        print("Name already taken")
        runMenu()


def collect_dataset():
    images = []
    labels = []
    labels_dic = {}

    people = [person for person in os.listdir('people/')]
    for i, person in enumerate(people):
        if not person == '.DS_Store':
            labels_dic[i] = person
            for image in os.listdir('people/' + person):
                images.append(cv2.imread('people/' + person + '/' + image, 0))
                labels.append(i)
    return(images, np.array(labels), labels_dic)

def live_recognition():
    models = train_models()
    lbph = models[2]
    labels_dic = models[3]
    threshold = 45
    detector = FaceDetector("haarcascade_frontalface_default.xml")
    vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
    time.sleep(2.0)

    while True:

        frame = vs.read()
        #frame = imutils.resize(frame, width=600)
        faces_coord = detector.detect(frame, True)

        if len(faces_coord) > 0:
            faces = normalize_faces(frame, faces_coord)
            for i, face in enumerate(faces):
                pred, conf = lbph.predict(face)
                print ("Prediction: " + labels_dic[pred].capitalize() + '\nConfidence: ' + str(round(conf)))

                if conf < threshold:
                    if conf < threshold:
                        cv2.putText(frame,
                                    labels_dic[pred].capitalize(),
                                    (faces_coord[i][0], faces_coord[i][1] - 10),
                                    cv2.FONT_HERSHEY_PLAIN,
                                    3,
                                    (66, 53, 243),
                                    2,
                                    cv2.LINE_AA)
                else:
                    cv2.putText(frame,
                        "Unknown",
                        (faces_coord[i][0], faces_coord[i][1]),
                        cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)

                draw_rectangle(frame, faces_coord)
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
    cv2.destroyAllWindows()
    vs.stop()
    runMenu()

runMenu()
