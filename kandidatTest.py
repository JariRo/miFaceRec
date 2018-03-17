import cv2
import cv2.face
import numpy as np
import os
import math
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from IPython.display import clear_output

#Classes
from FaceDetector import FaceDetector
from VideoCamera import VideoCamera



def cut_faces(image, faces_coord):
    faces = []

    for (x,y,w,h) in faces_coord:
        w_rm = int(0.2 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])

    return faces

def normalize_intensity(images):
    images_norm= []

    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_norm.append(cv2.equalizeHist(image))
    return images_norm

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


def collectImages():
    folder = "people/" + input('Person: ').lower() #Input Your name
    cv2.namedWindow('Frame', cv2.WINDOW_AUTOSIZE)
    cap = VideoCamera()
    detector = FaceDetector('haarcascade_frontalface_default.xml')

    if not os.path.exists(folder):
        os.makedirs(folder)
        counter = 0
        timer = 0

        while counter < 10:
            try:
                frame = cap.get_frame()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

                faces_coord = detector.detect(rgb)
                if len(faces_coord) and timer % 700 == 50:
                    faces = normalize_faces(frame, faces_coord)
                    cv2.imwrite(folder + "/" + str(counter) + ".jpg", faces[0])
                    print ("Images saved: " + str(counter))
                    counter += 1

                draw_rectangle(frame, faces_coord)
                cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                cv2.imshow('frame', frame)
                cv2.waitKey(50)
                timer += 50
            except KeyboardInterrupt:
                cv2.destroyAllWindows()
                print ("Live Video interrupted")
                break
            cv2.destroyAllWindows()
        else:
            cv2.destroyAllWindows()
            print ("This name already taken")


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
    eig = themodels[0]
    fish = themodels[1]
    lbph = themodels[2]

    labels_dic = themodels[3]

    webcam = VideoCamera(0)
    detector = FaceDetector("haarcascade_frontalface_default.xml")
    frame = webcam.get_frame()
    faces_coord = detector.detect(frame)
    faces = normalize_faces(frame, faces_coord)
    face = faces[0]

    plt.imshow(face)
    plt.show()
    del webcam

    #collector = cv2.face.MinDistancePredictCollection()

    '''
    rec_eig.predict(face, collector)
    conf = collector.getDist()
    pred = collector.getLabel()
    '''
    prediction, confidence = eig.predict(face)
    print ('Eigen faces -> prediction: ' + labels_dic.get(prediction).capitalize() + " Confidence: " + str(round(confidence)))

    '''
    rec_fisher.predict(face, collector)
    conf = collector.getDist()
    pred = collector.getLabel()
    '''
    prediction, confidence = fish.predict(face)
    print ('Fisher Faces -> prediction: ' + labels_dic.get(prediction).capitalize() + " Confidence: " + str(round(confidence)))

    '''
    rec_lbph.predict(face, collector)
    conf = collector.getDist()
    pred = collector.getLabel()
    '''
    prediction, confidence = lbph.predict(face)
    print ('LBPH -> prediction: ' + labels_dic.get(prediction).capitalize() + " Confidence: " + str(round(confidence)))


def live_recognition():
    detector = FaceDetector("haarcascade_frontalface_default.xml")
    webcame = VideoCamera(0)
    cv2.namedWindow('Frame', cv2.WINDOW_AUTOSIZE)
    models = train_models()

    lbph = models[2]
    labels_dic = models[3]

    while True:
        frame = webcame.get_frame()
        faces_coord = detector.detect(frame, True) #detects more than 1 face

        if len(faces_coord) > 0:
            faces = normalize_faces(frame, faces_coord) #normalize
            for i, face in enumerate(faces): #for each detected face
                pred, conf = lbph.predict(face)
                threshold = 45
                print ("Prediction: " + labels_dic[pred].capitalize() + '\nConfidence: ' + str(round(conf)))
                clear_output(wait = True)

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
            cv2.imshow("Jari och Oliver", frame)
            if cv2.waitKey(40) & 0xFF == 27:
                cv2.destroyAllWindows()
                break
        else:
            cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2, cv2.LINE_AA)
            cv2.imshow("Jari och Oliver", frame)
            if cv2.waitKey(40) & 0xFF == 27:
                break

live_recognition()
#collectImages()
