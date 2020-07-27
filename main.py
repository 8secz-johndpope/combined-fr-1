import logging
import os
import sys
sys.path.append('/root/')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

torch.backends.cudnn.bencmark = True

import os, sys, cv2, random, datetime
import pickle
import argparse
import numpy as np
import zipfile
from timeit import default_timer as timer
from PIL import Image
from itertools import cycle
from glob import glob
import youtube_dl

from sphereface.dataset import ImageDataset
from cp2tform import get_similarity_transform_for_cv2
import sphereface.net_sphere as net_sphere

# import DeepFace
# from deepface.basemodels import OpenFace, Facenet, FbDeepFace

from random import shuffle
#################################

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
from cv2 import VideoCapture
import os

#####################################

import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--url", default="https://www.youtube.com/watch?v=8uHmXEKltHo")#, required=True, help="Youtube url")
ap.add_argument("-d", "--detector", default="face_detection_model",#, required=True,
        help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", default="openface_nn4.small2.v1.t7",#, required=True,
        help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", default="output/recognizer.pickle",#, required=True,
        help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", default="output/le.pickle",#, required=True,
        help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.6,
        help="minimum probability to filter weak detections")


# construct the argument parser and parse the arguments
ap.add_argument("-e", "--encodings", default="embeddings.pickle",
    help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str, default="output/jurassic_park_trailer_output.avi",
    help="path to output video")
ap.add_argument("-y", "--display", type=int, default=0,
    help="whether or not to display output frame to screen")
ap.add_argument("-dm", "--detection-method", type=str, default="cnn",
    help="face detection model to use: either `hog` or `cnn`")

ap.add_argument("-dt", "--dt", type=str, default="opencv",
        help="Type of detector")
ap.add_argument(
    "--sphere_model", "-sm", default="sphereface/model/sphere20a_20171020.pth", type=str
)

args = vars(ap.parse_args())



net = getattr(net_sphere, "sphere20a")()
net.load_state_dict(torch.load(args["sphere_model"]))
net.cuda()
net.eval()
net.feature = True

def predict(face, embd):
    img = cv2.resize(face, (112, 96))
    imglist = [img, cv2.flip(img, 1), img, cv2.flip(img, 1)]
    for i in range(len(imglist)):
        imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1, 3, 112, 96))
        imglist[i] = (imglist[i] - 127.5) / 128.0

    img = np.vstack(imglist)
    img = Variable(torch.from_numpy(img).float(), volatile=True).cuda()
    output = net(img)
    f = output.data
    f1, f2 = f[0].cpu(), f[2]
    distance = i = 0
    for i, e in enumerate(embd):
        if distance < 1 - (f1.dot(e) / (f1.norm() * e.norm() + 1e-5)):
            distance = 1 - (f1.dot(e) / (f1.norm() * e.norm() + 1e-5))
            idx = i
    return idx, 1-distance


# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())


###########################################YOUTUBE
# create youtube-dl object
ydl = youtube_dl.YoutubeDL()

# set video url, extract video information
video_url = args["url"]
info_dict = ydl.extract_info(video_url, download=False)

# get video formats available
formats = info_dict.get('formats',None)

for f in formats:

    # I want the lowest resolution, so I set resolution as 144p
    if f.get('format_note',None) == '720p':

        #get the video url
        url = f.get('url',None)

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
cap = VideoCapture(url)
# vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
# fps = FPS().start()

# loop over frames from the video file stream
first_run = True
i = 0
##########################################################

if args["dt"]=="opencv":
    # USAGE
    # python recognize_video.py --detector face_detection_model \
    #        --embedding-model openface_nn4.small2.v1.t7 \
    #        --recognizer output/recognizer.pickle \
    #        --le output/le.pickle

    # import the necessary packages
    from imutils.video import VideoStream
    from imutils.video import FPS
    import numpy as np
    import argparse
    import imutils
    import pickle
    import time
    import cv2
    from cv2 import VideoCapture
    import os



    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"],
            "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    # print("[INFO] loading face recognizer...")
    # embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open(args["recognizer"], "rb").read())
    le = pickle.loads(open(args["le"], "rb").read())


    while True:
            # grab the frame from the threaded video stream
            ret, frame = cap.read()
            if not ret: break
            # frame = vs.read()

            i = not i
            if i: continue

            # resize the frame to have a width of 600 pixels (while
            # maintaining the aspect ratio), and then grab the image
            # dimensions
            # frame = imutils.resize(frame, width=600)
            (h, w) = frame.shape[:2]

            if first_run:
                    out = cv2.VideoWriter(args["output"],cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w,h))
                    first_run = False

            # construct a blob from the image
            imageBlob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                    (104.0, 177.0, 123.0), swapRB=False, crop=False)
            # imageBlob = cv2.dnn.blobFromImage(frame, mean=(104.0, 177.0, 123.0))

            # apply OpenCV's deep learning-based face detector to localize
            # faces in the input image
            detector.setInput(imageBlob)
            detections = detector.forward()

            # loop over the detections
            # for i in range(0, detections.shape[2]):
            if len(detections) > 0:
                # we're making the assumption that each image has only ONE
                # face, so find the bounding box with the largest probability
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]

                # ensure that the detection with the largest probability also
                # means our minimum probability test (thus helping filter out
                # weak detections)
                if confidence > args["confidence"]:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # extract the face ROI and grab the ROI dimensions
                    face = frame[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    # ensure the face width and height are sufficiently large
                    if fW < 20 or fH < 20:
                        continue

                    # construct a blob for the face ROI, then pass the blob
                    # through our face embedding model to obtain the 128-d
                    # quantification of the face
                    # faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    # 	(96, 96), (0, 0, 0), swapRB=True, crop=False)
                    # embedder.setInput(faceBlob)
                    # vec = embedder.forward()
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    idx, proba = predict(face, data["embeddings"])
                    name = data["names"][idx]


                    # draw the bounding box of the face along with the
                    # associated probability
                    text = "{}: {:.2f}%".format(name, proba * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                            (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                    # # extract the confidence (i.e., probability) associated with
                    # # the prediction
                    # confidence = detections[0, 0, i, 2]

                    # # filter out weak detections
                    # if confidence > args["confidence"]:
                    #         # compute the (x, y)-coordinates of the bounding box for
                    #         # the face
                    #         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    #         (startX, startY, endX, endY) = box.astype("int")

                    #         # extract the face ROI
                    #         face = frame[startY:endY, startX:endX]
                    #         (fH, fW) = face.shape[:2]

                    #         # ensure the face width and height are sufficiently large
                    #         if fW < 20 or fH < 20:
                    #                 continue

                    #         # construct a blob for the face ROI, then pass the blob
                    #         # through our face embedding model to obtain the 128-d
                    #         # quantification of the face
                    #         # faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    #         #         (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    #         # embedder.setInput(faceBlob)
                    #         # vec = embedder.forward()
                    #         # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    #         idx, proba = predict(face, data["embeddings"])
                    #         name = data["names"][idx]


                    #         # draw the bounding box of the face along with the
                    #         # associated probability
                    #         text = "{}: {:.2f}%".format(name, proba * 100)
                    #         y = startY - 10 if startY - 10 > 10 else startY + 10
                    #         cv2.rectangle(frame, (startX, startY), (endX, endY),
                    #                 (0, 0, 255), 2)
                    #         cv2.putText(frame, text, (startX, y),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            # update the FPS counter
            # fps.update()

            # show the output frame
            out.write(frame)

            # cv2.imshow("Frame", frame)
            # key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            # if key == ord("q"):
            #        break

    # stop the timer and display FPS information
    # fps.stop()
    # print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    # vs.stop()

elif args["dt"]=="dlib":
    # USAGE
    # python recognize_faces_video.py --encodings encodings.pickle
    # python recognize_faces_video.py --encodings encodings.pickle --output output/jurassic_park_trailer_output.avi --display 0

    # import the necessary packages
    from imutils.video import VideoStream
    import face_recognition
    import argparse
    import imutils
    import pickle
    import time
    import cv2


    # initialize the video stream and pointer to output video file, then
    # allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = cap
    writer = None
    time.sleep(2.0)

    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        ret, frame = vs.read()
        if type(frame)==type(None):
            break

        # convert the input frame from BGR to RGB then resize it to have
        # a width of 750px (to speedup processing)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # rgb = imutils.resize(frame, width=750)
        r = frame.shape[1] / float(rgb.shape[1])

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input frame, then compute
        # the facial embeddings for each face
        boxes = face_recognition.face_locations(rgb,
            model=args["detection_method"])
        encodings = []
        names = []
        for (top, right, bottom, left) in boxes:
            startY = top
            endY = bottom
            startX = left
            endX = right
            # extract the face ROI
            face = rgb[startY:endY, startX:endX]
            (fH, fW) = rgb.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                    continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            # faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
            #         (96, 96), (0, 0, 0), swapRB=True, crop=False)
            # embedder.setInput(faceBlob)
            # vec = embedder.forward()
            idx, proba = predict(face, data["embeddings"])
            names.append(data["names"][idx])

        # loop over the facial embeddings
        # for encoding in encodings:
        #     # attempt to match each face in the input image to our known
        #     # encodings
        #     matches = face_recognition.compare_faces(data["encodings"],
        #         encoding)
        #     name = "Unknown"

        #     # check to see if we have found a match
        #     if True in matches:
        #         # find the indexes of all matched faces then initialize a
        #         # dictionary to count the total number of times each face
        #         # was matched
        #         matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        #         counts = {}

        #         # loop over the matched indexes and maintain a count for
        #         # each recognized face face
        #         for i in matchedIdxs:
        #             name = data["names"][i]
        #             counts[name] = counts.get(name, 0) + 1

        #         # determine the recognized face with the largest number
        #         # of votes (note: in the event of an unlikely tie Python
        #         # will select first entry in the dictionary)
        #         name = max(counts, key=counts.get)
            
            # update the list of names
            # names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # rescale the face coordinates
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
                (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)

        # if the video writer is None *AND* we are supposed to write
        # the output video to disk initialize the writer
        if writer is None and args["output"] is not None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 20,
                (frame.shape[1], frame.shape[0]), True)

        # if the writer is not None, write the frame with recognized
        # faces t odisk
        if writer is not None:
            writer.write(frame)

        # check to see if we are supposed to display the output frame to
        # the screen
        if args["display"] > 0:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

    # check to see if the video writer point needs to be released
    if writer is not None:
        writer.release()