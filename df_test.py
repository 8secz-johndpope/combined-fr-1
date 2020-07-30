import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
logging.getLogger("tensorflow").setLevel(logging.FATAL)
from deepface import DeepFace
from deepface.commons import realtime
import youtube_dl
import cv2
from cv2 import VideoCapture


###########################################YOUTUBE


f_list = glob("nested/*/*")
ids = defaultdict(list)
for f in f_list:
    ids[f.split('/')[1]].append(f)

dfs = DeepFace.find(img_path = f_list, db_path = "nested")


models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID"]
realtime.analysis(cap=cap, out=out, db_path="friends", model_name=models[1], distance_metric='cosine',
enable_face_analysis = False, embd_saved=True)