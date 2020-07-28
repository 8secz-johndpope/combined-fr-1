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
# create youtube-dl object
ydl = youtube_dl.YoutubeDL()

# set video url, extract video information
video_url = "https://www.youtube.com/watch?v=6DK0yrF_ffU"
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
###########################################

out = cv2.VideoWriter("output/deepface.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,720))


models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID"]
realtime.analysis(cap=cap, out=out, db_path="friends", model_name=models[1], distance_metric='cosine',
enable_face_analysis = False, embd_saved=False)