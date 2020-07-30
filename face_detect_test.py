import cv2, os


img = cv2.imread("/root/combined-fr-1/flex_milpitas_bak/ 506404_dac-nguyen_1562688792879.jpeg")

scales = []
for i in range(1001, 1500):
    folders = cv2.__file__.split(os.path.sep)[0:-1]
    path = folders[0]
    for folder in folders[1:]:
        path = path + "/" + folder
    face_detector_path = path+"/data/haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(face_detector_path)

    faces = face_detector.detectMultiScale(img, i/1000, 5)
    if faces!=tuple():
        print(f"{i/1000} Found")
        scales.append(i/1000)

print(scales)