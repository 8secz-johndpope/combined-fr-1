from deepface import DeepFace
from deepface.commons import functions
from glob import glob
from collections import defaultdict
import random
import os, shutil
import pandas as pd

f_list = glob("nested2/*/*")
d = defaultdict(list)

for f in f_list:
    d[f.split('/')[1].split('_')[0].lower()].append(f)

print(len(f_list))

tot = 0
f_p = 0

# det = 0
# random.shuffle(f_list)
# for i, f in enumerate(f_list):
#     try:
#         result = DeepFace.verify(f, f, model_name='Facenet', enforce_detection=True)["verified"]
#         print(f"Face found for {f}")
#         det += 1
#         print(det, det/(i+1))
#     except:
#         continue
# exit()

# f_list = glob("flex_milpitas/*")
# d = defaultdict(list)
# for f in f_list:
#     d[f.split('/')[1].split('_')[0]].append(f)
# for k, v in d.items():
#     os.makedirs(f"nested2/{k}", exist_ok=True)
#     for i, f in enumerate(v):
#         shutil.copyfile(f, f"nested2/{k}/{i}.jpg")



# f_list = glob("flex_milpitas/*")
# for i, f in enumerate(f_list):
#     print(i/len(f_list))
#     try:
#         functions.detectFace(f, (160, 160), enforce_detection=True)
#     except ValueError:
#         os.remove(f)
# exit()

df = pd.DataFrame(index=d.keys(), columns=d.keys())

tot = 0
f_p = 0
t_p = 0
for id0 in d.keys():
    for id1 in d.keys():
        if df[id0][id1] != df[id0][id1] and id0!=id1:
            try:
                result = DeepFace.verify(d[id0][0], d[id1][0],
                            model_name='Facenet', enforce_detection=True)["verified"]
            except ValueError:
                print(f"Face not found for {f}")
                continue
            tot += 1
            df[id0][id1] = result
            if result:
                f_p += 1
                print(f"False match for {f}")
            print(f"After {tot}, False postive percent {f_p/(tot+1)*100:.2f}")
        else:
            print("Same ID")
            

print(f"False postive percent {f_p/tot*100:.2f}")

from IPython import embed; embed()