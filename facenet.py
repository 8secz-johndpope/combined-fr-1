from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def collate_fn(x):
    return x[0]

workers = 10
dataset = datasets.ImageFolder('nested')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

sz = 50

aligned = []
names = []
embeddings_list = []
i = 0
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])
        if not (i+1) % sz:
            aligned = torch.stack(aligned).to(device)
            embeddings = resnet(aligned).detach().cpu()
            embeddings_list.append(embeddings)
            aligned = []
            # break
        i += 1
    print('Face not detected with probability: {:8f}'.format(prob))

# embeddings = torch.cat(embeddings_list, dim=0)
embeddings = embeddings_list[0]
torch.save((names, embeddings), "facenet_embeddings.pt")

names, embeddings = torch.load("facenet_embeddings.pt")

dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
from IPython import embed; embed()

diff = np.array([dists[i][j] for i in range(len(dists)) for j in range(len(dists[0])) if i//2 != j//2])
same = np.array([dists[i][j] for i in range(len(dists)) for j in range(len(dists[0])) if i // 2 == j // 2])

len([dists[i][j] < 0.001 for i in range(len(dists)) for j in range(len(dists[0])) if i//2 != j//2])
len([(names[i], i, names[j], j) for i in range(len(dists)) for j in range(len(dists[0]))
 if i//2 != j//2 and dists[i][j] < 0.4])

f_p = []
t_p = []
for thresh in np.arange(same.min(), diff.max(), 0.0001):
    f_p.append((diff < thresh).sum()/len(diff)*100)
    t_p.append((same < thresh).sum()/len(same)*100)

print(pd.DataFrame(dists, columns=names, index=names))

