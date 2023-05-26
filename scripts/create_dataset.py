# Script to Create Datasets

import os
import numpy as np
from tqdm import tqdm
import shutil

def moveIMages(no_images,source,destination):
    list_img = os.listdir(source)
    list_img = np.random.choice(list_img,size=no_images,replace=False)

    print(f'Copying {no_images} images from {source} to {destination}')
    for img in tqdm(list_img):
        src = os.path.join(source,img)
        dest = os.path.join(destination,img)
        shutil.copy(src,dest)

cartoon_dataset_path = '/home/cvblgita/tri/Datasets/Cartoon/train'
face_dataset_path = '/home/cvblgita/tri/Datasets/CelebA/train'
save_path = '/home/cvblgita/tri/Datasets/knowledge_distillation/train_50K_cartoon_10k_face'

no_cartoon = 50000   
no_face = 10000

moveIMages(no_cartoon,cartoon_dataset_path,save_path)
moveIMages(no_face,face_dataset_path,save_path)
