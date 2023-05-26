import tensorflow as tf 
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

def read_img(name):
    x = tf.image.decode_image(tf.io.read_file(name),dtype=tf.float32)
    x = tf.expand_dims(x, axis=0)
    return x

def dir_images(path):
    img_arr = [os.path.join(path,x) for x in sorted(os.listdir(path))]
    return img_arr

#tf.enable_eager_execution()

GT_arr = dir_images('/home/cvblgita/tri/branch8/EIPNet/checkpoint/CelebA/result/test_20k_cartoon_20k_face_celebA_1/SR/')
SR_arr = dir_images('/home/cvblgita/tri/branch8/EIPNet/checkpoint/CelebA/result/test_20k_cartoon_20k_face_celebA_1/HR/')

psnr = tf.convert_to_tensor(0,dtype=tf.float32)
ssim = tf.convert_to_tensor(0,dtype=tf.float32)
count = tf.convert_to_tensor(0,dtype=tf.float32)

#create a list of dictionaries
PSNR_list = []
SSIM_list = []
idx = 0

for GT_img_path , SR_img_path in tqdm(zip(GT_arr,SR_arr)):

  GT_img = read_img(GT_img_path)
  SR_img = read_img(SR_img_path)

  psnr_img = tf.image.psnr(GT_img, SR_img, max_val=1.0)
  ssim_img = tf.image.ssim(GT_img, SR_img, max_val=1.0)

  idx += 1
  PSNR_list.append({'idx':idx,'psnr':psnr_img.numpy()})
  SSIM_list.append({'idx':idx,'ssim':ssim_img.numpy()})

  psnr = tf.add(psnr,psnr_img)
  ssim = tf.add(ssim,ssim_img)
  count = tf.add(count,1)

psnr = tf.divide(psnr,count)
ssim = tf.divide(ssim,count)

#sort the list of dictionaries and get indices of top 10 PSNR and SSIM
PSNR_list = sorted(PSNR_list, key = lambda i: i['psnr'],reverse=True)
SSIM_list = sorted(SSIM_list, key = lambda i: i['ssim'],reverse=True)
PSNR_list = PSNR_list[:10]
SSIM_list = SSIM_list[:10]
PSNR_list = sorted(PSNR_list, key = lambda i: i['idx'])
SSIM_list = sorted(SSIM_list, key = lambda i: i['idx'])
PSNR_list = [i['idx'] for i in PSNR_list]
SSIM_list = [i['idx'] for i in SSIM_list]
print(PSNR_list)
print(SSIM_list) 

print(f"\n\n PSNR : {psnr}  ||  SSIM : {ssim}")
