import glob
import imageio
import numpy as np
import torch.nn as nn
import torch
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim
import piq
import logging
from tqdm import tqdm

device = torch.device("cuda:0")
logging.basicConfig(filename='EIP_results.log', level=logging.INFO)

path2 = '/home/cvblgita/tri/branch8/EIPNet/checkpoint/CelebA/result/test_20k_cartoon_20k_face_celebA_1/HR'
path1 = '/home/cvblgita/tri/branch8/EIPNet/checkpoint/CelebA/result/test_20k_cartoon_20k_face_celebA_1/SR'
# path1 = "/home/emrecan/Desktop/Comp411/data/results/Urban100/"
# path2 = "/home/emrecan/Desktop/Comp411/data/HR/Urban100/"

logging.info("-----------------------------------\n")
logging.info("Predicted Image path: " + str(path1))
logging.info("HR Image path: " + str(path2))



class SRFlowDataset(Dataset):
    def __init__(self, img_path):
        self.imgs_path1 = img_path
        file_list = glob.glob(self.imgs_path1 + "*")
        # print(file_list)
        self.data = []
        for img_path1 in natsorted(glob.glob(self.imgs_path1 + "/*.jpg")):
            self.data.append(img_path1)
        #print(self.data)
        # print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path1 = self.data[idx]
        img1 = imageio.imread(img_path1)
        img1 = (img1)/255.0
        # logging.info("Image Name:" + str(img_path1))
        # logging.info("Image Shape:" + str(img1.shape))
        # print(img_path1)
        # print(img1.shape)
        h, w, c = img1.shape
        h = (h//4)*4
        w = (w//4)*4
        img_tensor1 = torch.from_numpy(img1[:h, :w, :])
        img_tensor1 = img_tensor1.permute(2, 0, 1)
        img_tensor1 = img_tensor1.to(device).float()
        return {'images': img_tensor1, 'labels': idx}


dataset_01, dataset_div2k = SRFlowDataset(path1), SRFlowDataset(path2)
data_loader01 = DataLoader(dataset_01, batch_size=512, shuffle=False)
data_loaderdiv2k = DataLoader(dataset_div2k, batch_size=512, shuffle=False)


fid_metric = piq.FID()


first_feats = fid_metric.compute_feats(data_loader01)
second_feats = fid_metric.compute_feats(data_loaderdiv2k)
test_fid = fid_metric(first_feats, second_feats)

test_psnr = 0
test_ssim = 0

num_of_images = len(data_loader01)

for data01, datahr in tqdm(zip(data_loader01, data_loaderdiv2k)):
    # print(data['images'].shape)
    # print(data['labels'])
    img_pred = data01['images']
    img_hr = datahr['images']

    psnr_img = piq.psnr(img_pred, img_hr)
    test_psnr += psnr_img

    
    ssim_img = piq.ssim(img_pred, img_hr)
    test_ssim += ssim_img


test_psnr /= num_of_images
test_ssim /= num_of_images

logging.info("\n")
logging.info("PSNR:" + str(test_psnr))
logging.info("SSIM:" + str(test_ssim))
logging.info("FID Score:" + str(test_fid))

logging.info("************************************\n")


print("PSNR:" + str(test_psnr))
print("SSIM:" + str(test_ssim))
print("FID Score:" + str(test_fid))

#