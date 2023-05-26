import numpy as np
import tensorflow as tf
from networks import generator
from options.test_options import TestOptions
from tqdm import tqdm
import os
import cv2
import glob

feature_svae_path = '/home/cvblgita/tri/Datasets/knowledge_distillation/Features_Hidden_CelebA/'

opt = TestOptions().parse()
save_path = os.path.join(opt.checkpoint_path, opt.save_path, opt.image_save_path)
#svae_path = '/home/cvblgita/tri/branch7/EIPNet/checkpoint/CelebA/result'
if not os.path.exists(save_path):
    os.makedirs(save_path)
checkpoint_path = opt.weight_path

class get_evaluation(object):
    def __init__(self, opt):
        self.opt = opt
        self.test_list = glob.glob(os.path.join(self.opt.test_path, '*.jpg'))


    def open_image(self, path, width, height, angle, isDown=True, isCrop=False, isResize=True, isflip=False, isRotate=False):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if isCrop:
            img = img[20:198, 0:178]
        if isResize:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        if isflip:  # horizontal flip
            img = cv2.flip(img, 1)
        if isRotate:
            img_waf = img
            img = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
            img = cv2.warpAffine(img_waf, img, (width, height))
        if isDown:
            img_lr_2 = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)
            img_lr_4 = cv2.resize(img_lr_2, (32, 32), interpolation=cv2.INTER_LINEAR)
            img_lr = cv2.resize(img_lr_4, (16, 16), interpolation=cv2.INTER_LINEAR)


        img_lr = img_lr.astype(np.float32)
        img = img.astype(np.float32)

        return img, img_lr

    def get_psnr_ssim(self,):

        for image in tqdm(sorted(self.test_list)):
            name = image.split('/')[-1]
            name = name.split('.')[0]
            imgs = []
            imgs_lr = []

            img, img_lr = self.open_image(image, width=self.opt.crop_size, height=self.opt.crop_size,
                                          isDown=True,
                                          isCrop=False,
                                          isResize=True,
                                          isflip=False,
                                          isRotate=False,
                                          angle=0)

            imgs.append(img)
            imgs_lr.append(img_lr)

            imgs_hr = np.array(imgs)
            imgs_lr = np.array(imgs_lr)

            imgs_sr, edgex2, edgex4, edgex8, kd_edge = sess.run([RGB, Step1_edge, Step2_edge, Step3_edge, Step1_res],
                feed_dict={X_hr: imgs_hr,
                           X_lr: imgs_lr})

            #np.save(feature_svae_path+name, kd_edge)
            cv2.imwrite(os.path.join(save_path ,'test_50k_cartoon_10k_face_CelebA', 'SR'  , str(name) + '_SR' + '.jpg'), cv2.cvtColor(imgs_sr[0] / 255, cv2.COLOR_RGB2BGR) * 255)
           
            imgs_lr = cv2.resize(imgs_lr[0], (128, 128), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(save_path,'test_50k_cartoon_10k_face_CelebA', 'LR', str(name) + '_LR' + '.jpg'), cv2.cvtColor(imgs_lr / 255, cv2.COLOR_RGB2BGR) * 255)

            cv2.imwrite(os.path.join(save_path,'test_50k_cartoon_10k_face_CelebA', 'HR', str(name) + '_HR' + '.jpg'), cv2.cvtColor(imgs_hr[0]/ 255, cv2.COLOR_RGB2BGR) * 255)
           


X_lr = tf.placeholder(tf.float32, shape=[opt.batchSize, opt.crop_size/8, opt.crop_size/8, opt.output_nc])
X_hr = tf.placeholder(tf.float32, shape=[opt.batchSize, opt.crop_size, opt.crop_size, opt.output_nc])

training = False

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = opt.gpu_ids

RGB, Step1_edge, Step2_edge, Step3_edge, Step1_res = generator(X_lr)

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=None)
#save_file = os.path.join(checkpoint_path, 'G_weight.ckpt')
save_file = '/home/cvblgita/tri/branch7/EIPNet/checkpoint/CelebA/cpt_50k_cartoon_10k_face_default/G_epoch_141.ckpt'
saver.restore(sess, save_file)
evaluation = get_evaluation(opt)
evaluation.get_psnr_ssim()
print('\n HR images are generated !')
