import tensorflow as tf
from networks import generator, discriminator
from options.train_options import TrainOptions

from utils import make_data_path, data_augmentation
from visualizer import Visualizer
import os
from collections import OrderedDict

kd_save_path = '/home/cvblgita/tri/branch5/EIPNet/datasets/kd_train'
kd_hidden_path = '/home/cvblgita/tri/Datasets/knowledge_distillation/Features_Hidden_CelebA'

opt = TrainOptions().parse()
vis = Visualizer(opt)
save_path = os.path.join(opt.checkpoint_path, opt.save_path)
train_list, test_list = make_data_path(opt.train_path, opt.test_path)
data_generator = data_augmentation(train_list, test_list, kd_save_path, kd_hidden_path)

_BATCH_SIZE = opt.batchSize
_IMAGE_SIZE = opt.crop_size
_RESIZED_IMAGE = _IMAGE_SIZE / 8

X_lr = tf.placeholder(tf.float32, shape=[_BATCH_SIZE, _RESIZED_IMAGE, _RESIZED_IMAGE, 3])
X_hr = tf.placeholder(tf.float32, shape=[_BATCH_SIZE, _IMAGE_SIZE, _IMAGE_SIZE, 3])
X_grad = tf.placeholder(tf.float32, shape=[_BATCH_SIZE, _IMAGE_SIZE, _IMAGE_SIZE, 1])
X_grad_2 = tf.placeholder(tf.float32, shape=[_BATCH_SIZE, _IMAGE_SIZE/2, _IMAGE_SIZE/2, 1])
X_grad_4 = tf.placeholder(tf.float32, shape=[_BATCH_SIZE, _IMAGE_SIZE/4, _IMAGE_SIZE/4, 1])
kd_hr = tf.placeholder(tf.float32, shape=[_BATCH_SIZE, _IMAGE_SIZE, _IMAGE_SIZE, 3])
encoding_kd = tf.placeholder(tf.bool, shape=[_BATCH_SIZE])
kd_hidden = tf.placeholder(tf.float32, shape=[_BATCH_SIZE, _RESIZED_IMAGE, _RESIZED_IMAGE, 512])

RGB, Step1_edge, Step2_edge, Step3_edge, Step1_res = generator(X_lr)
neg_encoding_kd = tf.math.logical_not(encoding_kd)
# train discriminator
X_real = discriminator(tf.boolean_mask(X_hr,neg_encoding_kd), reuse=False)
X_fake = discriminator(tf.boolean_mask(RGB,neg_encoding_kd), reuse=True)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(X_real), logits=X_real))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(X_fake), logits=X_fake))
d_loss = 1000*(d_loss_real + d_loss_fake)

d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

# train generator
YUV_SR = tf.image.rgb_to_yuv(RGB)
YUV_HR = tf.image.rgb_to_yuv(X_hr)

MSE = tf.reduce_mean(tf.math.square(tf.boolean_mask(RGB-X_hr,neg_encoding_kd)))

MSE_edge = tf.reduce_mean(tf.math.square(tf.boolean_mask(Step1_edge-X_grad_4,neg_encoding_kd))) \
          + tf.reduce_mean(tf.math.square(tf.boolean_mask(Step2_edge-X_grad_2,neg_encoding_kd))) \
          + tf.reduce_mean(tf.math.square(tf.boolean_mask(Step3_edge-X_grad,neg_encoding_kd)))

MSE_YUV = tf.reduce_mean(tf.math.square(tf.boolean_mask(YUV_SR-YUV_HR,neg_encoding_kd)))

ad_loss = opt.ad_r * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(X_fake), logits=X_fake))

g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

# kd on output layer of teacher
kd_masked_RGB = tf.boolean_mask(RGB, encoding_kd)
kd_masked_hr = tf.boolean_mask(kd_hr, encoding_kd)
kd_loss = tf.reduce_mean(tf.math.square(kd_masked_RGB-kd_masked_hr))
value_not_nan = tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(kd_loss)), dtype=tf.float32)
kd_loss = tf.math.multiply_no_nan(kd_loss, value_not_nan)


# kd on hidden layer of teacher
kd_masked_hidden_GT = tf.boolean_mask(Step1_res,encoding_kd)
kd_masked_hidden_curr = tf.boolean_mask(kd_hidden,encoding_kd)
kd_hidden_loss = tf.reduce_mean(tf.math.square(kd_masked_hidden_GT-kd_masked_hidden_curr))
value_not_nan = tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(kd_hidden_loss)), dtype=tf.float32)
kd_hidden_loss = tf.math.multiply_no_nan(kd_hidden_loss, value_not_nan)

cost =  MSE + opt.lambda_edge * MSE_edge + MSE_YUV + ad_loss + 15*kd_loss + 0.08*kd_hidden_loss

start_lr = opt.learning_rate
global_step = tf.Variable(0, trainable=False)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=start_lr).minimize(cost, var_list=g_vars, global_step=global_step)
    optimizer_d = tf.train.AdamOptimizer(learning_rate=start_lr).minimize(d_loss, var_list=d_vars)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = opt.gpu_ids
iteration = int(len(train_list) / _BATCH_SIZE)
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=opt.max_to_keep)
    G_saver = tf.train.Saver(var_list=g_vars, max_to_keep=opt.max_to_keep)
    if opt.continue_train:
        #save_file = os.path.join(save_path, 'epoch_%03d.ckpt' % opt.which_epoch)
        save_G = '/home/cvblgita/tri/branch7/EIPNet/checkpoint/CelebA/G_epoch_17.ckpt'
        save_D = '/home/cvblgita/tri/branch7/EIPNet/checkpoint/CelebA/epoch_17.ckpt'
        G_saver.restore(sess, save_G)
        saver.restore(sess, save_D)
        print('Generator Model restored from file: %s' % save_G)
        print('Discriminator Model restored from file: %s' % save_D)


    print('Learning Started ! ========================================')

    for epoch in range(opt.epoch):
        losses = vis.loss_initialization()
        for batch in range(iteration):
            for d_batch in range(opt.D_steps_per_G):
                imgs_hr, imgs_lr, imgs_grad, imgs_grad_2, imgs_grad_4, imgs_encoding, kd_imgs_hr, kd_hidden_imgs = data_generator.next_batch(_BATCH_SIZE,
                                                                                                  _IMAGE_SIZE,
                                                                                                  _IMAGE_SIZE)

                d_cost, opti_d_train = sess.run(
                    [d_loss, optimizer_d],
                    feed_dict={X_hr: imgs_hr,
                               X_lr: imgs_lr,
                               X_grad: imgs_grad,
                               X_grad_2: imgs_grad_2,
                               X_grad_4: imgs_grad_4,
                               kd_hr: kd_imgs_hr,
                               encoding_kd: imgs_encoding})

            imgs_hr, imgs_lr, imgs_grad, imgs_grad_2, imgs_grad_4, imgs_encoding, kd_imgs_hr, kd_hidden_imgs = data_generator.next_batch(_BATCH_SIZE,
                                                                                              _IMAGE_SIZE,
                                                                                              _IMAGE_SIZE)
            cost_train, mse_train, edge_train, yuv_train, g_train, opti_train, global_steps, kd_train, kd_hidden_train = sess.run(
                [cost, MSE, MSE_edge, MSE_YUV, ad_loss, optimizer, global_step, kd_loss, kd_hidden_loss],
                feed_dict={X_hr: imgs_hr,
                           X_lr: imgs_lr,
                           X_grad: imgs_grad,
                           X_grad_2: imgs_grad_2,
                           X_grad_4: imgs_grad_4,
                           kd_hr: kd_imgs_hr,
                           encoding_kd: imgs_encoding,
                           kd_hidden: kd_hidden_imgs})


            losses['G_total'] += cost_train / iteration
            losses['MSE'] += mse_train / iteration
            losses['EDGE'] += edge_train / iteration
            losses['YUV'] += yuv_train / iteration
            losses['G_GAN'] += g_train / iteration
            losses['D_GAN'] += d_cost / iteration
            losses['kd_output'] += kd_train/ iteration
            losses['kd_hidden'] += kd_hidden_train / iteration


        vis.print_save_current_error(epoch + 1, global_steps, losses)

        saver.save(sess, save_path + '/epoch_%02d.ckpt' % (epoch + 1))
        G_saver.save(sess, save_path + '/G_epoch_%02d.ckpt' % (epoch + 1))

        val_iteration = 1
        # for v_epoch in range(val_iteration):
        #     imgs_hr, imgs_lr, imgs_grad, imgs_grad_2, imgs_grad_4 = data_generator.next_batch_test(_BATCH_SIZE,
        #                                                                                            _IMAGE_SIZE,
        #                                                                                            _IMAGE_SIZE)
        #     cost_val, imgs_sr, imgs_sr_grad = sess.run([cost, RGB, Step3_edge],
        #                                                                   feed_dict={X_hr: imgs_hr,
        #                                                                              X_lr: imgs_lr,
        #                                                                              X_grad: imgs_grad,
        #                                                                              X_grad_2: imgs_grad_2,
        #                                                                              X_grad_4: imgs_grad_4})
        # visuals = OrderedDict([('reconstructed_image', imgs_sr), ('real_image', imgs_hr)])
        # vis.save_image(epoch + 1, visuals)



print('Learning Finished !')



