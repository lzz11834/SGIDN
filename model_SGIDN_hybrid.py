from utils import *
import time
import random


def ssdcnn(input, input_gradient, is_training=True, output_channels=10):
    with tf.variable_scope('block1'):
        output = tf.transpose(input_gradient, [0, 3, 1, 2, 4])
        output = tf.layers.conv3d(output, 1, [5, 3, 3], padding='same', activation=tf.nn.relu)
        output = tf.squeeze(output,-1)
        output = tf.transpose(output, [0, 2, 3, 1])
    for layers in range(2, 19):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    with tf.variable_scope('block20'):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same')
    return input - output


class denoiser(object):
    def __init__(self, sess, input_c_dim=10, batch_size=128):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name='clean_image')
        self.X = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name='noisy_image')
        self.XG = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim, 3], name='noisy_image_gradient')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.Y = ssdcnn(self.X, self.XG, is_training=self.is_training)
        self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf_psnr(self.Y, self.Y_)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def evaluate(self, iter_num, eva_data, summary_merged, summary_writer):
        print("[*] Evaluating...")
        clean_image = eva_data.astype(np.float32)
        noisy_image = np.zeros((clean_image.shape[0], clean_image.shape[1], clean_image.shape[2], clean_image.shape[3]), dtype="float32")
        stripe = np.zeros((clean_image.shape[0], 1, clean_image.shape[2], clean_image.shape[3]), dtype="float32")

        for num in range(clean_image.shape[0]):
            for band in range(clean_image.shape[3]):
                stripe[num, :, :, band] = np.random.normal(loc=0, scale= 0.05, size=[1, clean_image.shape[2]])
                noisy_image[num, :, :, band] = clean_image[num, :, :, band] + np.random.normal(loc=0, scale=0, size=[clean_image.shape[1], clean_image.shape[2]])
                noisy_image[num, :, :, band] = salt_and_pepper_noise(noisy_image[num, :, :, band], 0)
        noisy_image = noisy_image + np.tile(stripe, [1, clean_image.shape[1], 1, 1])
        c_x = np.zeros((noisy_image.shape[0], noisy_image.shape[1], noisy_image.shape[2], noisy_image.shape[3]), dtype="float32")
        c_y = np.zeros((noisy_image.shape[0], noisy_image.shape[1], noisy_image.shape[2], noisy_image.shape[3]), dtype="float32")
        c_x[:, :, 0, :] = noisy_image[:, :, 0, :]
        c_x[:, :, 1:, :] = noisy_image[:, :, :-1, :]
        c_y[:, 0, :, :] = noisy_image[:, 0, :, :]
        c_y[:, 1:, :, :] = noisy_image[:, :-1, :, :]
        Gx1 = noisy_image - c_x
        Gy1 = noisy_image - c_y
        gradient_img = np.zeros((noisy_image.shape[0], noisy_image.shape[1], noisy_image.shape[2], noisy_image.shape[3], 3),dtype="float32")
        gradient_img[:, :, :, :, 0] = noisy_image
        gradient_img[:, :, :, :, 1] = Gx1
        gradient_img[:, :, :, :, 2] = Gy1
        output_clean_image, psnr_summary = self.sess.run([self.Y,  summary_merged],
                                                         feed_dict={self.Y_: clean_image, self.X: noisy_image,self.XG:gradient_img,
                                                                    self.is_training: False})
        summary_writer.add_summary(psnr_summary, iter_num)

    def denoise(self, data, noisy, noisy_gradient):
        output_clean_image, psnr = self.sess.run([self.Y, self.eva_psnr],
                                                 feed_dict={self.Y_: data, self.X: noisy, self.XG:noisy_gradient,self.is_training: False})
        return output_clean_image, psnr

    def train(self, data, eval_data, batch_size, ckpt_dir, epoch, lr, eval_every_epoch=10):
        num_batch = int(data.shape[0] / batch_size)
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // num_batch
            start_step = global_step % num_batch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pre-trained model!")
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', self.lr)
        writer = tf.summary.FileWriter('./logs', self.sess.graph)
        merged = tf.summary.merge_all()
        summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr)
        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        self.evaluate(iter_num, eval_data, summary_merged=summary_psnr, summary_writer=writer)
        for epoch in range(start_epoch, epoch):
            np.random.shuffle(data)
            for batch_id in range(start_step, num_batch):
                batch_images = data[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                batch_noisy = np.zeros((batch_images.shape[0], batch_images.shape[1], batch_images.shape[2], batch_images.shape[3]), dtype="float32")
                stripe = np.zeros((batch_images.shape[0], 1, batch_images.shape[2], batch_images.shape[3]), dtype="float32")
                s = np.zeros((batch_images.shape[0], batch_images.shape[3]), dtype="float32")
                g = np.zeros((batch_images.shape[0], batch_images.shape[3]), dtype="float32")
                p = np.zeros((batch_images.shape[0], batch_images.shape[3]), dtype="float32")
                for num in range (batch_images.shape[0]):
                    for band in range(batch_images.shape[3]):
                        s[num, band]= random.uniform(0, 0.1)
                        g[num, band] = random.uniform(0, 0)
                        p[num, band] = random.uniform(0, 0)
                        stripe[num, :, :, band] = np.random.normal(loc=0, scale=s[num, band], size=[1, batch_images.shape[2]])
                        batch_noisy[num, :, :, band] = batch_images[num, :, :, band] + np.random.normal(loc=0, scale=g[num,band],size=[batch_images.shape[1], batch_images.shape[2]])
                        batch_noisy[num, :, :, band] = salt_and_pepper_noise(batch_noisy[num, :, :, band],p[num, band])
                batch_noisy = batch_noisy + np.tile(stripe, [1, batch_images.shape[1], 1, 1])
                c_x = np.zeros((batch_images.shape[0], batch_images.shape[1], batch_images.shape[2],batch_images.shape[3]), dtype="float32")
                c_y = np.zeros((batch_images.shape[0], batch_images.shape[1], batch_images.shape[2],batch_images.shape[3]), dtype="float32")
                c_x[:, :, 0, :] = batch_noisy[:, :, 0, :]
                c_x[:, :, 1:, :] = batch_noisy[:, :, :-1, :]
                c_y[:, 0, :, :] = batch_noisy[:, 0, :, :]
                c_y[:, 1:, :, :] = batch_noisy[:, :-1, :, :]
                Gx1 = batch_noisy - c_x
                Gy1 = batch_noisy - c_y
                batch_gradient_img = np.zeros((batch_images.shape[0], batch_images.shape[1], batch_images.shape[2],batch_images.shape[3], 3), dtype="float32")
                batch_gradient_img[:, :, :, :, 0] = batch_noisy
                batch_gradient_img[:, :, :, :, 1] = Gx1
                batch_gradient_img[:, :, :, :, 2] = Gy1
                _, loss, summary = self.sess.run([self.train_op, self.loss, merged],
                                                 feed_dict={self.Y_: batch_images, self.X: batch_noisy,self.XG: batch_gradient_img, self.lr: lr[epoch],
                                                            self.is_training: True})
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (epoch + 1, batch_id + 1, num_batch, time.time() - start_time, loss))
                iter_num += 1
                writer.add_summary(summary, iter_num)

            if np.mod(epoch + 1, eval_every_epoch) == 0:
                self.evaluate(iter_num, eval_data,  summary_merged=summary_psnr, summary_writer=writer)
                if iter_num % (num_batch*10) == 0:
                    self.save(iter_num, ckpt_dir)
        self.save(iter_num, ckpt_dir)
        print("[*] Finish training.")

    def save(self, iter_num, ckpt_dir, model_name='SSDCNN-tensorflow'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def test(self, test_img, noisy_img, ckpt_dir):
        tf.initialize_all_variables().run()
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print("[*] Load weights SUCCESS...")
        [test_h, test_w, test_d] = np.shape(test_img)
        clean_image = np.reshape(np.array(test_img, dtype="float32"), (1,test_h, test_w, test_d))
        noisy_image = np.reshape(np.array(noisy_img, dtype="float32"), (1, test_h, test_w, test_d))
        output_clean_image =  np.zeros((1, test_h, test_w, test_d), dtype="float32")
        test_x = np.zeros((noisy_image.shape[0], noisy_image.shape[1], noisy_image.shape[2], noisy_image.shape[3]),dtype="float32")
        test_y= np.zeros((noisy_image.shape[0], noisy_image.shape[1], noisy_image.shape[2], noisy_image.shape[3]),dtype="float32")
        test_x[:, :, 0, :] = noisy_image[:, :, 0, :]
        test_x[:, :, 1:, :] = noisy_image[:, :, :-1, :]
        test_y[:, 0, :, :] = noisy_image[:, 0, :, :]
        test_y[:, 1:, :, :] = noisy_image[:, :-1, :, :]
        test_Gx1 = noisy_image - test_x
        test_Gy1 = noisy_image - test_y
        test_gradient_img = np.zeros((noisy_image.shape[0], noisy_image.shape[1], noisy_image.shape[2], noisy_image.shape[3], 3),dtype="float32")
        test_gradient_img[:, :, :, :, 0] = noisy_image
        test_gradient_img[:, :, :, :, 1] = test_Gx1
        test_gradient_img[:, :, :, :, 2] = test_Gy1
        start = time.time()
        group_size = 10
        for i in range(0, test_d, group_size):
            clean_group = clean_image[:, :, :, i:i + group_size]
            noisy_group = noisy_image[:, :, :, i:i + group_size]
            test_gradient_group = test_gradient_img[:, :, :, i:i + group_size]
            output_clean_group, test_psnr = self.sess.run([self.Y, self.eva_psnr],feed_dict={self.Y_: clean_group, self.X: noisy_group,self.XG: test_gradient_group,
                                                               self.is_training: False})
            output_clean_image[:, :, :, i:i + group_size] = output_clean_group
        print("The denoise time = %4.4f" % (time.time() - start))
        return output_clean_image
