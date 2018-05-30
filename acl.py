import json
import random
import imageio
import os
import argparse
import time
import numpy as np
import tensorflow as tf
from scipy import misc
import utils
from ops import *
import sys
from generate_gif import *
BOOL_LABEL=3

def load_dataset(dataset_file, min_group_size, max_jpgs=-1):
    with open(dataset_file) as f:
        doc = json.load(f)
    last_fid = -2
    data = []
    for i, line in enumerate(doc):
        if max_jpgs != -1 and i > max_jpgs:
            break
        fid = line[1]
        if fid - last_fid == 1:
            data[-1].append(line)
        else:
            data.append([line])
        last_fid = fid
    return [group for group in data if len(group) >= min_group_size]

def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

def print_in_file(sstr):
    sys.stdout.write(str(sstr)+'\n')
    sys.stdout.flush()
    os.fsync(sys.stdout)

class Net:
    def __init__(self, H):
        self.batch_size = H["batch_size"]
        self.dataset_file = H["dataset_file"]
        self.sim_dataset_file = H.get("sim_dataset_file", self.dataset_file)
        self.test_size = H.get("test_size", 0)
        self.h_dim = H["h_dim"]
        self.r_dim = H.get("r_dim", 64)
        self.d_noise = H["d_noise"]
        self.physics = H["physics"]
        self.symb_dim = H["symb_dim"]
        self.mode = H["mode"]
        self.lstm_dim = H["lstm_dim"]
        self.im_height = self.im_width = H["im_height"]
        self.group_size = H["group_size"]
        self.max_jpgs = H["max_jpgs"]
        self.test_batch_size = H.get('self.test_batch_size', 32)
        self.loc_vec = [0, 2, 4, 6, 8, 10]
        seed = 0
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.build()
        
    def regressor(self, x, keep_prob, reuse=False):
        with tf.variable_scope('G', initializer=tf.random_uniform_initializer(-0.1, 0.1), reuse=reuse):
            ip = tf.get_variable('ip', shape=(256, self.symb_dim))
            feat = make_feat(x, self.im_height, keep_prob)
            return tf.matmul(feat, ip)

    def discriminator(self, xs, reuse, scope='D'):
        with tf.variable_scope(scope, initializer=tf.random_uniform_initializer(-0.1, 0.1), reuse=reuse):
            w1 = tf.get_variable('w1', shape=(self.symb_dim, self.h_dim))
            w2 = tf.get_variable('w2', shape=(self.h_dim, self.h_dim))
            w3 = tf.get_variable('w3', shape=(self.h_dim, self.lstm_dim))
            ip = tf.get_variable('ip', shape=(self.lstm_dim, 2))
            disc_input = xs
            h = tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(disc_input, w1)), w2)), w3)
            lstm_input = tf.reshape(h, (self.batch_size, self.group_size, self.lstm_dim))
            lstm_output = build_lstm_inner(lstm_input, self.batch_size, self.lstm_dim, self.group_size)[-1]
            return tf.matmul(lstm_output, ip)

    def build(self):
        self.keep_prob = keep_prob = tf.placeholder(tf.float32, [], name='kb')
        self.learning_rate = learning_rate = tf.placeholder(tf.float32, [], name='lr')
        self.x_image = x_image = tf.placeholder(tf.float32, shape=[None, self.im_height, self.im_width, 3], name='x_image')
        self.x_label = x_label = tf.placeholder(tf.float32, shape=[None, self.symb_dim], name='x_label')
        self.x_real = x_real = tf.placeholder(tf.float32, shape=[None, self.symb_dim], name='x_real')
        self.x_bool_label = x_bool_label = tf.placeholder(tf.float32, shape=[None])
        self.y = y = x_label + self.d_noise * tf.random_normal(tf.shape(x_label))
        self.y_ = y_ = self.regressor(self.x_image, keep_prob, reuse=False)
        self.pred = pred = self.discriminator(y, reuse=None, scope='D2')
        self.pred_ = pred_ = self.discriminator(y_, reuse=True, scope='D2')

        scale = 10.
        self.g_loss_label = tf.reduce_sum(tf.reduce_mean(tf.abs(y_ - x_real), axis=1) * x_bool_label)
        self.g_loss_w = -tf.reduce_mean(pred_)
        self.d_loss_w = -tf.reduce_mean(pred) + tf.reduce_mean(pred_)
        eps = tf.random_uniform([], 0., 1.)
        y_hat = eps * self.y + (1 - eps) * self.y_
        d_hat = self.discriminator(y_hat, reuse=True, scope='D2')
        ddy = tf.gradients(d_hat, y_hat)[0]
        ddy = tf.sqrt(tf.reduce_sum(tf.square(ddy), axis=1))
        ddy = tf.reduce_mean(tf.square(ddy - 1.) * scale)
        self.loss_g_with_label = self.g_loss_w + scale * self.g_loss_label        
        self.loss_g_only_label = self.g_loss_label
        self.loss_d = self.d_loss_w + ddy
        self.opt_g = opt_g = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)
        self.opt_d = opt_d = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)
        
        theta_G = [v for v in tf.trainable_variables() if 'G' in v.name]
        self.train_op_g_with_label = opt_g.minimize(self.loss_g_with_label, var_list = theta_G) 
        self.train_op_g_only_label = opt_g.minimize(self.loss_g_only_label, var_list = theta_G)
        theta_D = [v for v in tf.trainable_variables() if 'D' in v.name]
        self.train_op_d = opt_d.minimize(self.loss_d, var_list=theta_D)
        
        self.saver = tf.train.Saver(max_to_keep=None)

    def read_info(self, group, phase):
        labels = np.array([label for _, _, label, _, _ in group])
        bool_labels = np.array([bool_label for _, _, _, bool_label, _ in group])
        pck_limits = np.array([pck_limit for _, _, _, _, pck_limit in group])
        if phase != 'sim':
            imgs = np.array([utils.image_cache(fname, self.im_height) for fname, _, _, _, _ in group])
            return imgs, labels, bool_labels, pck_limits
        else:
            return _, labels, bool_labels, pck_limits

    def gen_data_render_catch(self, dataset, phase, num_with_label):
        def label_data(dataset_to_label):
            for group in dataset_to_label:
                for frame in group:
                    frame[BOOL_LABEL] = 1
            return
        if phase == 'train':
            label_data(dataset[:num_with_label])
        elif phase == 'label':
            label_data(dataset)
        print_in_file('Beginning %s phase, dataset has %d groups' % (phase, len(dataset)))
        # print dataset[0]        
        while True:
            if phase == 'train' or phase == 'label' or phase == 'sim':
                random.shuffle(dataset)
            for group in dataset:
                start_id = np.random.randint(0, len(group) - self.group_size + 1)
                group_short = group[start_id:start_id + self.group_size]
                yield self.read_info(group_short, phase)

    def eval(self, sess, args):
        if not os.path.exists(args.logdir+'/output'):
            os.makedirs(args.logdir+'/output')

        if args.eval_only:
            self.dataset_for_test = load_dataset(self.dataset_file, self.group_size)

        if args.weights is not None:
            self.saver.restore(sess, args.weights)
            print_in_file("Saved")

        pck_result = np.zeros((6))
        flattened_set = []

        for group in self.dataset_for_test:
            flattened_set.extend(group)

        test_num = len(flattened_set)
        test_imgs, test_labels, _, test_pck_limits = self.read_info(flattened_set, 'test')
        i_pos = range(test_num//self.test_batch_size)
        s_pos = [0 for _ in i_pos]
        i_pos = [i * self.test_batch_size for i in i_pos]
        i_pos.append(test_num - self.test_batch_size)
        s_pos.append(test_num // self.test_batch_size * self.test_batch_size - i_pos[-1])

        for pos in range(len(i_pos)):
            feed = {self.x_image: test_imgs[i_pos[pos]:i_pos[pos]+self.test_batch_size], self.keep_prob: 1.0}
            np_x_real = test_labels[i_pos[pos]:i_pos[pos]+self.test_batch_size]
            np_pck_limits = test_pck_limits[i_pos[pos]:i_pos[pos]+self.test_batch_size]
            np_y_ = sess.run(self.y_, feed_dict=feed)
            assert np_x_real.shape == np_y_.shape
            for pos_s in range(s_pos[pos], self.test_batch_size):
                for loc in self.loc_vec:
                    if np.abs(np_y_[pos_s,loc] - ((1+np_x_real[pos_s,loc])*32) ) < np_pck_limits[pos_s] and \
                        np.abs(np_y_[pos_s,loc+1] - ((1+np_x_real[pos_s,loc+1])*32) ) < np_pck_limits[pos_s]:
                        pck_result[loc//2] += 1
        
        for loc in xrange(6):
            print_in_file(pck_result[loc] / (test_num))

    def reshape_arrays(self, imgs, real, bool_labels, pck_limits, labels):
        imgs = np.reshape(imgs, (self.batch_size*self.group_size, self.im_height, self.im_width, 3))
        real = np.reshape(real, (self.batch_size*self.group_size, self.symb_dim))
        real = (1+real)*32
        bool_labels = np.reshape(bool_labels, (self.batch_size*self.group_size))
        pck_limits = np.reshape(pck_limits, (self.batch_size*self.group_size))
        labels = np.reshape(labels, (self.batch_size*self.group_size, self.symb_dim))
        return imgs, real, bool_labels, pck_limits, labels
    
    def calculate_pck(self, group_len, y_, x_real, pck_limits):
        pck_result = np.zeros((self.symb_dim/2))
        for frame in range(group_len):
            for loc in self.loc_vec:
                if np.abs(y_[frame,loc] - x_real[frame,loc]) < pck_limits[frame] and \
                     np.abs(y_[frame,loc+1] - x_real[frame,loc+1]) < pck_limits[frame]:
                    pck_result[loc/2] += 1
        pck_result = pck_result / group_len
        return pck_result

    def train(self, sess, args):
        self.logdir = args.logdir+parse_time()
        while os.path.exists(self.logdir):
            time.sleep(random.randint(1,5))
            self.logdir = args.logdir+parse_time()
        os.makedirs(self.logdir)

        if not os.path.exists('%s/logs'%self.logdir):
            os.makedirs('%s/logs'%self.logdir)  

        if args.train_with_label_only:
            assert args.num_with_label > 0

        fsock_err = open('./%s/error.log'%(self.logdir), 'w')               
        fsock_out = open('./%s/out.log'%(self.logdir), 'w')               
        sys.stderr = fsock_err     
        sys.stdout = fsock_out
    
        self.stat_str = '%s/stat.txt'%(self.logdir)
        self.hyper_str = '%s/hyper.txt'%(self.logdir)
        f_stat = open(self.stat_str, 'w')
        f_hyper = open(self.hyper_str, 'w')

        self.dataset = load_dataset(self.dataset_file, self.group_size, max_jpgs=self.max_jpgs)
        self.sim_dataset = load_dataset(self.sim_dataset_file, self.group_size, max_jpgs=self.max_jpgs)
        random.shuffle(self.dataset)
        self.dataset_for_train = self.dataset[:-self.test_size]
        self.dataset_for_test = self.dataset[-self.test_size:]
        self.dataset_for_label = self.dataset[:args.num_with_label]
        print >> f_hyper, "Train Set:"
        for i in self.dataset_for_train:
            print >> f_hyper, i[0][1]
        print >> f_hyper, "Label Set:"
        for i in self.dataset_for_label:
            print >> f_hyper, i[0][1]
        print >> f_hyper, "Test Set:"
        for i in self.dataset_for_test:
            print >> f_hyper, i[0][1]
        f_hyper.close()

        data_gen0 = utils.gen_data_batch(self.batch_size, self.gen_data_render_catch, 
                        [self.dataset_for_train, 'train', args.num_with_label])
        data_gen1 = utils.gen_data_batch(self.batch_size, self.gen_data_render_catch, 
                        [self.dataset_for_label, 'label', args.num_with_label])
        data_gen2 = utils.gen_data_batch(self.batch_size, self.gen_data_render_catch, 
                        [self.sim_dataset, 'sim', 0])
        data_gen3 = utils.gen_data_batch(self.batch_size, self.gen_data_render_catch, 
                        [self.dataset_for_test, 'test', 0])

        if args.weights is not None:
            self.saver.restore(sess, args.weights)

        lr = args.learning_rate
        for i in range(args.iters):
            d_iters = 5
            if i < 30 or i % 300 == 0:
                d_iters = 100
            if args.train_with_label_only:
                d_iters = 0  
            if i > 0 and i % (args.iters//10) == 0:
                lr = lr * 2./3

            for _ in range(d_iters):
                np_x_image, np_x_real, np_x_bool, np_pck_limits  = data_gen0.next()
                _, np_x_label, _, _  = data_gen2.next()
                np_x_image, np_x_real, np_x_bool, np_pck_limits, np_x_label = \
                        self.reshape_arrays(np_x_image, np_x_real, np_x_bool, np_pck_limits, np_x_label)
                feed = {self.x_image: np_x_image, self.x_label: np_x_label, self.learning_rate: lr,
                            self.keep_prob: 0.5, self.x_real: np_x_real, self.x_bool_label: np_x_bool}
                _ = sess.run(self.train_op_d, feed_dict = feed)

            if args.train_with_label_only:
                np_x_image, np_x_real, np_x_bool, np_pck_limits = data_gen1.next()
                np_x_image, np_x_real, np_x_bool, np_pck_limits, _ = \
                        self.reshape_arrays(np_x_image, np_x_real, np_x_bool, np_pck_limits, np.zeros_like(np_x_real))
            else:
                np_x_image, np_x_real, np_x_bool, np_pck_limits = data_gen0.next()
                np_x_image, np_x_real, np_x_bool, np_pck_limits, np_x_label = \
                        self.reshape_arrays(np_x_image, np_x_real, np_x_bool, np_pck_limits, np_x_label)

            if args.train_with_label_only:
                feed = {self.x_image: np_x_image, self.keep_prob: 0.5, self.learning_rate: lr, 
                            self.x_real: np_x_real, self.x_bool_label: np_x_bool}
            else:
                feed = {self.x_image: np_x_image, self.x_label: np_x_label, self.keep_prob: 0.5, 
                            self.x_real: np_x_real, self.x_bool_label: np_x_bool, self.learning_rate: lr}

            if args.train_with_label_only:
                _ = sess.run(self.train_op_g_only_label, feed_dict=feed)
            else:
                _ = sess.run(self.train_op_g_with_label, feed_dict=feed)

            if i % 100 == 0:
                np_y_, np_loss_label = sess.run([self.y_, self.g_loss_label], feed_dict=feed)
                if not args.train_with_label_only:
                    np_y, np_loss_d, np_loss_g_with_label = sess.run([self.y, self.loss_d, self.g_loss_w], feed_dict=feed)
                print_in_file(i)
                # print "Iteration %d"%i
                if not args.train_with_label_only:
                    print_in_file('mean output: %f, d_loss: %f, g_loss: %f, label_loss: %f' % \
                                    (np.mean(np_y_), np_loss_d, np_loss_g_with_label, np_loss_label))
                    # print 'mean output: %f, d_loss: %f, g_loss: %f, label_loss: %f' % \
                    #         (np.mean(np_y_), np_loss_d, np_loss_g_with_label, np_loss_label)
                else:
                    print_in_file('mean output: %f, label_loss: %f' % (np.mean(np_y_), np_loss_label))
                    # print 'mean output: %f, label_loss: %f' % (np.mean(np_y_), np_loss_label)
                    
                assert np_y_.shape == np_x_real.shape, ("Shape not match", np_y_.shape, np_x_real.shape)
                if self.physics == 'kick':
                    pck_result = self.calculate_pck(self.batch_size*self.group_size, np_y_, np_x_real, np_pck_limits)
                    for loc in range(self.symb_dim/2):
                        # print >> f_err, 'Testing trained... pck%d: %f'%(loc, pck_result[loc])
                        print_in_file('Testing trained... pck%d: %f'%(loc, pck_result[loc]))
                    print >>f_stat, pck_result[0], pck_result[1], pck_result[2], pck_result[3], pck_result[4], pck_result[5]

            if i % 200 == 0 and True:
                val_x_image, val_x_real, val_x_bool, val_pck_limits = data_gen3.next()
                _, val_x_label, _, _ = data_gen3.next()
                val_x_image, val_x_real, val_x_bool, val_pck_limits, val_x_label = \
                    self.reshape_arrays(val_x_image, val_x_real, val_x_bool, val_pck_limits, val_x_label)
                feed_val = {self.x_image: val_x_image, self.x_label: val_x_label, 
                                self.keep_prob: 1., self.x_real: val_x_real, self.x_bool_label: val_x_bool}
                val_y_, val_y, val_loss_d, val_loss_g_with_label, val_loss_label = \
                        sess.run([self.y_, self.y, self.loss_d, 
                            self.loss_g_with_label, self.g_loss_label], feed_dict=feed_val)

                mims = [val_x_image[idx] for idx in range(self.group_size)]
                refims = [render(val_y_[xxx],'square',self.mode,self.im_height,img=mims[xxx], 
                            pck=val_pck_limits[xxx], truth=val_x_real[xxx]) for xxx in xrange(self.group_size)]
                imageio.mimsave('%s/val_%d.gif' % (self.logdir, i), refims, 'GIF-FI')

                np_x_image, np_x_real, np_x_bool, np_pck_limits = data_gen0.next()
                _, np_x_label, _, _ = data_gen3.next()
                np_x_image, np_x_real, np_x_bool, np_pck_limits, np_x_label = \
                        self.reshape_arrays(np_x_image, np_x_real, np_x_bool, np_pck_limits, np_x_label)
                feed = {self.x_image: np_x_image, self.keep_prob: 1., 
                            self.x_real: np_x_real, self.x_bool_label: np_x_bool}
                np_y_ = sess.run(self.y_, feed_dict=feed)
                
                mims = [np_x_image[idx] for idx in range(self.group_size)]
                refims = [render(np_y_[xxx],'square',self.mode,self.im_height,img=mims[xxx], 
                            pck=np_pck_limits[xxx], truth=np_x_real[xxx]) for xxx in xrange(self.group_size)]
                imageio.mimsave('%s/train_%d.gif' % (self.logdir, i), refims, 'GIF-FI')
                
            # if i % 1000 == 0:
            #     self.save_model(sess, self.logdir, i)

        self.save_model(sess, self.logdir, args.iters)
        f_stat.close()

    def save_model(self, sess, logdir, counter):
        ckpt_file = '%s/model-%d.ckpt' % (logdir, counter)
        print_in_file('Checkpointing to %s' % ckpt_file)
        self.saver.save(sess, ckpt_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=None, type=str)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--eval_only', default=False, action='store_true')
    parser.add_argument('--train_with_label_only', default=False, action='store_true')
    parser.add_argument('--hypes', required=False, type=str)
    parser.add_argument('--logdir', default='log/log_kick', type=str)
    parser.add_argument('--num_with_label', default=0, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--iters', default=15000, type=int)
    args = parser.parse_args()

    with open(args.hypes, 'r') as f:
        hypes_dict = json.loads(f.read())
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    assert args.logdir[-1] != '/'
    if args.train_with_label_only:
        args.logdir = '%s_L'%(args.logdir)
    args.logdir = '%s_%s/'%(args.logdir, args.num_with_label)

    net = Net(hypes_dict)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        if not args.eval_only:
            net.train(sess, args)
        net.eval(sess, args)

if __name__ == '__main__':
    main()
