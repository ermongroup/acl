import tensorflow as tf
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
import functools
import random
import numpy as np
from scipy import misc
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def load_jpgs(im_height, data_dir, max_jpgs=-1):
    data = []
    hi_res_data = []
    last_fid = -2
    for i, f in enumerate(sorted(os.listdir(data_dir))):
        if max_jpgs != -1 and i > max_jpgs:
            break
        if 'jpg' not in f:
            continue
        fid = int(f.split('-')[1].split('.')[0])
        if fid > last_fid + 1:
            data.append([])
            hi_res_data.append([])
        last_fid = fid
        im = misc.imread(os.path.join(data_dir, f))
        data[-1].append(misc.imresize(im, (im_height, im_height), interp='cubic') / 256.)
        hi_res_data[-1].append(im / 256.)
    return data, hi_res_data
#load_jpgs(im_height=28, data_dir='../labelfree/cushion')


def gen_data(group_size, data, labels, hi_res_data):
    if hi_res_data is None:
        hi_res_data = data
    if labels is None:
        labels = data
    while True:
        if not len(data) == len(labels) == len(hi_res_data):
            print len(data), len(labels), len(hi_res_data)
            assert len(data) == len(labels) == len(hi_res_data)
        idx = random.randrange(len(data))
        idx_len = len(data[idx])
        assert idx_len >= group_size
        start_id = random.randrange(idx_len - group_size + 1)
        yield (data[idx][start_id:start_id + group_size],
               hi_res_data[idx][start_id:start_id + group_size],
               [[y] for y in labels[idx][start_id:start_id + group_size]])

def std(x, axis):
    residual = x - tf.expand_dims(tf.reduce_mean(x, axis), axis)
    return tf.reduce_mean(tf.square(residual)) ** 0.5

def sphere(x, axis):
    residual = x - tf.expand_dims(tf.reduce_mean(x, axis), axis)
    std = tf.reduce_mean(tf.square(residual)) ** 0.5
    return (x - residual) / tf.maximum(std, 0.1)

def gen_data_batch(batch_size, gen_fn, gen_fn_args):
    data_source = gen_fn(*gen_fn_args)
    while True:
        batch_elems = []
        for i in range(batch_size):
            elems = data_source.next()
            for j, elem in enumerate(elems):
                if i == 0:
                    batch_elems.append(np.zeros([batch_size] + list(np.array(elem).shape)))
                batch_elems[j][i] = np.array(elem)
        yield batch_elems

class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)

@memoized
def image_cache(fname, im_height):
    img = misc.imread(fname)
    return misc.imresize(img, (im_height, im_height))
@memoized
def image_cache_seg(fname, im_height, im_width):
    img = misc.imread(fname)
    return img
#@memoized
def process_img_seg(img):
    #print img.shape
    _h = img.shape[0]
    _l = img.shape[1]
    mat = np.zeros((_h,_l,2))
    for i in xrange(_h):
      for j in xrange(_l):
        if img[i,j,0] > 200:
          mat[i,j,1] = 1
          mat[i,j,0] = 0
        else:
          mat[i,j,1] = 0
          mat[i,j,0] = 1
    return mat
def np_plot(y, imgs):
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    ax.plot(y)
    canvas.draw()       # draw the canvas, cache the renderer
    imdata = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

    #imdata = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    imdata = imdata.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    results = []
    new_size, concat_axis = (360, 540), 1
#     new_size, concat_axis = (360, 640), 0
    for img in imgs:
        plt_data = misc.imresize(imdata, new_size)
        img_r = misc.imresize(img, (360, 360))
        conc = np.concatenate((img_r, plt_data), axis=concat_axis)
        results.append(conc)
    return results

