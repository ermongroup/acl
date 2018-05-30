import os
from PIL import Image
import scipy.misc
import numpy as np
import imageio
import cv2

def extractFrames(inGif, outFolder):
    frame = Image.open(inGif)
    nframes = 0
    while frame:
        frame.save( '%s/%s-%02d.png' % (outFolder, os.path.basename(inGif), nframes ))
        nframes += 1
        try:
            frame.seek( nframes )
        except EOFError:
            break;
    return True
    

def gen_gif(output_dir, image_name, image_num, length = 5, encodes = None):
  print output_dir, image_name ,image_num
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  extractFrames('%s/%s%d.gif'%(output_dir,image_name, image_num), output_dir)
  extractFrames('%s/%s%d.gif'%(output_dir,image_name, image_num+1), output_dir)
  
  for i in xrange(length):
    a = scipy.misc.imread('%s/%s%d.gif-%d.png'%(output_dir,image_name,image_num,i))
    b = scipy.misc.imread('%s/%s%d.gif-%d.png'%(output_dir,image_name,image_num+1,i))
    c = np.concatenate((a,b),axis=1)
    scipy.misc.imsave('%s/%d.png'%(output_dir,i),c)
    
  images = []
  filenames = ['%s/%d.png'%(output_dir,i) for i in xrange(length)]
  for filename in filenames:
      images.append(imageio.imread(filename))
  if encodes != None:
      imageio.mimsave('%s/%d_%d.gif'%(output_dir,encodes,image_num+2), images)
  else:
      imageio.mimsave('%s/%d.gif'%(output_dir,image_num+2), images)
  
  for i in xrange(length):
    os.remove('%s/%s%d.gif-%d.png'%(output_dir,image_name,image_num,i))
    os.remove('%s/%s%d.gif-%d.png'%(output_dir,image_name,image_num+1,i))
  os.remove('%s/%s%d.gif'%(output_dir,image_name, image_num))
  os.remove('%s/%s%d.gif'%(output_dir,image_name, image_num+1))

def render(location, render_mode, mode, im_height, img, pck, truth):
    #img = np.zeros((im_height, im_height, 3))
    #img += np.random.random((im_height, im_height, 3)) * 0.1
    colors = [(255, 0, 0), (0, 255, 0), (0,0,255)]
    num_points = len(location) / 2
    mid = []
    
    for i in xrange(len(location)):
        #location[i] = (location[i] + 1) * 32.
        if i % 2 != 0:
            location[i] = 64. - location[i]
            truth[i] = 64. - truth[i]
            
    for i in xrange(3):
        tl = [int(location[2*i] - 1), int(location[2*i+1] - 1)]
        br = [int(location[2*i] + 1), int(location[2*i+1] + 1)]
        mid.append([int(location[2*i]), int(location[2*i+1])])
        cv2.rectangle(img, tuple(tl), tuple(br), colors[1], 1)
    cv2.line(img, tuple(mid[0]), tuple(mid[1]), colors[0], 1)
    cv2.line(img, tuple(mid[1]), tuple(mid[2]), colors[0], 1)
    
    for i in range(6):
        tl = [int(truth[2*i] - pck), int(truth[2*i+1] - pck)]
        br = [int(truth[2*i] + pck), int(truth[2*i+1] + pck)]
        cv2.rectangle(img, tuple(tl), tuple(br), colors[2], 1)
       
    mid_2 = []
    for i in xrange(3,6):
        tl = [int(location[2*i] - 1), int(location[2*i+1] - 1)]
        br = [int(location[2*i] + 1), int(location[2*i+1] + 1)]
        mid_2.append([int(location[2*i]), int(location[2*i+1])])
        cv2.rectangle(img, tuple(tl), tuple(br), colors[1], 1)
    cv2.line(img, tuple(mid_2[0]), tuple(mid_2[1]), colors[0], 1)
    cv2.line(img, tuple(mid_2[1]), tuple(mid_2[2]), colors[0], 1)
    
    cv2.line(img, tuple(mid[0]), tuple(mid_2[0]), colors[0], 1)
    return img