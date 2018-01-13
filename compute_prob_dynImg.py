# the reference of dynamic images: http://www.egavves.com/data/cvpr2016bilen.pdf


import sys
# correctly install the caffe and pycaffe, and set up the path
sys.path.append('../../python')
import caffe
import numpy as np
import pdb

dynImgValidationProbabilities = {}
dynImgValidationFeatures = {}

prototxt_file = 'dynImgs.prototxt'
model_file_dyn = 'dynImgs.caffemodel'
if not os.access(model_file_dyn, os.W_OK):
    weight_url = 'http://moments.csail.mit.edu/moments_models/' + model_file_dyn
    os.system('wget ' + weight_url)


def init_model(prototxt_file, model_file):
    caffe.set_mode_gpu()
    caffe.set_device(0)

    net = caffe.Net(prototxt_file, model_file, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.array([111,111,111]))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    net.blobs['data'].reshape(1,3,224,224)

    return net, transformer

def generate_image_feats(images, mode):
      ims = np.array([transformer_dyn.preprocess('data', im) for im in images], dtype=np.float32)
      net_dyn.blobs['data'].reshape(*ims.shape)
      net_dyn.blobs['data'].data[...] = ims
      out = net_dyn.forward()
      preds = out['probs'][0:ims.shape[0]]
      return preds

net_dyn, transformer_dyn = init_model(prototxt_file, model_file_dyn)

im = 'sample_dynImg_stacking.jpg'
images_dyn= [caffe.io.load_image(im)]
preds_dyn = np.squeeze(generate_image_feats(images_dyn, 'dyn'))
pdb.set_trace()

