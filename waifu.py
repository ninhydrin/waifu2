import cPickle as pickle
from datetime import timedelta
import json
from multiprocessing import Pool
from Queue import Queue
import random
import sys
from threading import Thread
import time
import skimage.io
import skimage.transform
import numpy as np
import copy
import csv

from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F

class SRCNN(FunctionSet):
    insize = 64
    outputsize = 64-(9-1)-(1-1)-(5-1)
    def __init__(self):
        super(SRCNN, self).__init__(
            conv1=F.Convolution2D(3,64,9),
            conv2=F.Convolution2D(64,32,1),
            conv3=F.Convolution2D(32,3,5),
        )
    def forward(self, x_data, y_data, train=True):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))

        loss = F.mean_squared_error(h, t)

        return loss, h


#input setting
CONST_IMG_SIZE = 64
CONST_N_TRAIN = 10000
CONST_N_EVAL = 1000
CONST_PATH_TRAIN = "train.txt"
CONST_PATH_EVAL = "train.txt"
CONST_N_BATCH = 50
CONST_N_BATCH_EVAL = 250
CONST_N_EPOCH = 100000
CONST_GPU_ID = 0 #cpu:-1,gpu:0~
CONST_N_LOADER = 1

#initial setting
#craete model instance
model = SRCNN()

if CONST_GPU_ID >= 0:
    cuda.init(CONST_GPU_ID)
    model.to_gpu()

# Setup optimizer
optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model.collect_parameters())

# Prepare dataset
def load_image_list(path):
    pathlist = []
    csvf = open(path)
    reader = csv.reader(csvf)
    for line in reader:
        pathlist.append(line[0])
    csvf.close()
    return pathlist

train_list = load_image_list(CONST_PATH_TRAIN)
val_list   = load_image_list(CONST_PATH_EVAL)

# ------------------------------------------------------------------------------
# This example consists of three threads: data feeder, logger and trainer. These
# communicate with each other via Queue.
data_q = Queue(maxsize=1)
res_q  = Queue()

# Data loading routine
def read_image(path):
    image = skimage.io.imread(path,as_gray=False)
    v_rand = random.randint(0,3)
    if v_rand == 0:
        image = image[:,:,:]
    elif v_rand == 1:
        image = image[::-1,:,:]
    elif v_rand == 2:
        image = image[:,::-1,:]
    elif v_rand == 3:
        image = image[::-1,::-1,:]
    #create output data
    output_top = (CONST_IMG_SIZE- model.outputsize) / 2
    output_bottom = CONST_IMG_SIZE- output_top
    image_out = image[output_top:output_bottom,output_top:output_bottom,:]
    image_out = image_out.transpose(2, 0, 1).astype(np.float32)/255.
    #create input coarse data
    image_in = skimage.transform.rescale(image,0.5)
    image_in = skimage.transform.rescale(image_in,2.)#/255 already 0~1
    image_in = image_in.transpose(2, 0, 1).astype(np.float32)
    
    return image_in,image_out

def eval_image_show(y,predict,name):
    y_convert = y.transpose(1, 2, 0)
    predict_convert = predict.transpose(1, 2, 0)
    #create coarse data
    x_convert = skimage.transform.rescale(y_convert,0.5)
    x_convert = skimage.transform.rescale(x_convert,2.)
    
    #convert for output
    y_convert[np.where(y_convert>1.0)] = 1.0
    x_convert[np.where(x_convert>1.0)] = 1.0
    predict_convert[np.where(predict_convert>1.0)] = 1.0
    y_convert = (y_convert*255).astype(np.uint8)
    x_convert = (x_convert*255).astype(np.uint8)
    predict_convert = (predict_convert*255).astype(np.uint8)

    skimage.io.imsave(name+"in.png",x_convert)
    skimage.io.imsave(name+"out.png",y_convert)
    skimage.io.imsave(name+"pre.png",predict_convert)

# Data feeder
def feed_data():
    i     = 0
    count = 0

    x_batch = np.ndarray((CONST_N_BATCH, 3, CONST_IMG_SIZE, CONST_IMG_SIZE), dtype=np.float32)
    y_batch = np.ndarray((CONST_N_BATCH,3, model.outputsize, model.outputsize), dtype=np.float32)
    val_x_batch = np.ndarray((CONST_N_BATCH_EVAL, 3, CONST_IMG_SIZE, CONST_IMG_SIZE), dtype=np.float32)
    val_y_batch = np.ndarray((CONST_N_BATCH_EVAL,3, model.outputsize, model.outputsize), dtype=np.float32)

    batch_pool     = [None] * CONST_N_BATCH
    val_batch_pool = [None] * CONST_N_BATCH_EVAL
    pool           = Pool(CONST_N_LOADER)
    data_q.put('train')
    for epoch in xrange(1, 1 + CONST_N_EPOCH):
        print >> sys.stderr, 'epoch', epoch
        print >> sys.stderr, 'learning rate', optimizer.lr
        perm = np.random.permutation(len(train_list))
        for idx in perm:
            path = train_list[idx]
            batch_pool[i] = pool.apply_async(read_image, args = (path, ),)
            i += 1

            if i == CONST_N_BATCH:
                for j, x in enumerate(batch_pool):
                    x_batch[j],y_batch[j] = x.get()
                data_q.put((x_batch.copy(), y_batch.copy()))
                i = 0

            count += 1
            if count % 1000 == 0:
                data_q.put('val')
                j = 0
                for path in val_list:
                    val_batch_pool[j] = pool.apply_async(read_image, args = (path, ),)
                    j += 1

                    if j == CONST_N_BATCH_EVAL:
                        for k, x in enumerate(val_batch_pool):
                            val_x_batch[k],val_y_batch[k] = x.get()
                        data_q.put((val_x_batch.copy(), val_y_batch.copy()))
                        j = 0
                data_q.put('train')

        optimizer.lr *= 0.97
    pool.close()
    pool.join()
    data_q.put('end')

# Logger
def log_result():
    train_count = 0
    train_cur_loss = 0
    begin_at = time.time()
    val_begin_at = None
    best_loss = np.Infinity
    while True:
        result = res_q.get()
        if result == 'end':
            print >> sys.stderr, ''
            break
        elif result == 'train':
            print >> sys.stderr, ''
            train = True
            if val_begin_at is not None:
                begin_at += time.time() - val_begin_at
                val_begin_at = None
            continue
        elif result == 'val':
            print >> sys.stderr, ''
            train = False
            val_count = val_loss = 0
            val_begin_at = time.time()
            continue

        loss, y,predict,tmp_model = result
        if train:
            train_count += 1
            duration     = time.time() - begin_at
            throughput   = train_count * CONST_N_BATCH / duration
            sys.stderr.write(
                '\rtrain {} updates ({} samples) time: {} ({} images/sec)'
                .format(train_count, train_count * CONST_N_BATCH,
                        timedelta(seconds=duration), throughput))

            train_cur_loss += loss
            if train_count % 20 == 0:
                y_tmp = y[0,:,:,:]
                pre_tmp = predict[0,:,:,:]
                eval_image_show(y=y_tmp,predict=pre_tmp,name="train_"+str(train_count)+"_")
                mean_loss  = train_cur_loss / 20
                print >> sys.stderr, ''
                print json.dumps({'type': 'train', 'iteration': train_count,'loss': mean_loss})
                sys.stdout.flush()
                train_cur_loss = 0
        else:
            val_count  += CONST_N_BATCH_EVAL
            duration    = time.time() - val_begin_at
            throughput  = val_count / duration
            sys.stderr.write(
                '\rval   {} batches ({} samples) time: {} ({} images/sec)'
                .format(val_count / CONST_N_BATCH_EVAL, val_count,
                        timedelta(seconds=duration), throughput))

            val_loss += loss
            if val_count == CONST_N_EVAL:
                y_tmp = y[0,:,:,:]
                pre_tmp = predict[0,:,:,:]
                eval_image_show(y=y_tmp,predict=pre_tmp,name="eval_"+str(train_count)+"_")
                mean_loss  = val_loss * CONST_N_BATCH_EVAL / CONST_N_EVAL
                if(best_loss > mean_loss):
                    filename_model = "model_" + str(mean_loss) + "_" + str(train_count)
                    pickle.dump(tmp_model, open(filename_model, 'wb'), -1)
                    best_loss = mean_loss
                print >> sys.stderr, ''
                print json.dumps({'type': 'val', 'iteration': train_count,'loss': mean_loss})
                sys.stdout.flush()

# Trainer
def train_loop():
    while True:
        while data_q.empty():
            time.sleep(0.1)
        inp = data_q.get()
        if inp == 'end':  # quit
            res_q.put('end')
            break
        elif inp == 'train':  # restart training
            res_q.put('train')
            train = True
            continue
        elif inp == 'val':  # start validation
            res_q.put('val')
            pickle.dump(model, open('model', 'wb'), -1)
            train = False
            continue

        x, y = inp
        if CONST_GPU_ID >= 0:
            x = cuda.to_gpu(x)
            y = cuda.to_gpu(y)

        if train:
            optimizer.zero_grads()
            loss, predict = model.forward(x, y)
            loss.backward()
            optimizer.update()
        else:
            loss, predict = model.forward(x, y, train=False)
        tmp_model = copy.deepcopy(model)
        tmp_model.to_cpu()
        res_q.put((float(cuda.to_cpu(loss.data)),
                   cuda.to_cpu(y),
                   cuda.to_cpu(predict.data),
                   tmp_model))
        del loss, predict, x, y
"""
# Invoke threads
feeder = Thread(target=feed_data)
feeder.daemon = True
feeder.start()
logger = Thread(target=log_result)
logger.daemon = True
logger.start()

train_loop()
feeder.join()
logger.join()

# Save final model
pickle.dump(model, open('model', 'wb'), -1)
"""
