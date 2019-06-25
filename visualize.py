#!/usr/bin/env python

import os

import numpy as np
from PIL import Image

import chainer
import chainer.backends.cuda
from chainer import Variable


def out_generated_image(gen, dis, rows, cols, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(n_images)))
        with chainer.using_config('train', False):
            x = gen(z)
        x = chainer.backends.cuda.to_cpu(x.data)
        np.random.seed()

        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
        print(x.shape)
        _, _, H, W = x.shape

        for i in range(len(x)):
            x_i = x[i]
            x_i = x_i.reshape((1, 1, 3, H, W))
            x_i = x_i.transpose(0, 3, 1, 4, 2)
            #変えた H,W→32,32
            x_i = x_i.reshape((32, 32, 3))
        
            #x = x.reshape((rows, cols, 3, H, W))
            #x = x.transpose(0, 3, 1, 4, 2)
            #x = x.reshape((rows * H, cols * W, 3))

            preview_dir = '{}/preview'.format(dst)
            preview_path = preview_dir +\
                '/image{:0>8}_{}.png'.format(trainer.updater.iteration, i)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            Image.fromarray(x_i).save(preview_path)
    return make_image
