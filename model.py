# -*- coding: utf-8 -*-
"""
Created on July  9 20:16:49 2018

@author: Qingsen Yan
"""

import numpy as np
import os
import os.path as osp

import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import reporter

global iteratetime
iteratetime = 1

cuda.check_cuda_available()
xp = cuda.cupy


class Model(chainer.Chain):
    def __init__(self):
        super(Model, self).__init__(

            conv1=L.Convolution2D(3, 32, 7, pad=3),
            conv2=L.Convolution2D(32, 32, 7, pad=3),

            conv3=L.Convolution2D(32, 64, 5, pad=2),
            conv4=L.Convolution2D(64, 64, 5, pad=2),

            conv5=L.Convolution2D(64, 128, 3, pad=1),
            conv6=L.Convolution2D(128, 128, 3, pad=1),

            conv7=L.Convolution2D(128, 256, 3, pad=1),
            conv8=L.Convolution2D(256, 256, 3, pad=1),

            conv9=L.Convolution2D(256, 512, 3, pad=1),
            conv10=L.Convolution2D(512, 512, 3, pad=1),

            # gradient layer
            conv2_1=L.Convolution2D(3, 32, 7, pad=3),
            conv2_2=L.Convolution2D(32, 32, 7, pad=3),

            conv2_3=L.Convolution2D(32, 64, 5, pad=2),
            conv2_4=L.Convolution2D(64, 64, 5, pad=2),

            conv2_5=L.Convolution2D(64, 128, 3, pad=1),
            conv2_6=L.Convolution2D(128, 128, 3, pad=1),

            conv2_7=L.Convolution2D(128, 256, 3, pad=1),
            conv2_8=L.Convolution2D(256, 256, 3, pad=1),

            conv2_9=L.Convolution2D(256, 512, 3, pad=1),
            conv2_10=L.Convolution2D(512, 512, 3, pad=1),

            # region
            convP1=L.Convolution2D(256, 216, 1),
            convP2=L.Convolution2D(256, 216, 1),

            fc1_a=L.Linear(1024, 512),
            fc2_a=L.Linear(512, 1),

            fc1=L.Linear(432, 512),
            fc2=L.Linear(512, 1),
        )

    def __call__(self, x_data, y_data, train=True, n_patches=32):
        if not isinstance(x_data, Variable):
            length = x_data.shape[0]
            x1 = Variable(x_data[0:length:2])  
            x2 = Variable(x_data[1:length:2])  
        else:
            x = x_data
            x_data = x.data

        self.n_images = 1
        self.n_patches = x_data.shape[0] / 2
        self.n_patches_per_image = self.n_patches / self.n_images

        h = F.relu(self.conv1(x1))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv7(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.convP1(h))
        h = F.max_pooling_2d(h, int(x_data.shape[2]/8))


        ########## gradient stream
        h1 = F.relu(self.conv2_1(x2))
        h1 = F.max_pooling_2d(h1, 2)

        h1 = F.relu(self.conv2_3(h1))
        h1 = F.max_pooling_2d(h1, 2)

        h1 = F.relu(self.conv2_5(h1))
        h1 = F.max_pooling_2d(h1, 2)

        h1 = F.relu(self.conv2_7(h1))
        h1 = F.max_pooling_2d(h1, 2)

        h1 = F.relu(self.convP2(h1))
        h1 = F.max_pooling_2d(h1, int(x_data.shape[2]/8))

        #
        h = F.concat((h, h1), axis=1)

        h_ = h
        self.h = h_

        h = F.dropout(F.relu(self.fc1(h_)), ratio=0.5)
        h = self.fc2(h)


        #########################
        a_patch = xp.ones_like(h.data)
        t = xp.repeat(y_data[0:length:2], 1) 
        t_patch = xp.array(t.astype(np.float32))

        self.average_loss(h, a_patch, t_patch)

        model = self

        self._save_model(model)

        if train:
            reporter.report({'loss': self.loss}, self)
            # print 'self.lose:', self.loss.data
            return self.loss
        else:
            return self.loss, self.y

    def forward(self, x_data, x_data1, y_data, train=True, n_patches=32):

        if not isinstance(x_data, Variable):
            x1 = Variable(x_data)
            x2 = Variable(x_data1)
        else:
            x = x_data
            x_data = x.data


        h = F.relu(self.conv1(x1))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv7(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.convP1(h))
        h = F.max_pooling_2d(h, int(x_data.shape[2]/8))

        ########## gradient stream
        h1 = F.relu(self.conv2_1(x2))
        h1 = F.max_pooling_2d(h1, 2)

        h1 = F.relu(self.conv2_3(h1))
        h1 = F.max_pooling_2d(h1, 2)

        h1 = F.relu(self.conv2_5(h1))
        h1 = F.max_pooling_2d(h1, 2)

        h1 = F.relu(self.conv2_7(h1))
        h1 = F.max_pooling_2d(h1, 2)

        h1 = F.relu(self.convP2(h1))
        h1 = F.max_pooling_2d(h1, int(x_data.shape[2]/8))

        #  # # # # # #
        h = F.concat((h, h1), axis=1)

        h_ = h
        self.h = h_

        h = F.dropout(F.relu(self.fc1(h_)), ratio=0.5)
        h = self.fc2(h)

        return  F.sum(h.data) / len(h)




    def average_loss(self, h, a, t):

        self.loss = F.sum(abs(h - F.reshape(t, (-1, 1))))
        self.loss /= self.n_patches
        if self.n_images > 1:
            h = F.split_axis(h, self.n_images, 0)
            a = F.split_axis(a, self.n_images, 0)
        else:
            h, a = [h], [a]

        self.y = h
        self.a = a

    def _save_model(self, model):
        global iteratetime
        iteratetime = iteratetime + 1
        self.out = 'Res/'
        out_model_dir = osp.join(self.out, 'models')
        if not osp.exists(out_model_dir):
            os.makedirs(out_model_dir)
        model_name = 'My'
        out_model = osp.join(out_model_dir, '%s_iter%08d.model' %
                             (model_name, iteratetime))
        if (iteratetime % 100 == 0):
            chainer.serializers.save_hdf5(out_model, model)
