# -*- coding: utf-8 -*-
"""
Created on July  20 11:04:15 2018

"""
# Qingsen Yan
# to evaluate all the images from the path in txt file, and save results

import os
import time
import numpy as np
import argparse

import chainer
import six
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers

import cv2
from PIL import Image
from sklearn.feature_extraction.image import extract_patches

from model import Model

import os

# os.environ['CUDA_VISIBLE_DEVICES']='0'
# ./Res/models/My_iter00515300.model

parser = argparse.ArgumentParser(description='evaluate the proposed model')
parser.add_argument('--model', '-m', default='models/LIVE_test.model',
                    help='path to the trained model')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID')

args = parser.parse_args()
patches_per_img = 512 
model = Model()
cuda.cudnn_enabled = True
cuda.check_cuda_available()
xp = cuda.cupy
serializers.load_hdf5(args.model, model)
model.to_gpu()
patchSize = 128


test_label_path = 'data_list/test.txt'
test_img_path = 'data/live/'
test_Graimg_path = 'data/live_grad/'

result_ptr = open('result/result_score.txt', 'wt')
with open(test_label_path, 'rt') as f:
    for line in f:
        # line = line.strip()                                 #get test image name
        line, la = line.strip().split()  # for debug

        tic = time.time()
        full_path = os.path.join(test_img_path, line)
        Grafull_path = os.path.join(test_Graimg_path, line)

        f = Image.open(full_path)
        Graf = Image.open(Grafull_path)
        img = np.asarray(f, dtype=np.float32)
        Gra = np.asarray(Graf, dtype=np.float32)
        img = img.transpose(2, 0, 1)
        Gra = Gra.transpose(2, 0, 1)

        img1 = np.zeros((1, 3, Gra.shape[1], Gra.shape[2]))
        img1[0, :, :, :] = img
        Gra1 = np.zeros((1, 3, Gra.shape[1], Gra.shape[2]))
        Gra1[0, :, :, :] = Gra


        patches = extract_patches(img, (3, patchSize, patchSize), patchSize)
        Grapatches = extract_patches(Gra, (3, patchSize, patchSize), patchSize)

        test_patches = []
        X = patches.reshape((-1, 3, patchSize, patchSize)) 
        GraX = Grapatches.reshape((-1, 3, patchSize, patchSize))

        y = []
        t = xp.zeros((1, 1), np.float32)
        # t[0][0] = int(la)                                  #for debug

        X_batch = np.zeros(X.shape)
        for i in range(len(X_batch)):
            X_batch[i] = X[int(i)]
        X_batch = X_batch[:patches_per_img]
        X_batch = xp.array(X_batch.astype(np.float32))

        GX_batch = np.zeros(GraX.shape)
        for i in range(len(GX_batch)):
            GX_batch[i] = GraX[int(i)]
        GX_batch = GraX[:GraX.shape[0]]
        GX_batch = xp.array(GX_batch.astype(np.float32))

        score = model.forward(X_batch, GX_batch, t, False, X_batch.shape[0])


        result_ptr.write('{:f}\n'.format(cuda.to_cpu(score.data)))

result_ptr.close()

