import cv2
import torch
import tqdm
import os
import numpy as np
from scipy.misc import imsave, imresize

import data.eecs442_challenge.ref as ds
import train as net

def evaluate():
    #test = ds.get_test_set()
    test = range(10)
    func, config = net.init()

    for idx in test:
        img = ds.load_image(idx, False)
        img = img / 255
        img = img.astype(np.float32)
        img = img[None, :, :, :]
        img = torch.FloatTensor(img)
        output = func(-1, config, phase='inference', imgs=img)
        pred = torch.FloatTensor(output['preds'][0][:, -1])
        pred = pred[0, :, :, :]
        pred = pred.permute(1, 2, 0)
        for i in range(3):
            pred[:,:,i] = pred[:,:,i] / torch.norm(pred, 2, 2)

        pred = pred * 255
        pred = pred.numpy()
        pred = pred.astype(np.uint8)
        pred = imresize(pred, (128, 128))
        imsave('./save/pred_{}.png'.format(idx), pred)


if __name__=='__main__':
    evaluate()
