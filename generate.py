import cv2
import torch
import tqdm
import os
import numpy as np
from scipy.misc import imsave

import data.eecs442_challenge.ref as ds
import train as net

def evaluate():
    test = ds.get_test_set()
    func, config = net.init()

    for idx in test:
        img = ds.load_image(idx, False)
        img = img / 255
        img = img.astype(np.float32)

        output = (-1, config, phase='inference', imgs=img)
        pred = torch.FloatTensor(output['preds'][0][:, -1])
        pred = pred / torch.norm(pred, 2, 3)

        pred = pred * 255
        imsave('./save/pred_{}.png'.format(idx), pred.astype(np.uint8))


if __name__=='__main__':
    evaluate()
