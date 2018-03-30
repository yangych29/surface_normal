import cv2
import sys
import os
import torch
import numpy as np
import torch.utils.data
from scipy.misc import imresize, imsave

from utils.misc import get_transform, kpt_affine


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, ds, index):
        self.input_res = config['train']['input_res']
        self.output_res = config['train']['output_res']

        self.ds = ds
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.loadImage(self.index[idx % len(self.index)])

    def loadImage(self, idx):
        ds = self.ds
        # load image
        #img = (ds.load_image(idx) / 255 - 0.5) * 2
        img = ds.load_image(idx)

        # data argumentation
        #height, width, _ = img.shape
        #center = np.array((width/2, height/2))
        #scale = max(height, width)/200
        #inp_res = self.input_res
        #res = (inp_res, inp_res)
        #aug_rot = (np.random.random() * 2 - 1) * 30.
        #aug_rot = 0.0
        #aug_scale = np.random.random() * (1.25 - 0.75) + 0.75
        #aug_scale = 1.0
        #scale *= aug_scale
        #dx = np.random.randint(-40 * scale, 40 * scale)/center[0]
        #dy = np.random.randint(-40 * scale, 40 * scale)/center[1]
        #center[0] += dx * center[0]
        #center[1] += dy * center[1]
        #trans1 = get_transform(center, scale, (self.input_res, self.input_res), aug_rot)[:2]
        #trans2 = get_transform(center, scale, (self.output_res, self.output_res), aug_rot)[:2]

        # for img
        #img = cv2.warpAffine(img.astype(np.uint8), trans1, (self.input_res, self.input_res))
        img = (img / 255 - 0.5) * 2

        # for mask
        mask = ds.load_mask(idx)
        #mask = cv2.warpAffine(mask.astype(np.uint8), trans2, (self.output_res, self.output_res))
        mask = imresize(mask, (self.output_res, self.output_res))
        mask = mask / 255
        mask[mask >= 0.5] = 1.0
        mask[mask < 0.5] = 0.0

        # ground true
        gt = ds.load_gt(idx)
        #gt = cv2.warpAffine(gt.astype(np.uint8), trans2, (self.output_res, self.output_res))
        gt = imresize(gt, (self.output_res, self.output_res))
        gt = (gt / 255 - 0.5) * 2

        return img.astype(np.float32), mask.astype(np.float32), gt.astype(np.float32)

    """
    def preprocess(self, data):
        # random hue and saturation
        data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV);
        delta = (np.random.random() * 2 - 1) * 0.2
        data[:, :, 0] = np.mod(data[:,:,0] + (delta * 360 + 360.), 360.)

        delta_sature = np.random.random() + 0.5
        data[:, :, 1] *= delta_sature
        data[:,:, 1] = np.maximum( np.minimum(data[:,:,1], 1), 0 )
        data = cv2.cvtColor(data, cv2.COLOR_HSV2RGB)

        # adjust brightness
        delta = (np.random.random() * 2 - 1) * 0.3
        data += delta

        # adjust contrast
        mean = data.mean(axis=2, keepdims=True)
        data = (data - mean) * (np.random.random() + 0.5) + mean
        data = np.minimum(np.maximum(data, 0), 1)
        #cv2.imwrite('x.jpg', (data*255).astype(np.uint8))
        return data
    """


def init(config):
    batchsize = config['train']['batchsize']
    current_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_path)
    import ref as ds
    ds.init()

    train, valid = ds.setup_val_split()
    dataset = { key: Dataset(config, ds, data) for key, data in zip( ['train', 'valid'], [train, valid] ) }

    use_data_loader = config['train']['use_data_loader']

    loaders = {}
    for key in dataset:
        loaders[key] = torch.utils.data.DataLoader(dataset[key], batch_size=batchsize, shuffle=True, num_workers=config['train']['num_workers'], pin_memory=False)

    def gen(phase):
        batchsize = config['train']['batchsize']
        batchnum = config['train']['{}_iters'.format(phase)]
        loader = loaders[phase].__iter__()
        for i in range(batchnum):
            imgs, masks, gts = next(loader)
            yield {
                'imgs': imgs,
                'masks': masks,
                'gts': gts,
            }

    return lambda key: gen(key)

if __name__ == "__main__":
    cf = {
        'inference': {
            'nstack': 8,
            'inp_dim': 256,
            'oup_dim': 3,
            'num_parts': 3,
            'increase': 128,
            'keys': ['imgs']
        },
        'train': {
            'batchsize': 30,
            'input_res': 128,
            'output_res': 64,
            'train_iters': 600,
            'valid_iters': 0,
            'num_workers': 2,
            'use_data_loader': True,
        },
    }
    func = init(cf)('train')
    for data in func:
        for i in range(5):
            imsave("{}_color.png".format(i), data['imgs'][i])
            imsave("{}_mask.png".format(i), data['masks'][i])
            imsave("{}_normal.png".format(i), data['gts'][i])
        exit(0)
