import os.path

import torch
from glob2 import glob
from torch.utils.data.dataset import Dataset
from utils.him import fitsread
import numpy as np


class AIADataset(Dataset):

    def __init__(self, dataroot):
        self.paths = sorted(glob(os.path.join(dataroot, '*.fits')))
        # self.wave = [171, 193, 211, 335, 131, 94]

    def __getitem__(self, index):
        path = self.paths[index]
        name = os.path.basename(path)
        fit_data = fitsread(path)[0]
        in_idx = [0, 1, 2]
        out_idx = [3, 4, 5]
        inputs = np.stack([fit_data[i] for i in in_idx])
        outputs = np.stack([fit_data[i] for i in out_idx])

        inputs_t = torch.from_numpy(inputs)
        labels_t = torch.from_numpy(outputs)

        # return inputs_t, labels_t, name
        return {'inputs': inputs_t, 'outputs': labels_t, 'name': name}

    def __len__(self):
        return len(self.paths)


if __name__ == '__main__':
    dataroot = '/Volumes/BLBL/datasets/AIA/proce_and_crop_comp_xrt_2012'
    dataset = AIADataset(dataroot)
    data = dataset.__getitem__(1)
    print(data)
