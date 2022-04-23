import os
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms.functional as F
from util import pose_utils
from PIL import Image
import pandas as pd
import torch
import cv2

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def initialize(self, opt):
        self.opt = opt
        self.image_dir, self.bone_file, self.name_pairs = self.get_paths(opt)
        size = len(self.name_pairs)
        self.dataset_size = size
        self.segdir = opt.dirSem
        if isinstance(opt.load_size, int):
            self.load_size = (opt.load_size, opt.load_size)
        else:
            self.load_size = opt.load_size

        transform_list=[]
        # transform_list.append(transforms.Resize(size=self.load_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
        self.trans = transforms.Compose(transform_list) 

        self.annotation_file = pd.read_csv(self.bone_file, sep=':')
        self.annotation_file = self.annotation_file.set_index('name')

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        assert False, "A subclass of MarkovAttnDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths

    def __getitem__(self, index):

        P1_name, P2_name = self.name_pairs[index]
        P1_path = os.path.join(self.image_dir, P1_name) # person 1
        P2_path = os.path.join(self.image_dir, P2_name) # person 2

        P1_img = Image.open(P1_path).convert('RGB')
        P2_img = Image.open(P2_path).convert('RGB')

        P1_img = F.resize(P1_img, self.load_size)
        P2_img = F.resize(P2_img, self.load_size)

        BP1 = self.obtain_bone(P1_name)
        P1 = self.trans(P1_img)

        BP2 = self.obtain_bone(P2_name)
        P2 = self.trans(P2_img)

        SP1_name = self.split_name(P1_name, 'semantic_merge3')
        SP1_path = os.path.join(self.segdir, SP1_name)
        SP1_path = SP1_path[:-4] + '.npy'
        SP1_data = np.load(SP1_path)
        SP1 = np.zeros((8, int(self.load_size[0]), int(self.load_size[0])), dtype='float32')
        for id in range(8):
            SP1[id] = cv2.resize((SP1_data == id).astype('float32'), self.load_size)

        SP2_name = self.split_name(P2_name, 'semantic_merge3')
        SP2_path = os.path.join(self.segdir, SP2_name)
        SP2_path = SP2_path[:-4] + '.npy'
        SP2_data = np.load(SP2_path)
        SP2 = np.zeros((8, int(self.load_size[0]), int(self.load_size[0])), dtype='float32')
        for id in range(8):
            SP2[id] = cv2.resize((SP2_data == id).astype('float32'), self.load_size)

        return {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2, 'SP1': SP1, 'SP2': SP2,
                'P1_path': P1_name, 'P2_path': P2_name}

    def obtain_bone(self, name):

        string = self.annotation_file.loc[name]
        array = pose_utils.load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])
        pose  = pose_utils.cords_to_map(array, self.load_size, (256,176), None)
        pose = np.transpose(pose,(2, 0, 1))
        pose = torch.Tensor(pose)
        return pose

    def split_name(self,str,type):

        list = []
        list.append(type)
        if (str[len('fashion'):len('fashion') + 2] == 'WO'):
            lenSex = 5
        else:
            lenSex = 3
        list.append(str[len('fashion'):len('fashion') + lenSex])
        idx = str.rfind('id0')
        list.append(str[len('fashion') + len(list[1]):idx])
        id = str[idx:idx + 10]
        list.append(id[:2]+'_'+id[2:])
        pose = str[idx + 10:]
        list.append(pose[:4]+'_'+pose[4:])

        head = ''
        for path in list:
            head = os.path.join(head, path)
        return head

    def __len__(self):
        return self.dataset_size

    def name(self):
        assert False, "A subclass of BaseDataset must override self.name"