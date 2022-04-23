import os.path
from data.base_dataset import BaseDataset
import pandas as pd

class FashionDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.set_defaults(load_size=256)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(old_size=(256, 176))
        parser.set_defaults(structure_nc=18)
        parser.set_defaults(image_nc=3)
        parser.set_defaults(display_winsize=256)

        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = opt.phase
        if not opt.is_texturet or phase == 'train':
            pairLst = os.path.join(root, 'fasion-resize-pairs-%s.csv' % phase)
            name_pairs = self.init_categories(pairLst)
        else:
            if opt.isFig:
                pairLst = os.path.join(root, 'fasion-tryon-pairs-test-fig.csv')
            else:
                pairLst = os.path.join(root, 'fasion-tryon-pairs-%s.csv' % phase)
            name_pairs = self.init_categories(pairLst)

        image_dir = os.path.join(root, '%s' % phase)
        bonesLst = os.path.join(root, 'fasion-resize-annotation-%s.csv' % phase)

        return image_dir, bonesLst, name_pairs

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        size = len(pairs_file_train)
        pairs = []
        print('Loading data pairs ...')
        for i in range(size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            pairs.append(pair)

        print('Loading data pairs finished ...')  
        return pairs    

    def name(self):
        return "FashionDataset"

                