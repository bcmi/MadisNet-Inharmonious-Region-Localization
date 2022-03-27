import os.path
import os
import torch
import torchvision.transforms.functional as tf
from dataset.base_dataset import BaseDataset, get_transform
#from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as transforms
import random
import torch.nn.functional as F
import copy

class MultiObjectsIhdDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--is_train', type=bool, default=True, help='whether in the training phase')
        parser.set_defaults(max_dataset_size=float("inf"), new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.image_paths = []
        self.opt = copy.copy(opt)
        self.phase = opt.phase
        
        if opt.phase=='train':
            # print('loading training file: ')
            self.trainfile = os.path.join(opt.dataset_root,'released_train_le50.txt')
            self.keep_background_prob = 0.05 # 0.05
            with open(self.trainfile,'r') as f:
                for line in f.readlines():
                    self.image_paths.append(line.rstrip())
        elif opt.phase == 'val' or opt.phase == 'test':
            print('loading {} file'.format(opt.phase))
            self.keep_background_prob = -1
            self.trainfile = os.path.join(opt.dataset_root,'released_{}_le50.txt'.format('test'))
            with open(self.trainfile,'r') as f:
                for line in f.readlines():
                    self.image_paths.append(line.rstrip())
                    
        self.transform = get_transform(opt)
        self.input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)
                )
            ])
        
        # avoid the interlock problem of the opencv and dataloader
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

    def __getitem__(self, index):
        sample = self.get_sample(index)
        self.check_sample_types(sample)
        sample = self.augment_sample(sample)

        comp = self.input_transform(sample['image'])
        real = self.input_transform(sample['real'])
        mask = sample['mask'][np.newaxis, ...].astype(np.float32)
        mask = np.where(mask > 0.5, 1, 0).astype(np.uint8)

        output = {
            'comp': comp,
            'mask': mask,
            'real': real,
            'img_path':sample['img_path']
        }
        return output

    def check_sample_types(self, sample):
        assert sample['comp'].dtype == 'uint8'
        if 'real' in sample:
            assert sample['real'].dtype == 'uint8'

    def augment_sample(self, sample):
        if self.transform is None:
            return sample
        #print(self.transform.additional_targets.keys())
        additional_targets = {target_name: sample[target_name]
                              for target_name in self.transform.additional_targets.keys()}

        valid_augmentation = False
        while not valid_augmentation:
            aug_output = self.transform(image=sample['comp'], **additional_targets)
            valid_augmentation = self.check_augmented_sample(sample, aug_output)

        for target_name, transformed_target in aug_output.items():
            #print(target_name,transformed_target.shape)
            sample[target_name] = transformed_target

        return sample

    def check_augmented_sample(self, sample, aug_output):
        if self.keep_background_prob < 0.0 or random.random() < self.keep_background_prob:
            return True
        return aug_output['mask'].sum() > 10

    def get_sample(self, index):
        fn = self.image_paths[index].split('.')[0]
        composite_path = os.path.join(self.opt.dataset_root, 'composite_images', fn+'.jpg')
        mask_path = os.path.join(self.opt.dataset_root, 'masks', fn+'.png')
        target_path = os.path.join(self.opt.dataset_root, 'real_images', fn.split('_')[0]+'.jpg')
        
        comp = cv2.imread(composite_path)
        comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
        real = cv2.imread(target_path)
        
        real = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        mask = mask[:, :, 0].astype(np.float32) / 255.
        
        return {'comp': comp, 'mask': mask, 'real': real,'img_path':composite_path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
