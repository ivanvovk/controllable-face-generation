import os
import pandas as pd
import numpy as np
import torch
from PIL import Image


class CelebaDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 usecols=['Black_Hair', 'Blond_Hair'],
                 attr_file='/root/data/CelebA/list_attr_celeba.txt',
                 dataset_location='/root/data/CelebA/img_align_celeba/', 
                 transforms=None):
        """
        Avaliable attributes
        [0, '5_o_Clock_Shadow'], [1, 'Arched_Eyebrows'],  [2, 'Attractive'],  [3, 'Bags_Under_Eyes'],  [4, 'Bald'],
        [5, 'Bangs'], [6, 'Big_Lips'], [7, 'Big_Nose'], [8, 'Black_Hair'], [9, 'Blond_Hair'], [10, 'Blurry'],
        [11, 'Brown_Hair'], [12, 'Bushy_Eyebrows'], [13, 'Chubby'], [14, 'Double_Chin'], [15, 'Eyeglasses'],
        [16, 'Goatee'], [17, 'Gray_Hair'], [18, 'Heavy_Makeup'], [19, 'High_Cheekbones'], [20, 'Male'],
        [21, 'Mouth_Slightly_Open'], [22, 'Mustache'], [23, 'Narrow_Eyes'], [24, 'No_Beard'], [25, 'Oval_Face'],
        [26, 'Pale_Skin'], [27, 'Pointy_Nose'], [28, 'Receding_Hairline'], [29, 'Rosy_Cheeks'], [30, 'Sideburns'],
        [31, 'Smiling'], [32, 'Straight_Hair'], [33, 'Wavy_Hair'], [34, 'Wearing_Earrings'], [35, 'Wearing_Hat'],
        [36, 'Wearing_Lipstick'], [37, 'Wearing_Necklace'], [38, 'Wearing_Necktie'], [39, 'Young']
        """
        super(CelebaDataset, self).__init__()
        self.dataset_location = dataset_location
        self.attr_file = attr_file
        self.usecols = usecols
        self.transform = transforms
        
        with open(self.attr_file) as f:
            content = f.readlines()

        content = [[symb for symb in x.strip().split(' ') if symb != ''] for x in content[1:]] 
        content_df = pd.DataFrame(content[1:])
        content_df.columns = ['path'] + content[0]
        
        self.attributes = content_df[['path'] + usecols]
        self.attributes[self.usecols] = self.attributes[self.usecols].astype(int)

        unique_labels = np.unique(self.attributes[self.usecols], axis=0)
        self.num_classes = unique_labels.shape[0]
        self.multi2one = {tuple(label): i for i, label in enumerate(unique_labels)}
        self.one2multi = {i: label.tolist() for i, label in enumerate(unique_labels)}
        
        encoded_labels = np.apply_along_axis(lambda x: self.multi2one[tuple(x)], 
                                             axis=1,
                                             arr=self.attributes[self.usecols])
        self.attributes['target'] = encoded_labels
        
    def __len__(self):
        return len(self.attributes)
    
    def __getitem__(self, idx):
        person = self.attributes.iloc[idx]
        
        attributes = person['target']
        img = Image.open(os.path.join(self.dataset_location, person['path']))
        
        if self.transform:
            img = self.transform(img)
        
        return (img, attributes)
    
    
class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, emb):
        self.embedding = emb
        one_hot_labels = np.array([a[-2:].numpy() for a in emb])
        unique_labels = np.unique(one_hot_labels, axis=0)
        self.num_classes = unique_labels.shape[0]
        self.multi2one = {tuple(label): i for i, label in enumerate(unique_labels)}
        self.one2multi = {i: label.tolist() for i, label in enumerate(unique_labels)}
        
        self.labels = np.apply_along_axis(lambda x: self.multi2one[tuple(x)], 
                                          axis=1,
                                          arr=one_hot_labels)
        
        
    def __len__(self):
        return len(self.embedding)
    
    def __getitem__(self, idx):
        obj = self.embedding[idx]
        label = self.labels[idx]
        while label == self.num_classes - 1:
            idx = np.random.choice(self.__len__())
            obj = self.embedding[idx]
            label = self.labels[idx]
        return (obj[:-2], label)

    
class BatchCollate(object):
    def __call__(self, batch):
        images = torch.stack([img for img, _ in batch])
        labels = torch.from_numpy(np.array([label for _, label in batch]))
        return images, labels