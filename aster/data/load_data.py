import torch
from torch.utils import data
from data.utils import strLabelConverter
from data.data_aug import data_augment
import os, pickle
from torchvision import transforms
from PIL import Image, ImageFile
import numpy as np
import string, lmdb, six
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pylab
import cv2
import Augmentor

ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_collate_fn(padding):
    def collate_fn(img_label):
        imgs,labels,index = zip(*img_label)
        imgs = torch.cat([img.unsqueeze(0) for img in imgs],0)
        lengths = [len(x) for x in labels]
        lengths_tensor = torch.LongTensor(lengths)
        batch_length = max(lengths)
        label_tensor = torch.LongTensor(batch_length,len(labels)).fill_(padding)
        for i,label in enumerate(labels):
            label_tensor[:lengths[i],i].copy_(label)
        return [imgs, (label_tensor, lengths_tensor),index]
    return collate_fn

class LoadData(data.Dataset):
    def __init__(self,img_list, input_size, input_transform=None, is_train=True, cache_file=None):
        self.converter = strLabelConverter()
        self.pad_token = self.converter.dict.get('<EOS>')
        self.is_train = is_train
        self.input_size = input_size[::-1]


        data_root = os.path.dirname(img_list)
        if cache_file is None:
            self._cache_file = ".cache_%s.pkl" % (os.path.basename(img_list).split(".")[0])
        else:
            self._cache_file = cache_file


        self.img_names, self.labels = [], []
        if os.path.exists(self._cache_file):
            with open(self._cache_file, "rb") as f:
                self.img_names, self.labels = pickle.load(f)
            print("Initial dataset from %s finished!" % self._cache_file)
        else:
            with open(img_list,'r') as f:
                for line in f.readlines():
                    name,tag = line.strip().split()
                    tag = self.converter.encode(tag)
                    self.img_names.append(os.path.join(data_root,name))
                    self.labels.append(tag)
            with open(self._cache_file, "wb") as f:
                pickle.dump((self.img_names, self.labels), f)
            print("Initial dataset from %s finished!" % img_list)

        print('Loaded {} {} images'.format(len(self.labels),'training' if is_train else 'test'))
        self.input_transform = input_transform

    def __getitem__(self,index):
        img = Image.open(self.img_names[index])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        target = np.array(self.labels[index])
        img = np.array(img.resize(self.input_size))
        if self.is_train:
            img = data_augment(img)
        if self.input_transform is not None:
            img = self.input_transform(img)
        return img, torch.LongTensor(target), index
    def __len__(self):
        return len(self.img_names)

class LmdbDataset(data.Dataset):
    def __init__(self, lmdb_path, input_size, transform, is_train=True):
        super(LmdbDataset, self).__init__()
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.is_train = is_train
        self.input_size = input_size[::-1] # PIL resize [w, h]
        self.padding_ratio = 0.005

        self.env = lmdb.open(lmdb_path, max_readers=32, readonly=True)
        assert self.env is not None, "cannot create lmdb obj from %s" %  lmdb_path
        self.txn = self.env.begin()
        self.count = int(self.txn.get(b'count'))
        print('Loaded {} {} images'.format(self.count,'training' if is_train else 'test'))
    
    @property
    def pad_token(self):
        return 1
    
    def __getitem__(self, idx):
        image_key = b'image-%08d' % idx
        image_buf = self.txn.get(image_key)
        try:
            io_buf = six.BytesIO()
            io_buf.write(image_buf)
            io_buf.seek(0)
            image = Image.open(io_buf)
            w, h = image.size
            padding_w, padding_h = int(self.padding_ratio*w), int(self.padding_ratio*h)
            image = transforms.Pad((padding_w, padding_h), padding_mode='edge')(image)

        except Exception as e:
            print("Error image: ", image_key)
            return self[(idx + 1) % len(self)]
        image = np.array(image.resize(self.input_size))
        if self.is_train:
            image = data_augment(image)
        if self.transform:
            image = Image.fromarray(image)
            # image = Augmentor.GenerateStretch(image, 4)
            image = self.transform(image)
            # image = transforms.ToPILImage(image)
            # print("111111111111111111111")
            # plt.imshow(image)
            # plt.show(image)
            # pylab.show(image)
            # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # cv2.imshow('image', image)
            # cv2.waitKey(0)



        label_key = b"label-%08d" % idx
        label_buf = self.txn.get(label_key)
        target = np.fromstring(label_buf, dtype=np.int32)
        
        return image, torch.LongTensor(target), idx
    
    def __len__(self):
        return self.count

def get_dataloader(img_list, input_size, trsf,batch_size,is_train=True, cache_file=None):
    kwargs = {'num_workers':6, 'pin_memory':True}
    if img_list.endswith('.txt'):
        dataset = LoadData(img_list, input_size, trsf, is_train)
    elif img_list.endswith('lmdb'):
        dataset = LmdbDataset(img_list, input_size, trsf, is_train)
    elif img_list.endswith('lmdb_train3'):
        dataset = LmdbDataset(img_list, input_size, trsf, is_train)
    elif img_list.endswith('lmdb_val3'):
        dataset = LmdbDataset(img_list, input_size, trsf, is_train)
    else:
        raise TypeError("Not support this dataset type, got {}".format(os.path.basename(img_list)))

    shuffle = True if is_train else False
    dataloader = data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,
                                 collate_fn=create_collate_fn(dataset.pad_token),**kwargs)
    return dataloader

if __name__ == '__main__':
    train_list = "/workspace/xwh/aster/train/train/"
    print('num_class: {}'.format(strLabelConverter().num_class))

    train_trsf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    train_loader = get_dataloader(train_list, [32, 100], train_trsf,20)
    for idx,(imgs,(labels,lengths),indexes) in enumerate(train_loader):
        # print(imgs.size())
        print(labels) # [seq,batch]
        # print(indexes,lengths)
        
