import torch
from torch.utils import data
from data.utils import strLabelConverter
import os, pickle
from torchvision import transforms
from PIL import Image, ImageFile
import numpy as np
import string, lmdb, six
from sklearn.utils import shuffle

ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_collate_fn(padding):
    def collate_fn(img_labels):
        img_label1, img_label2 = zip(*img_labels)
        imgs1, labels1, index1 = zip(*img_label1)
        imgs2, labels2, index2 = zip(*img_label2)
        imgs = torch.cat([torch.stack(imgs1),torch.stack(imgs2)])
        labels = list(labels1) + list(labels2)
        index = np.concatenate([index1, index2])

        lengths = [len(x) for x in labels]
        lengths_tensor = torch.LongTensor(lengths)
        batch_length = max(lengths)
        label_tensor = torch.LongTensor(batch_length,len(labels)).fill_(padding)
        for i,label in enumerate(labels):
            label_tensor[:lengths[i],i].copy_(torch.LongTensor(label))
        return [imgs, (label_tensor, lengths_tensor),index]
    return collate_fn

class LoadData(data.Dataset):
    def __init__(self,img_list, input_transform=None, is_train=True, cache_file=None):
        self.converter = strLabelConverter()
        self.pad_token = self.converter.dict.get('<EOS>')

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
        target = np.array(self.labels[index])
        if self.input_transform is not None:
            img = self.input_transform(img)
        return img, torch.LongTensor(target), index
    def __len__(self):
        return len(self.img_names)

class LmdbDataset(data.Dataset):
    def __init__(self, lmdb_path, transform, is_train=True):
        super(LmdbDataset, self).__init__()
        self.lmdb_path = lmdb_path
        self.transform = transform

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
        except Exception as e:
            print("Error image: ", image_key)
            return self[(idx + 1) % len(self)]
        if self.transform:
            image = self.transform(image)
        
        label_key = b"label-%08d" % idx
        label_buf = self.txn.get(label_key)
        target = np.fromstring(label_buf, dtype=np.int32)
        
        return image, target, idx
    
    def __len__(self):
        return self.count

class ConcatDataset(data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, index):
        return [d[index] for d in self.datasets]

    def __len__(self):
        return min(len(d) for d in self.datasets)


def get_multi_dataloader(img_list,trsf,batch_size,is_train=True, cache_file=None):
    kwargs = {'num_workers':6, 'pin_memory':True}
    if img_list.endswith('.txt'):
        dataset = LoadData(img_list,trsf,is_train)
    elif img_list.endswith('lmdb'):
        dataset1 = LmdbDataset('/workspace/xqq/datasets/synth90k/lmdb', trsf, is_train)
        dataset2 = LmdbDataset('/workspace/xqq/datasets/croped_SynthText/lmdb', trsf, is_train)
    else:
        raise TypeError("Not support this dataset type, got {}".format(os.path.basename(img_list)))

    shuffle = True if is_train else False
    dataloader = data.DataLoader(ConcatDataset(dataset1, dataset2), batch_size=batch_size,shuffle=shuffle,
                                 collate_fn=create_collate_fn(dataset1.pad_token),**kwargs)
    return dataloader

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    train_list = "/workspace/xqq/aster/params/demo/lmdb"

    train_trsf = transforms.Compose([transforms.Resize([32,100]),
                                    transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    train_loader = get_mutli_dataloader(train_list,train_trsf,2)
    for idx,(imgs,(labels,lengths),indexes) in enumerate(train_loader):
        print(imgs.size())
        print(labels) # [seq,batch]
        print(indexes,lengths)
        
