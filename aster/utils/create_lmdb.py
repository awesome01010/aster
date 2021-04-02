import lmdb
import os
import sys
from data.utils import strLabelConverter
import numpy as np
from tqdm import tqdm

tags_file = '/workspace/xqq/datasets/synth90k/synth90k.txt'
lmdb_save_path = '/workspace/xqq/datasets/synth90k/lmdb'

max_size = pow(2, 40)
env = lmdb.open(lmdb_save_path, map_size=max_size)
txn = env.begin(write=True)

count = 0
label_map = strLabelConverter()
with open(tags_file, 'r') as f:
    for index, line in enumerate(tqdm(f.readlines())):
        image_path, gt = line.strip().split() # image_path: abs path
        image_buf = open(image_path, 'rb').read()
        image_key = b'image-%08d' % index
        txn.put(key=image_key, value=image_buf)

        label = label_map.encode(gt)
        label_buf = np.array(label, np.int32).tostring()
        label_key = b'label-%08d' % index
        txn.put(key=label_key, value=label_buf)
        count += 1
txn.put(key=b'count', value=str(count).encode())
txn.commit()
env.close()
print('Finished %d' % count)
