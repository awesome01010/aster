import torch, os, argparse
from data.utils import strLabelConverter
from model.seq2seq import Encoder, AttentionDecoder
from torchvision import transforms
from data.load_data import get_dataloader
from model_utils import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, default="./params_4datasets/DA")
parser.add_argument("--adjust_dir", type=str, default="/workspace/xwh/aster/train/test_MTWI_18test/train_images/")
parser.add_argument("--restore_iter", type=int, default=1312000)
parser.add_argument("--test_data", type=str, default="/workspace/xwh/aster/train/test_MTWI_18test/train_images/lmdb")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument('--use_stn',action='store_true',default=False)
args = parser.parse_args()
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'


# accuracy_file = os.path.join(args.ckpt_dir, "accuracy.txt")
accuracy_file = os.path.join("/workspace/xwh/aster/train/test_MTWI_18test/train_images/", "accuracy.txt")

label_map = strLabelConverter()

input_size = [64, 256] if args.use_stn else [32, 128]
# Pad = transforms.Pad((10, 0, 0, 0), fill=0, padding_mode="edge")
# test_trsf = transforms.Compose([Pad, transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,
#     0.224, 0.225])])
test_trsf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,
    0.224, 0.225])])

test_loader = get_dataloader(args.test_data, input_size, test_trsf, args.batch_size, is_train=False)

encoder = Encoder(use_stn=args.use_stn).cuda()
decoder = AttentionDecoder(hidden_dim=256, attention_dim=256, y_dim=label_map.num_class, 
            encoder_output_dim=512).cuda() # y_dim for classes_num

evaluate(encoder, decoder, args.ckpt_dir, args.adjust_dir, args.restore_iter, test_loader, label_map, accuracy_file)
