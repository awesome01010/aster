import time,os,datetime
import argparse, sys
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from utils.utils import Logger, save_state, load_state
from data.load_data import get_dataloader
from data.multi_dataloader import get_multi_dataloader
from data.utils import strLabelConverter
from torch.autograd import Variable
from torchvision import utils as vutils, transforms
from model.seq2seq import Encoder, AttentionDecoder
from model_utils import batch_train, batch_test
from loss import SequenceCrossEntropyLoss
from tensorboardX import SummaryWriter

converter = strLabelConverter()

def test(encoder, decoder, test_loader, step=1, tfLogger=None):
    total, correct = 0, 0
    start = time.time()
    encoder.eval(), decoder.eval()
    for batch_idx,(imgs,(targets,lengths),_) in enumerate(test_loader):
        total += imgs.size(0)
        input_tensor = Variable(imgs.cuda())
        preds, _ = batch_test(input_tensor, encoder, decoder) # [b,t]
        targets = targets.numpy()               # [t,b]
        pred_seq = converter.decode(preds.cpu().numpy())
        target_seq = converter.decode(np.transpose(targets))
        for label, pred in zip(target_seq, pred_seq):
            print('===' * 10)
            print('pred: %s' % pred)
            print('label: %s' % label)
            if label == pred:
                correct += 1
    print('Finished testing in {}s\tAccuracy:{:.2f}%'.format(time.time() - start, 100. * (correct / total)))

    if tfLogger is not None:
        info = {
            'accuracy': correct / total,
        }
        for tag, value in info.items():
            tfLogger.add_scalar(tag, value, step)
    return correct / total

def main(args):
    input_size = [64, 256] if args.use_stn else [32, 64]

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # randomAffine = transforms.RandomAffine(10)
    # random_perspective = transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3)
    # color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0)
    # random_rotation = transforms.RandomRotation(5, resample=False, expand=False, center=None)
    # train_trsf = transforms.Compose([Pad, randomAffine, transforms.ToTensor(), normalize])
    # train_trsf = transforms.Compose([randomAffine, random_perspective, color_jitter, transforms.ToTensor(), normalize])
    train_trsf = transforms.Compose([transforms.ToTensor(), normalize])

    train_loader = get_dataloader(args.training_data, input_size, train_trsf, args.batch_size)
    # train_loader = get_multi_dataloader(args.training_data, train_trsf, args.batch_size)
    test_trsf = transforms.Compose([transforms.ToTensor(), normalize])
    test_loader = get_dataloader(args.test_data, input_size, test_trsf, args.batch_size, is_train=False)

    encoder = Encoder(use_stn=args.use_stn).cuda()
    decoder = AttentionDecoder(hidden_dim=256, attention_dim=256, y_dim=converter.num_class, 
                encoder_output_dim=512).cuda() # output_size for classes_num

    encoder_optimizer = optim.Adadelta(encoder.parameters(),lr=args.lr)
    decoder_optimizer = optim.Adadelta(decoder.parameters(),lr=args.lr)
    optimizers = [encoder_optimizer, decoder_optimizer]
    # lr_step = [200000, 300000, 400000]
    lr_step = [3000000, 4000000, 5000000, 6000000, 7000000, 8000000]
    encoder_scheduler = optim.lr_scheduler.MultiStepLR(encoder_optimizer, lr_step, gamma=0.1)
    decoder_scheduler = optim.lr_scheduler.MultiStepLR(decoder_optimizer, lr_step, gamma=0.1)
    criterion = SequenceCrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss()

    step, total_loss, best_res = 1, 0, 0
    if args.restore_step > 0:
        step = args.restore_step
        load_state(args.logs_dir, step, encoder, decoder, optimizers)

    sys.stdout = Logger(os.path.join(args.logs_dir, 'log.txt'))
    train_tfLogger = SummaryWriter(os.path.join(args.logs_dir, 'train'))
    test_tfLogger = SummaryWriter(os.path.join(args.logs_dir, 'test'))

    # start training
    while True:
        for batch_idx,(imgs,(targets,targets_len),idx) in enumerate(train_loader):
            # print(targets.size())
            # print(targets_len)
            # print(targets)
            input_data, targets, targets_len = imgs.cuda(), targets.cuda(), targets_len.cuda()
            encoder_optimizer.zero_grad(), decoder_optimizer.zero_grad()

            loss, recitified_img = batch_train(input_data, targets, targets_len, encoder, decoder, criterion, 1.0)
            encoder_optimizer.step(), decoder_optimizer.step()
            encoder_scheduler.step(), decoder_scheduler.step()
            total_loss += loss

            if step % 100 == 0:
                print('==' * 30)
                preds, _ = batch_test(input_data, encoder, decoder)
                print('preds: ',converter.decode(preds.cpu().numpy()))
                print('lable: ', converter.decode(targets.permute(1,0).cpu().numpy()))
                encoder.train(), decoder.train()

            if step % args.log_interval == 0:
                print('{} step:{}\tLoss: {:.6f}'.format(
                    datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
                    step, total_loss/args.log_interval))
                if train_tfLogger is not None:
                    x = vutils.make_grid(input_data.cpu())
                    train_tfLogger.add_image('train/input_img', x, step)
                    if args.use_stn:
                        x = vutils.make_grid(recitified_img.cpu())
                        train_tfLogger.add_image('train/recitified_img', x, step)
                    for param_group in encoder_optimizer.param_groups:
                        lr = param_group['lr']
                    info = {'loss':total_loss/args.log_interval,
                            'learning_rate':lr}
                    for tag,value in info.items():
                        train_tfLogger.add_scalar(tag,value,step)
                total_loss = 0
            if step % args.save_interval == 0:
                # save params
                save_state(args.logs_dir, step, encoder, decoder, optimizers)

                # Test after an args.save_interval
                res = test(encoder, decoder, test_loader, step=step, tfLogger=test_tfLogger)
                is_best = res >= best_res
                best_res = max(res, best_res)
                if best_res >= res:
                    best_step = step
                print('\nFinished step {:3d}  TestAcc: {:.6f}  best: {:.4%}  best step: {:3d}{}\n'.
                    format(step, res, best_res, best_step, ' *' if is_best else ''))
                encoder.train(), decoder.train()

            step += 1

    # Close the tf logger
    train_tfLogger.close()
    test_tfLogger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ASTER')
    parser.add_argument('--training_data', type=str,
                        default="/workspace/xwh/aster/train/ctw/workspace/xqq/ctw-baseline/data/all_images/lmdb_train3")
    parser.add_argument('--test_data', type=str, default="/workspace/xwh/aster/train/ctw/workspace/xqq/ctw-baseline/data/all_images/lmdb_val3")
    # parser.add_argument('--training_data', type=str,
    #                     default="/workspace/xwh/aster/train/val/lmdb")
    # parser.add_argument('--test_data', type=str, default="/workspace/xwh/aster/train/val/lmdb")

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_interval',type=int,default=1000,metavar='N',
                        help='how many steps to wait before saving model parmas')
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default='./params_test/DA/')
    parser.add_argument('--restore_step', type=int, default=0, help='restore for restore_step')
    parser.add_argument('--use_stn',action='store_true',default=False)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')

    #os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    main(parser.parse_args())
