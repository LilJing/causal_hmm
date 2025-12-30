from __future__ import print_function
from datetime import datetime
from torch import optim
from model import *
import torch
from tensorboardX import SummaryWriter
import os
from train import train
from val import val
from test import test
from data_helper import RETINA
import logging
import warnings
warnings.filterwarnings('ignore')


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


import argparse
parser = argparse.ArgumentParser(description='Model hyperparameters')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
parser.add_argument('--epoch', type=int, default=100, help='training epoch')
parser.add_argument('--batch-size', type=int, default=16, help='batch size')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--fc-dim', type=int, default=128, help='FC layer dimension')
parser.add_argument('--cnn-layer', type=int, default=5, help='cnn layers')
parser.add_argument('--lstm-out', type=int, default=256, help='output dimension of lstm')
parser.add_argument('--image-size', type=int, default=128, help='image size')
parser.add_argument('--z-size', type=int, default=96, metavar='G', help='latent size of z for non-causal factors in vae')
parser.add_argument('--v-size', type=int, default=32, metavar='G', help='latent size of v for disease-causative factors in vae' )
parser.add_argument('--s-size', type=int, default=128, metavar='G', help='latent size of s for disease-causative factors in vae')
parser.add_argument('--optimizer', default='Adam', help='optimizer (Adam or RMSprop, SGD, Adagrad, Momentum, Adadelta)')
parser.add_argument('--kl-weight', type=float, default=0.1, metavar='LR', help='KL loss weight')
parser.add_argument('--init', default='xavier', help='pytorch default, kaiming_normal, xavier_normal')
parser.add_argument('--cls-loss-weight', type=float, default=1, metavar='LR', help='classification loss weight')
parser.add_argument('--to-grade', type=int, default=5, help='use data up to this grade')
parser.add_argument('--from-grade', type=int, default=0, help='use data from this grade onward')
parser.add_argument('--cls-fc-dim', type=int, default=128, help='FC layer dimension')


def main():
    args = parser.parse_args()

    batch_size = args.batch_size
    causal_hmm_model = Causal_HMM(args, z_size=args.z_size, v_size=args.v_size,s_size=args.s_size,
                      A_size=15, B_size=16, batch_size=args.batch_size, layer_count = args.cnn_layer, channels = 3)

    classifier = Disease_Classifier(args, in_dim=args.v_size + args.s_size)

    print('Causal HMM Model', causal_hmm_model)
    print('Classifier Model', classifier)

    causal_hmm_model._init_papameters(args)

    causal_hmm_model.cuda(), classifier.cuda()

    lr_vae, lr_pre = args.lr, args.lr

    vae_optimizer = optim.Adam(causal_hmm_model.parameters(), lr=lr_vae, betas=(0.5, 0.999), weight_decay=1e-5)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=lr_pre, betas=(0.5, 0.999), weight_decay=1e-5)

    if 'Adam' in args.optimizer:
        vae_optimizer = optim.Adam(causal_hmm_model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-5)
        classifier_optimizer = optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-5)
    elif 'RMSprop' in args.optimizer:
        vae_optimizer = torch.optim.RMSprop(causal_hmm_model.parameters(), lr=args.lr, alpha=0.9)
        classifier_optimizer = torch.optim.RMSprop(classifier.parameters(), lr=args.lr, alpha=0.9)
    elif 'SGD' in args.optimizer:
        vae_optimizer = torch.optim.SGD(causal_hmm_model.parameters(), lr=args.lr, momentum=0, weight_decay=1e-5)
        classifier_optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, momentum=0, weight_decay=1e-5)
    elif 'Adagrad' in args.optimizer:
        vae_optimizer = torch.optim.Adagrad(causal_hmm_model.parameters(), lr=args.lr, lr_decay=0, weight_decay=1e-5)
        classifier_optimizer = torch.optim.Adagrad(classifier.parameters(), lr=args.lr, lr_decay=0, weight_decay=1e-5)
    elif 'Momentum' in args.optimizer:
        vae_optimizer = torch.optim.SGD(causal_hmm_model.parameters(), lr=args.lr, momentum=0.9)
        classifier_optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9)
    elif 'Adadelta' in args.optimizer:
        vae_optimizer = torch.optim.Adadelta(causal_hmm_model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=1e-5)
        classifier_optimizer = torch.optim.Adadelta(classifier.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=1e-5)

    scheduler_vae = torch.optim.lr_scheduler.MultiStepLR(vae_optimizer, milestones=[10, 30, 50, 80], gamma=0.9)
    scheduler_pre = torch.optim.lr_scheduler.MultiStepLR(classifier_optimizer, milestones=[10, 30, 50, 80], gamma=0.9)

    current_time = datetime.now().strftime('%b%d_%H-%M')
    print('current_time', current_time)
    save_dir = os.path.join('./logs', current_time)
    writer = SummaryWriter(os.path.join(save_dir, 'Train'))

    # save params log
    log = {}
    setup_logger('params_log', r'{0}/logger'.format(save_dir))
    log['params_log'] = logging.getLogger('params_log')
    d_args = vars(args)
    for k in d_args.keys():
        log['params_log'].info('{0}: {1}'.format(k, d_args[k]))

    train_log = {}
    setup_logger('train_log', r'{0}/train_logger'.format(save_dir))
    train_log['train_log'] = logging.getLogger('train_log')

    val_log = {}
    setup_logger('val_log', r'{0}/val_logger'.format(save_dir))
    val_log['val_log'] = logging.getLogger('val_log')

    test_log = {}
    setup_logger('test_log', r'{0}/test_logger'.format(save_dir))
    test_log['test_log'] = logging.getLogger('test_log')

    train_epoch = args.epoch

    train_label_path = 'data path of the train set labels'
    val_label_path = 'data path of the validation set labels'
    test_label_path = 'data path of the test set labels'

    image_path = 'data path of image data set'
    train_dataset = RETINA(args, image_path, train_label_path)
    val_dataset = RETINA(args, image_path, val_label_path)
    test_dataset = RETINA(args, image_path, test_label_path)

    print('train num', train_dataset.__len__(), 'val num',
          val_dataset.__len__(), 'test num', test_dataset.__len__())

    n_iter = 0
    current_best_auc = 0
    for epoch in range(train_epoch):
        print('training in epoch {0}'.format(epoch))
        n_iter = train(args, causal_hmm_model, classifier, train_dataset, batch_size, vae_optimizer,
                      classifier_optimizer, writer, epoch, n_iter, train_log)

        scheduler_vae.step(), scheduler_pre.step()
        auc_val_value = val(args, causal_hmm_model, classifier, val_dataset, batch_size, writer, epoch, n_iter, val_log)

        if auc_val_value > current_best_auc:
            current_best_auc = auc_val_value
            torch.save(causal_hmm_model.state_dict(), os.path.join(save_dir, 'VAE_epoch_{0}.pkl'.format(epoch)))
            torch.save(classifier.state_dict(), os.path.join(save_dir, 'Classifier_epoch_{0}.pkl'.format(epoch)))
            val_log['val_log'].info("current best val auc:{0} at epoch {1}".format(current_best_auc, epoch))

        torch.cuda.empty_cache()
        test(args, causal_hmm_model, classifier, test_dataset, batch_size, writer, epoch, n_iter, test_log)

    print("Finish!... saved all results")


if __name__ == '__main__':
    main()
