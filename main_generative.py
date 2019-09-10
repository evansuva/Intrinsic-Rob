import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generative'))

import utils, argparse, os, torch
import numpy as np
from gan import GAN
from cgan import CGAN
from acgan import ACGAN
from dsgan import DSGAN

# from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from pretrained.small_cnn import SmallCNN
from torchvision.models.inception import inception_v3

import torchvision.datasets as dset
import torchvision.transforms as transforms
from dataloader import CustomLabelDataset, IgnoreLabelDataset
from scipy.stats import entropy

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)
    
    # for training generative model
    parser.add_argument('--gan_type', type=str, default='ACGAN', choices=['GAN', 'CGAN', 'ACGAN', 'DSGAN'], 
                        help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar10'],
                        help='The name of dataset')
    parser.add_argument('--mode', type=str, default='evaluate', choices=['train','evaluate','reconstruct'])
    parser.add_argument('--epoch', type=int, default=25, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=28, help='The size of input image')
    parser.add_argument('--channels',  type=int, default=1, help='The number of rgb channels')
    
    parser.add_argument('--save_dir', type=str, default='generative/models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='generative/results', 
                        help='Directory name to save the generated images')
    # parser.add_argument('--log_dir', type=str, default='generative/logs', 
    #                     help='Directory name to save training logs')
    
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--benchmark_mode', type=bool, default=True)
    
    parser.add_argument('--alpha', type=float, default=1.0, help='penalize classification')
    parser.add_argument('--lambda_', type=float, default=0.05, help='penalize Lipschitz')
    # parser.add_argument('--n_repeat', type=int, default=10, help='number of sample noise')

    # for calculating local lipschitz constant
    parser.add_argument('--radius', type=float, default=1.0, help='latent space ball radius')
    parser.add_argument('--n_samples', type=int, default=500, help='number of natural samples')
    parser.add_argument('--n_neighbors', type=int, default=1000, help='number of neighboring points')

    # for reconstructing dataset
    parser.add_argument('--seed', type=int, default=0, help='manual seed number')
    parser.add_argument('--train_parts', type=int, default=6, help='number of partitions for training dataset')
    parser.add_argument('--train_size', type=int, default=10000, help='number of training samples')
    parser.add_argument('--test_size', type=int, default=10000, help='number of testing samples')
    
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # # --log_dir
    # if not os.path.exists(args.log_dir):
    #     os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    # --sample
    try:
        assert args.n_samples >= 1
    except:
        print('number of samples must be larger than or equal to one')

    # --neighbor
    try:
        assert args.n_neighbors >= 1
    except:
        print('number of neighbors must be larger than or equal to one')

    return args

""" Computing the inception score of the generative model"""
def inception_score(imgs, model, cuda, batch_size, img_size, n_class, resize=False, splits=1):

    N = len(imgs)
    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    def get_pred(x):
        if resize:
            x = F.interpolate(x, size=(img_size, img_size), mode='bilinear', align_corners=False).type(dtype)
        y = model(x)

        return F.softmax(y, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, n_class))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


"""main"""
def main():
    # parse arguments
    args = parse_args()

    if args is None:
        exit()

    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True

    # declare instance for GAN
    if args.gan_type == 'GAN':
        gan = GAN(args)
    elif args.gan_type == 'CGAN':
        gan = CGAN(args)
    elif args.gan_type == 'ACGAN':
        gan = ACGAN(args)
    elif args.gan_type == 'DSGAN':
        gan = DSGAN(args)
    else:
        raise Exception("[!] There is no option for " + args.gan_type)

    if args.mode == 'train':
        # launch the graph in a session
        gan.train()
        print(" [*] Training finished!")

        # visualize learned generator
        gan.visualize_results(args.epoch)
        print(" [*] Testing finished!")

    elif args.mode == 'evaluate':
        print(" [*] Compute the Lipschitz parameter")
        gan.get_lipschitz()
        print("")

        # print(" [*] Compute the inception score")
        # if args.dataset == 'mnist':
        #     model = SmallCNN()
        #     model.load_state_dict(torch.load('generative/pretrained/small_cnn/mnist.pt'))
        #     dataset = dset.MNIST(root='data/mnist/', train=False, 
        #                          download=True, transform=transforms.ToTensor())
        #     img_size = 28
        #     n_class = 10

        # elif args.dataset == 'fashion-mnist':
        #     model = SmallCNN()
        #     model.load_state_dict(torch.load('generative/pretrained/small_cnn/fashion-mnist.pt'))
        #     dataset = dset.FashionMNIST(root='data/fashion-mnist/', train=False, 
        #                                 download=True, transform=transforms.ToTensor())
        #     img_size = 28
        #     n_class = 10

        # elif args.dataset == 'cifar10':
        #     model = inception_v3(pretrained=True, transform_input=False)
        #     transform = transforms.Compose([transforms.ToTensor(), 
        #                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        #     dataset = dset.CIFAR10(root='data/cifar10/', download=True, transform=transform)
        #     img_size = 299    
        #     n_class = 1000         

        # else:
        #     raise Exception("[!] There is no option for " + args.dataset)

        # if args.gpu_mode:
        #     model = model.cuda()
        # model.eval()

        # print("Calculating Inception Score for originial dataset...")
        # IS_origin = inception_score(IgnoreLabelDataset(dataset), model, cuda=args.gpu_mode, 
        #                             batch_size=32, img_size=img_size, n_class=n_class, resize=True, splits=10)
        # print(IS_origin[0])

        # # test_sample_path = 'data/'+args.dataset+'/'+args.gan_type+'/'+'samples_test.npy'
        # # test_label_path ='data/'+args.dataset+'/'+args.gan_type+'/'+'labels_test.npy'

        # test_path = 'data/'+args.dataset+'/'+args.gan_type+'/'+'test.npz'
        # dataset_acgan = CustomLabelDataset(test_path, args.input_size, 
        #                         args.input_size, args.channels, transform=transforms.ToTensor())

        # print ("Calculating Inception Score for ACGAN...")
        # IS_gan = inception_score(IgnoreLabelDataset(dataset_acgan), model, cuda=args.gpu_mode, 
        #                          batch_size=32, img_size=img_size, n_class=n_class, resize=True, splits=10)
        # print(IS_gan[0])

        # # save the inception score
        # IS_log = open(args.log_dir+'/'+args.dataset+'/'+args.gan_type+'/ACGAN_IS.txt', 'w')
        # print("%.4f, %.4f" % (IS_origin[0], IS_gan[0]), file=IS_log)

    elif args.mode == 'reconstruct':
        print(" [*] Reconstruct "+args.dataset+" dataset using "+args.gan_type)
        gan.reconstruct()
    
    else: 
        raise Exception("[!] There is no option for " + args.mode)


if __name__ == '__main__':
    main()


