from src.biggan import BigGAN128
from src.biggan import BigGAN256
from src.biggan import BigGAN512

import torch
import torchvision

import numpy as np
import scipy.misc
import imageio
import os


class ImageNetGenerator():
    def __init__(self):
        self.dataset = 'imagenet'
        self.model_name = 'biggan'

        self.train_parts = 500
        self.test_parts = 100
        self.train_size = 100
        self.test_size = 100
        self.seed = 141
        self.z_dim = 120
        self.class_labels = [404, 609, 11, 281, 351, 153, 30, 339, 724, 717]

        self.class_num = 10
        self.gpu_mode = True

        self.n_samples = 100
        self.n_neighbors = 100
        self.radius = 1

        self.checkpoint = './biggan128-release.pt'
        self.G = BigGAN128().cuda()
        self.G.load_state_dict(torch.load(self.checkpoint))
        self.G.eval()

    def generate_image(self, zs, ys):
        with torch.no_grad():
            imgs = self.G(zs, ys)
        imgs = 0.5 * (imgs.data + 1)
        imgs = torch.nn.functional.interpolate(imgs, size=(32, 32))
        return imgs

    def save_images(self, images, size, path):
        image = np.squeeze(self.merge(images, size))
        imageio.imwrite(path, image)

    def merge(self, images, size):
        h, w = images.shape[1], images.shape[2]
        if (images.shape[3] in (3, 4)):
            c = images.shape[3]
            img = np.zeros((h * size[0], w * size[1], c))
            for idx, image in enumerate(images):
                i = idx % size[1]
                j = idx // size[1]
                img[j * h:j * h + h, i * w:i * w + w, :] = image
            return img
        elif images.shape[3] == 1:
            img = np.zeros((h * size[0], w * size[1]))
            for idx, image in enumerate(images):
                i = idx % size[1]
                j = idx // size[1]
                img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
            return img
        else:
            raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

    def get_local_lipschitz(self, sample_z, sample_y, n_neighbors, gpu_mode, z_dim=100, radius=1):

        z_repeat = sample_z.repeat(n_neighbors, 1)
        y_repeat = sample_y.repeat(n_neighbors, 1).squeeze()

        # generate uniform noise from l2 ball
        v = torch.nn.functional.normalize(torch.rand(n_neighbors, z_dim), p=2, dim=1)
        u = torch.rand(n_neighbors) + 1e-12  # avoid underflow
        unif_noise = (radius * u ** (1 / float(z_dim))).unsqueeze(1) * v

        if gpu_mode:
            z_repeat, y_repeat, unif_noise = z_repeat.cuda(), y_repeat.cuda(), unif_noise.cuda()

        G_z = self.generate_image(z_repeat, y_repeat).view(n_neighbors, -1)
        G_z_neighbors = self.generate_image(z_repeat + unif_noise, y_repeat).view(n_neighbors, -1)

        # roughly estimate lipschitz using samples
        dist_z = torch.sqrt(torch.sum(unif_noise ** 2, dim=1))
        dist_x = torch.sqrt(torch.sum((G_z_neighbors - G_z) ** 2, dim=1))
        log_diff = torch.log(dist_x) - torch.log(dist_z)
        lipschitz = torch.exp(torch.max(log_diff)).cpu().item()

        return lipschitz

    def get_lipschitz(self):
        L = np.zeros((self.class_num, self.n_samples))

        for i in range(self.class_num):
            for j in range(self.n_samples):
                sample_z = torch.randn((1, self.z_dim))
                sample_y = torch.tensor([self.class_labels[i]])

                L[i, j] = self.get_local_lipschitz(sample_z, sample_y, self.n_neighbors, self.gpu_mode, self.z_dim,
                                                   self.radius)

            # print the results for class i
            print(
                """class: %d, (95%%) lipschitz: %.2f, (99%%) lipschitz: %.2f, (99.9%%) lipschitz: %.2f, (max) lipschitz: %.2f""" %
                (i, np.percentile(L[i, :], q=95), np.percentile(L[i, :], q=99), np.percentile(L[i, :], q=99.9),
                 np.max(L[i, :])))

    def generate_dataset(self):
        torch.manual_seed(self.seed)

        data_dir = 'data/' + self.dataset + '/' + self.model_name
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        for k in range(self.train_parts):
            sample_z = torch.randn((self.train_size, self.z_dim))
            labels = torch.randint(0, self.class_num, (self.train_size, 1)).type(torch.LongTensor)
            sample_y = torch.tensor([self.class_labels[label] for label in labels])
            # labels = labels_all / 5

            if self.gpu_mode:
                sample_z, sample_y = sample_z.cuda(), sample_y.cuda()

            samples = self.generate_image(sample_z, sample_y)

            if self.gpu_mode:
                samples = samples.cpu().data.numpy()
            else:
                samples = samples.data.numpy()

            if k == 0:
                labels_train = labels
                samples_train = samples
            else:
                labels_train = np.concatenate((labels_train, labels), axis=0)
                samples_train = np.concatenate((samples_train, samples), axis=0)

            print('train part ', k, ' done')

        np.savez(data_dir + '/train', sample=samples_train, label=labels_train.squeeze(1))

        # for testing
        torch.manual_seed(self.seed + 999)
        for k in range(self.test_parts):
            sample_z = torch.randn((self.train_size, self.z_dim))
            labels = torch.randint(0, self.class_num, (self.train_size, 1)).type(torch.LongTensor)
            sample_y = torch.tensor([self.class_labels[label] for label in labels])
            # labels = labels_all / 5

            if self.gpu_mode:
                sample_z, sample_y = sample_z.cuda(), sample_y.cuda()

            samples = self.generate_image(sample_z, sample_y)

            if k == 0:
                labels_t = labels
                samples_t = samples
                z_t = sample_z
            else:
                labels_t = torch.cat((labels_t, labels), axis=0)
                samples_t = torch.cat((samples_t, samples), axis=0)
                z_t = torch.cat((z_t, sample_z), axis=0)

            if self.gpu_mode:
                samples = samples.cpu().data.numpy()
            else:
                samples = samples.data.numpy()

            if k == 0:
                labels_test = labels
                samples_test = samples
            else:
                labels_test = np.concatenate((labels_test, labels), axis=0)
                samples_test = np.concatenate((samples_test, samples), axis=0)

            print('test part ', k, ' done')

        torch.save([samples_t, labels_t, z_t], data_dir + '/test/testset_with_z.pt')
        np.savez(data_dir + '/test', sample=samples_test, label=labels_test.squeeze(1))

        samples_test = samples_test.transpose(0, 2, 3, 1)
        self.save_images(samples_test[:100, :, :, :], [10, 10],
                         'data/' + self.dataset + '/' + self.model_name + '/gen_img.png')


if __name__ == '__main__':
    gen = ImageNetGenerator()
    gen.get_lipschitz()
    # gen.generate_dataset()
