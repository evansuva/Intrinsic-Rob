import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from dataloader import dataloader

from torch.autograd import grad

class generator(nn.Module):

    def __init__(self, input_dim=100, output_dim=1, input_size=32, class_num=10):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.class_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):

    def __init__(self, input_dim=1, output_dim=1, input_size=32, class_num=10):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.dc = nn.Sequential(
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        self.cl = nn.Sequential(
            nn.Linear(1024, self.class_num),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc1(x)
        d = self.dc(x)
        c = self.cl(x)

        return d, c

class DSGAN(object):
    def __init__(self, args):
        # parameters for train
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir 
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        # self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type + '_' + str(args.lambda_)
        self.input_size = args.input_size
        self.z_dim = 100
        self.class_num = 10
        self.sample_num = self.class_num ** 2

        self.alpha = args.alpha
        self.lambda_ = args.lambda_
        # self.n_repeat = args.n_repeat

        # parameters for evaluate
        self.n_samples = args.n_samples
        self.n_neighbors = args.n_neighbors
        self.radius = args.radius

        # parameters for reconstruct
        self.manual_seed = args.seed
        self.train_size = args.train_size
        self.test_size = args.test_size
        self.train_parts = args.train_parts


        # load dataset
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        data = self.data_loader.__iter__().__next__()[0]

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
            self.CE_loss = nn.CrossEntropyLoss()

        # print('---------- Networks architecture -------------')
        # utils.print_network(self.G)
        # utils.print_network(self.D)
        # print('-----------------------------------------------')

        # fixed noise & condition
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(self.class_num):
            self.sample_z_[i*self.class_num] = torch.rand(1, self.z_dim)
            for j in range(1, self.class_num):
                self.sample_z_[i*self.class_num + j] = self.sample_z_[i*self.class_num]

        temp = torch.zeros((self.class_num, 1))
        for i in range(self.class_num):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(self.class_num):
            temp_y[i*self.class_num: (i+1)*self.class_num] = temp

        self.sample_y_ = torch.zeros((self.sample_num, self.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)
        if self.gpu_mode:
            self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, y_) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break
                z_ = torch.rand((self.batch_size, self.z_dim))
                y_vec_ = torch.zeros((self.batch_size, self.class_num)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)
                if self.gpu_mode:
                    x_, z_, y_vec_ = x_.cuda(), z_.cuda(), y_vec_.cuda()

                # update D network
                self.D_optimizer.zero_grad()

                D_real, C_real = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)
                C_real_loss = self.CE_loss(C_real, torch.max(y_vec_, 1)[1])

                G_ = self.G(z_, y_vec_)
                D_fake, C_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)
                C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])

                D_loss = D_real_loss + self.alpha*C_real_loss + D_fake_loss + self.alpha*C_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                z_.requires_grad = True
                G_ = self.G(z_, y_vec_)
                D_fake, C_fake = self.D(G_)

                G_loss = self.BCE_loss(D_fake, self.y_real_)
                C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])

                G_loss += self.alpha*C_fake_loss

                # penalize global Lipschitz using gradient norm
                gradients = grad(outputs=G_, inputs=z_, grad_outputs=torch.ones(G_.size()).cuda(),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

                # gradients = torch.zeros(G_.size()).cuda()
                reg_loss = (gradients.view(gradients.size()[0], -1).norm(2, 1) ** 2).mean()
                G_loss += self.lambda_ * reg_loss

                # # penalize local Lipschitz
                # reg_loss = 0
                # for j in range(self.n_repeat):
                #     v = f.normalize(torch.rand(self.batch_size, self.z_dim), p=2, dim=1)
                #     u = torch.rand(self.batch_size) + 1e-12    # avoid underflow
                #     unif_noise = (u ** (1/float(self.z_dim))).unsqueeze(1)*v
                #     unif_noise = unif_noise.cuda()

                #     G_neighbor_ = self.G(z_+unif_noise, y_vec_)
                #     dist_x = torch.sqrt(torch.sum((G_neighbor_.view(self.batch_size, -1) - G_.view(self.batch_size, -1))**2, dim=1))
                #     dist_z = torch.sqrt(torch.sum(unif_noise**2, dim=1))

                #     reg_loss += torch.mean(dist_x / dist_z)
                
                # G_loss += self.lambda_ * reg_loss / self.n_repeat

                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, reg_loss: %.4f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item(), reg_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name, self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_, self.sample_y_)
        else:
            """ random noise """
            sample_y_ = torch.zeros(self.batch_size, self.class_num).scatter_(1, torch.randint(0, self.class_num - 1, (self.batch_size, 1)).type(torch.LongTensor), 1)
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_, sample_y_ = sample_z_.cuda(), sample_y_.cuda()

            samples = self.G(sample_z_, sample_y_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

    
    def get_lipschitz(self):
        self.G.eval()
        self.load()
        L = np.zeros((self.class_num, self.n_samples))

        log = open(self.save_dir+'/'+self.dataset+'/'+self.model_name+'/'+self.model_name+'_lipschitz.txt', "w")

        for i in range(self.class_num):
            for j in range(self.n_samples):
                sample_z = torch.rand((1, self.z_dim))
                sample_y = torch.zeros(1, self.class_num).scatter_(1, torch.LongTensor([[i]]), 1)

                L[i,j] = get_local_lipschitz(self.G, sample_z, sample_y, self.n_neighbors, self.gpu_mode, self.z_dim, self.radius)

            # print the results for class i
            print("""class: %d, (95%%) lipschitz: %.2f, (99%%) lipschitz: %.2f, (99.9%%) lipschitz: %.2f, (max) lipschitz: %.2f""" % 
                                    (i, np.percentile(L[i,:], q=95), np.percentile(L[i,:], q=99), np.percentile(L[i,:], q=99.9), np.max(L[i,:])))
                                    
            print("%d, %.2f, %.2f, %.2f, %.2f" % (i, np.percentile(L[i,:], q=95), np.percentile(L[i,:], q=99), np.percentile(L[i,:], q=99.9), np.max(L[i,:])), file=log)
            log.flush()

    def reconstruct(self):
        self.G.eval()
        self.load()

        # for training (partition the training data in case memory overflow)
        torch.manual_seed(self.manual_seed)

        for k in range(self.train_parts):
            sample_z = torch.rand((self.train_size, self.z_dim))
            labels = torch.randint(0, self.class_num, (self.train_size, 1)).type(torch.LongTensor)
            sample_y = torch.zeros(self.train_size, self.class_num).scatter_(1, labels, 1)

            # # estimate Lipschitz constants of the conditional generator for each sampled z 
            # lipschitz = np.zeros(self.train_size)
            # for i in range(self.train_size):
            #     lipschitz[i] = get_local_lipschitz(self.G, sample_z[i], sample_y[i], 
            #                             self.n_neighbors, self.gpu_mode, z_dim=100, radius=1)
            #     if i % 200 == 0:            
            #         print("Iteration: [%d] [%5d/%5d] lipschitz: %.2f" % (k, i, self.train_size, lipschitz[i]))

            if self.gpu_mode:
                sample_z, sample_y = sample_z.cuda(), sample_y.cuda()

            samples = (self.G(sample_z, sample_y) + 1) / 2

            if self.gpu_mode:
                samples = samples.cpu().data.numpy()
            else:
                samples = samples.data.numpy()

            if k==0:
                labels_train = labels
                samples_train = samples 
                # lipschitz_train = lipschitz
            else:
                labels_train = np.concatenate((labels_train, labels), axis=0)
                samples_train = np.concatenate((samples_train, samples), axis=0)
                # lipschitz_train = np.concatenate((lipschitz_train, lipschitz), axis=None)

        data_dir = 'data/'+self.dataset+'/'+self.model_name
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # np.save('data/'+self.dataset+'/'+self.model_name+'/samples_train', samples_train)
        # np.save('data/'+self.dataset+'/'+self.model_name+'/labels_train', labels_train.squeeze(1))
        # np.savez(data_dir+'/train', sample=samples_train, label=labels_train.squeeze(1), lipschitz=lipschitz_train)
        np.savez(data_dir+'/train', sample=samples_train, label=labels_train.squeeze(1))

        # for testing
        torch.manual_seed(self.manual_seed+999)
        sample_z_test = torch.rand((self.test_size, self.z_dim))
        labels_test = torch.randint(0, self.class_num, (self.test_size, 1)).type(torch.LongTensor)
        sample_y_test = torch.zeros(self.test_size, self.class_num).scatter_(1, labels_test, 1)

        # # estimate Lipschitz constants of the conditional generator for each sampled z 
        # lipschitz_test = np.zeros(self.test_size)
        # for i in range(self.test_size):
        #     lipschitz_test[i] = get_local_lipschitz(self.G, sample_z_test[i], sample_y_test[i], 
        #                             self.n_neighbors, self.gpu_mode, z_dim=100, radius=1)
        #     if i % 200 == 0:            
        #         print("Iteration: [%d] [%5d/%5d] lipschitz: %.2f" % (k, i, self.train_size, lipschitz_test[i]))

        if self.gpu_mode:
            sample_z_test, sample_y_test = sample_z_test.cuda(), sample_y_test.cuda()

        samples_test = (self.G(sample_z_test, sample_y_test) + 1) / 2

        if self.gpu_mode:
            samples_test = samples_test.cpu().data.numpy()
        else:
            samples_test = samples_test.data.numpy()

        # np.save('data/'+self.dataset+'/'+self.model_name+'/samples_test', samples_test)
        # np.save('data/'+self.dataset+'/'+self.model_name+'/labels_test', labels_test.squeeze(1))
        # np.savez(data_dir+'/test', sample=samples_test, label=labels_test.squeeze(1), lipschitz=lipschitz_test)
        np.savez(data_dir+'/test', sample=samples_test, label=labels_test.squeeze(1))

        samples_test = samples_test.transpose(0, 2, 3, 1)
        utils.save_images(samples_test[:100, :, :, :], [10, 10], 
                          self.save_dir+'/'+self.dataset+'/'+self.model_name+'/gen_img.png')


# compute the local lipschitz constant of a generator using samples
def get_local_lipschitz(G, sample_z, sample_y, n_neighbors, gpu_mode, z_dim=100, radius=1):

    z_repeat = sample_z.repeat(n_neighbors,1)
    y_repeat = sample_y.repeat(n_neighbors,1)

    # generate uniform noise from l2 ball
    v = f.normalize(torch.rand(n_neighbors, z_dim), p=2, dim=1)
    u = torch.rand(n_neighbors) + 1e-12    # avoid underflow
    unif_noise = (radius * u ** (1/float(z_dim))).unsqueeze(1)*v

    if gpu_mode:
        z_repeat, y_repeat, unif_noise = z_repeat.cuda(), y_repeat.cuda(), unif_noise.cuda()

    G_z = G(z_repeat, y_repeat).view(n_neighbors, -1)
    G_z_neighbors = G(z_repeat+unif_noise, y_repeat).view(n_neighbors, -1)

    # roughly estimate lipschitz using samples 
    dist_z = torch.sqrt(torch.sum(unif_noise**2, dim=1))
    dist_x = torch.sqrt(torch.sum((G_z_neighbors-G_z)**2, dim=1))
    log_diff = torch.log(dist_x) - torch.log(dist_z)
    lipschitz = torch.exp(torch.max(log_diff)).cpu().item()

    return lipschitz 