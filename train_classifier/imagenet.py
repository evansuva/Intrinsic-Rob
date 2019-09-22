# import waitGPU
# waitGPU.wait(utilization=40, available_memory=8000, interval=60)

import problems as pblm
from trainer import *
import setproctitle
import random


def select_model(m): 
    if m == 'small': 
        model = pblm.cifar_model().cuda() 
    elif m == 'large': 
        model = pblm.cifar_model_large().cuda()
    # elif m == 'resNet':
    #     model = pblm.cifar_model_resnet().cuda()
    else:
        raise ValueError('model argument not recognized for imagenet')
    return model

if __name__ == "__main__": 
    args = pblm.argparser(prefix='imagenet', gan_type='biggan',
                starting_epsilon=0.01, opt='sgd', lr=0.05, 
                batch_size_test=8, proj=50, norm_train='l2_normal', 
                norm_test='l2', epsilon=0.1412, seed=0)

    setproctitle.setproctitle('python')
    kwargs = pblm.args2kwargs(args)
    print("saving file to {}".format(args.proctitle))

    saved_filepath = ('./saved_log/'+args.proctitle)
    model_filepath = os.path.dirname('./models/'+args.proctitle)
    if not os.path.exists(saved_filepath):
        os.makedirs(saved_filepath)
    if not os.path.exists(model_filepath):
        os.makedirs(model_filepath)
    model_path = ('./models/'+args.proctitle+'.pth')

    train_res = open(saved_filepath + '/train_res.txt', "w")
    test_res = open(saved_filepath + '/test_res.txt', "w")

    # load the data
    if args.prefix == "imagenet":
        train_loader, _ = pblm.cifar_loaders(args.batch_size, '../data/cifar10', )
        _, test_loader  = pblm.cifar_loaders(args.batch_size_test, '../data/cifar10')

    elif args.prefix == "custom_imagenet":
        train_loader, _ = pblm.custom_cifar_loaders(batch_size=args.batch_size, 
                                train_path= '../imagenet_gen/data/imagenet/'+args.gan_type+'/train.npz',
                                test_path = '../imagenet_gen/data/imagenet/'+args.gan_type+'/test.npz')
        _, test_loader  = pblm.custom_cifar_loaders(batch_size=args.batch_size_test, 
                                train_path= '../imagenet_gen/data/imagenet/'+args.gan_type+'/train.npz',
                                test_path = '../imagenet_gen/data/imagenet/'+args.gan_type+'/test.npz')

    else:
        raise ValueError("prefix argument not recognized for imagenet")

    # specify the model and the optimizer
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(0)
    np.random.seed(0)
    model = select_model(args.model) 

    if args.opt == 'adam': 
        opt = optim.Adam(model_path.parameters(), lr=args.lr)
    elif args.opt == 'sgd': 
        opt = optim.SGD(model.parameters(), lr=args.lr, 
                        momentum=args.momentum, weight_decay=args.weight_decay)
    else: 
        raise ValueError("Unknown optimizer")

    # learning rate decay and epsilon scheduling
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    eps_schedule = np.linspace(args.starting_epsilon, args.epsilon, args.schedule_length)

    for t in range(args.epochs):
        lr_scheduler.step(epoch=max(t-len(eps_schedule), 0))
        if t < len(eps_schedule) and args.starting_epsilon is not None: 
            epsilon = float(eps_schedule[t])
        else:
            epsilon = args.epsilon

        # standard training
        if args.method == 'baseline': 
            train_baseline(train_loader, model, opt, t, train_res, args.verbose)
            
            clas_err = evaluate_baseline(test_loader, model, t, test_res, args.verbose)

            if (t+1) % 1 == 0:
                torch.save(model.state_dict(), model_path+"_epoch_"+str(t+1)+".pth")

        # robust training
        elif args.method == 'zico_robust':
            train_robust(train_loader, model, opt, epsilon, t, train_res, args.verbose, 
                        norm_type=args.norm_train, bounded_input=False, **kwargs)
            
            clas_err, robust_err = evaluate_robust(test_loader, model, args.epsilon, t, test_res, 
                args.verbose, norm_type=args.norm_test, bounded_input=False, **kwargs)

        else:
            raise ValueError("Unknown type of training method.")

            # save the checkpoint for robust training
            if (t+1) % 5 == 0:
                torch.save(model.state_dict(), model_path+"_epoch_"+str(t+1)+".pth")

