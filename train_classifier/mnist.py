import waitGPU
waitGPU.wait(utilization=20, available_memory=10000, interval=20)

import problems as pblm
from trainer import *
import setproctitle

if __name__ == "__main__": 
    args = pblm.argparser(prefix='mnist', gan_type='ACGAN', opt='adam', 
                    batch_size_test=10, proj=50, norm_train='l2_normal', 
                    norm_test='l2', epsilon=1.58, seed=0)
    
    kwargs = pblm.args2kwargs(args)
    setproctitle.setproctitle('python')
    print("saving file to {}".format(args.proctitle))

    saved_filepath = ('./saved_log/'+args.proctitle)
    model_filepath = os.path.dirname('./models/'+args.proctitle)
    if not os.path.exists(saved_filepath):
        os.makedirs(saved_filepath)
    if not os.path.exists(model_filepath):
        os.makedirs(model_filepath)
    model_path = ('./models/'+args.proctitle)

    train_res = open(saved_filepath + '/train_res.txt', "w")
    test_res = open(saved_filepath + '/test_res.txt', "w")

    # load the data
    if args.prefix == "mnist":
        train_loader, _ = pblm.mnist_loaders(args.batch_size, '../data/mnist', )
        _, test_loader  = pblm.mnist_loaders(args.batch_size_test, '../data/mnist')

    elif args.prefix == "custom_mnist":
        train_loader, _ = pblm.custom_mnist_loaders(batch_size=args.batch_size, 
                                train_path= '../data/mnist/'+args.gan_type+'/train.npz', 
                                test_path = '../data/mnist/'+args.gan_type+'/test.npz')
        _, test_loader  = pblm.custom_mnist_loaders(batch_size=args.batch_size_test, 
                                train_path= '../data/mnist/'+args.gan_type+'/train.npz', 
                                test_path = '../data/mnist/'+args.gan_type+'/test.npz')

    else:
        raise ValueError("prefix argument not recognized for MNIST")                         
    
    # specify the model and the optimizer
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    model = pblm.mnist_model().cuda() 
 
    if args.opt == 'adam': 
        opt = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd': 
        opt = optim.SGD(model.parameters(), lr=args.lr, 
                        momentum=args.momentum, weight_decay=args.weight_decay)
    else: 
        raise ValueError("Unknown optimizer.")

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

        # robust training
        elif args.method == 'zico_robust':
            train_robust(train_loader, model, opt, epsilon, t, train_res, 
                        args.verbose, norm_type=args.norm_train, bounded_input=True, **kwargs)
            
            clas_err, robust_err = evaluate_robust(test_loader, model, args.epsilon, t, test_res, 
                        args.verbose, norm_type=args.norm_test, bounded_input=True, **kwargs)

            torch.save(model.state_dict(), model_path+"_epoch_"+str(t)+".pth")

        else:
            raise ValueError("Unknown type of training method.")

