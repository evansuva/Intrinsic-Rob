import waitGPU
waitGPU.wait(utilization=20, available_memory=10000, interval=20)

import problems as pblm
from trainer import *
import setproctitle

if __name__ == "__main__": 
    # args = pblm.argparser(prefix='mnist', opt='adam', norm_train='l1', 
    #                     norm_test='l1', epsilon=0.1, thres=0.02)
    args = pblm.argparser(batch_size_test=10, opt='adam', proj=50, 
                        norm_train='l2_normal', norm_test='l2', epsilon=1.58, thres=0.15)
    kwargs = pblm.args2kwargs(args)
    setproctitle.setproctitle('python')
    print("saving file to {}".format(args.proctitle))

    if args.method == 'overall_robust' or args.method == 'lipschitz_robust':
        print("threshold for classification error: {:.1%}".format(args.thres))
    elif args.method != 'baseline':
        raise ValueError("Unknown training method.")

    saved_filepath = ('./saved_log/'+args.proctitle)
    model_filepath = os.path.dirname('./models/'+args.proctitle)
    if not os.path.exists(saved_filepath):
        os.makedirs(saved_filepath)
    if not os.path.exists(model_filepath):
        os.makedirs(model_filepath)
    model_path = ('./models/'+args.proctitle+'.pth')

    train_log = open(saved_filepath + '/train_log.txt', "w")
    train_res = open(saved_filepath + '/train_res.txt', "w")
    valid_res = open(saved_filepath + '/valid_res.txt', "w")
    best_res = open(saved_filepath + "/best_res.txt", "w")

    # train-validation split
    if args.prefix == "mnist":
        train_loader, _, _ = pblm.mnist_loaders(batch_size=args.batch_size, 
                                                path='../data/mnist', 
                                                ratio=args.ratio, 
                                                seed=args.seed)

        _, valid_loader, test_loader = pblm.mnist_loaders(batch_size=args.batch_size_test, 
                                                path='../data/mnist', 
                                                ratio=args.ratio, 
                                                seed=args.seed)

    elif args.prefix == "custom_mnist":
        train_loader, _, _ = pblm.custom_mnist_loaders(batch_size=args.batch_size, 
                                            train_path='../data/mnist/ACGAN/train.npz', 
                                            test_path = '../data/mnist/ACGAN/test.npz',
                                            ratio=args.ratio, seed=args.seed,
                                            is_lipschitz=False)

        _, valid_loader, test_loader = pblm.custom_mnist_loaders(batch_size=args.batch_size_test, 
                                            train_path='../data/mnist/ACGAN/train.npz', 
                                            test_path = '../data/mnist/ACGAN/test.npz',
                                            ratio=args.ratio, seed=args.seed,
                                            is_lipschitz=False)

    elif args.prefix == "custom_lipschitz_mnist":
        train_loader, _, _ = pblm.custom_mnist_loaders(batch_size=args.batch_size, 
                                            train_path='../data/mnist/ACGAN/train.npz', 
                                            test_path = '../data/mnist/ACGAN/test.npz',
                                            ratio=args.ratio, seed=args.seed,
                                            is_lipschitz=True)

        _, valid_loader, test_loader = pblm.custom_mnist_loaders(batch_size=args.batch_size_test, 
                                            train_path='../data/mnist/ACGAN/train.npz', 
                                            test_path = '../data/mnist/ACGAN/test.npz',
                                            ratio=args.ratio, seed=args.seed,
                                            is_lipschitz=False)

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

    clas_err_min = 1
    robust_err_min = 1
    flag = False    # indicate whether we can find a proper clasifier

    for t in range(args.epochs):
        lr_scheduler.step(epoch=max(t-len(eps_schedule), 0))
        if t < len(eps_schedule) and args.starting_epsilon is not None: 
            epsilon = float(eps_schedule[t])
        else:
            epsilon = args.epsilon

        # standard training
        if args.method == 'baseline': 
            train_baseline(train_loader, model, opt, t, train_log, train_res, args.verbose)
            clas_err = evaluate_baseline(valid_loader, model, t, valid_res, args.verbose)

            if clas_err < clas_err_min:
                flag = True
                t_best = t
                clas_err_min = clas_err
                torch.save(model.state_dict(), model_path)

        # robust training
        elif args.method == 'overall_robust':
            train_robust(train_loader, model, opt, epsilon, t, train_log, train_res, 
                        args.verbose, norm_type=args.norm_train, bounded_input=True, **kwargs)
            clas_err, robust_err = evaluate_robust(valid_loader, model, args.epsilon, t, valid_res, 
                        args.verbose, norm_type=args.norm_test, bounded_input=True, **kwargs)

            if clas_err <= args.thres and robust_err < robust_err_min and t >= args.schedule_length:
                flag = True
                t_best = t
                clas_err_best = clas_err
                robust_err_min = robust_err    
                torch.save(model.state_dict(), model_path)

        elif args.method == 'lipschitz_robust':
            train_lipschitz_robust(train_loader, model, opt, epsilon, t, train_log, train_res, 
                        args.verbose, norm_type=args.norm_train, bounded_input=True, **kwargs)
            # use the same evaluation criteria
            clas_err, robust_err = evaluate_robust(valid_loader, model, args.epsilon, t, valid_res, 
                        args.verbose, norm_type=args.norm_test, bounded_input=True, **kwargs)

            if clas_err <= args.thres and robust_err < robust_err_min and t >= args.schedule_length:
                flag = True
                t_best = t
                clas_err_best = clas_err
                robust_err_min = robust_err    
                torch.save(model.state_dict(), model_path)
        else:
            raise ValueError("Unknown type of training method.")

    print('======================================== tuning results ========================================')
    if flag == False:
        print('None of the epochs evaluated satisfy the criteria')
    else:
        if args.method == 'baseline':
            print('at epoch', t_best, 'achieves')
            print('lowest classification error:', '{:.2%}'.format(clas_err_min))
            print('baseline model:', t_best, '{:.2%}'.format(clas_err_min), file=best_res)
        elif args.method == 'overall_robust':
            print('at epoch', t_best, 'achieves')
            print('classification error:', '{:.2%}'.format(clas_err_best))
            print('lowest robust error:', '{:.2%}'.format(robust_err_min))
            print('overall robust model:', t_best, '{:.2%}'.format(clas_err_best), 
                    '{:.2%}'.format(robust_err_min), file=best_res)
        elif args.method == 'lipschitz_robust':
            print('at epoch', t_best, 'achieves')
            print('classification error:', '{:.2%}'.format(clas_err_best))
            print('lowest robust error:', '{:.2%}'.format(robust_err_min))
            print('lipschitz robust model:', t_best, '{:.2%}'.format(clas_err_best), 
                    '{:.2%}'.format(robust_err_min), file=best_res)

        # evaluating the saved best model on the testing dataset
        model.load_state_dict(torch.load(model_path))

        res_filepath = ('./results/'+args.proctitle)
        res_folder= os.path.dirname(res_filepath)
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)

        evaluate_test_clas_spec(test_loader, model, args.epsilon, res_filepath, 
                        args.verbose, norm_type=args.norm_test, bounded_input=True, **kwargs)


    # # evaluate the saved models on customized MNIST dataset        
    # model_path = ('../models/mnist/mnist.pth')
    # model_path = ('../models/'+args.proctitle+'.pth')
    # model.load_state_dict(torch.load(model_path))

    # res_filepath = ('../temporary')
    # evaluate_test_clas_spec(test_loader, model, args.epsilon, res_filepath, 
    #             args.verbose, l1_type=args.l1_test, bounded_input=True, **kwargs)

