from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, transforms
from models import *
from utils import find_sigm_weights, train, test, print_summary, copy_net2net
from torch.utils.tensorboard import SummaryWriter

def main_train():
    # TODO
    # lr 0.1 or 0.01
    parser = argparse.ArgumentParser(description='PyTorch MNIST/CIFAR10 Training')
    parser.add_argument('--mnist', action='store_true', default=False, help='mnist flag')
    parser.add_argument('--cifar10', action='store_true', default=False, help='cifar10 flag')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M', help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--step-size', type=int, default=100, metavar='M',
                        help='Step size for scheduler (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    parser.add_argument('--load-pre-trained', action='store_true', default=False,
                        help='For Loading Params from Trained Full Precision Model')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N', help='num_workers (default: 4)')
    parser.add_argument('--wd', type=int, default=4, metavar='N', help='wd is 10**((-1)*wd)')
    parser.add_argument('--pd', type=int, default=11, metavar='N', help='pd is 10**((-1)*pd)')
    parser.add_argument('--bn-wd', type=int, default=4, metavar='N', help='pd is 10**((-1)*bn_wd)')
    parser.add_argument('--binary-mode', action='store_true', default=False, help='binary mode bit')
    parser.add_argument('--nohup', action='store_true', default=False, help='nohup mode')
    parser.add_argument('--dont-save', action='store_true', default=False, help='dont_save mode')
    parser.add_argument('--save-file', action='store', default='no_need_to_save', help='name of saved model')
    parser.add_argument('--cudnn', action='store_true', default=False, help='using cudnn benchmark=True')
    parser.add_argument('--suffix', action='store', default='', help='suffix for saved model name')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': args.num_workers,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    best_acc = 0  # best test accuracy
    best_epoch = 0
    best_sampled_acc = 0  # best test accuracy
    best_sampled_epoch = 0
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if args.save_file != 'no_need_to_save':
        writer_suffix = '_' + str(args.save_file)
    else:
        writer_suffix = ''
    writer_name = "runs/" + str('mnist' if args.mnist else 'cifar10') + str('_ver2/' if args.ver2 else '/') + str(
        writer_suffix)
    #     writer_name = "runs/" + str('mnist' if args.mnist else 'cifar10') + str('_new_run')
    writer = SummaryWriter(writer_name)

    # Data
    if args.cifar10:
        print('==> Preparing CIFAR10 data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='../data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, **train_kwargs)

        testset = torchvision.datasets.CIFAR10(
            root='../data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, **test_kwargs)

    elif args.mnist:
        print('==> Preparing MNIST data..')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.MNIST('../data', train=True, download=True,
                                  transform=transform)
        testset = datasets.MNIST('../data', train=False,
                                 transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, **train_kwargs)
        testloader = torch.utils.data.DataLoader(testset, **test_kwargs)
    else:
        print("############################")
        print("## no data set was chosen ##")
        print("############################")
        exit(1)

    # Model
    print('==> Building model..')
    if args.cifar10:
        print("Training LR-Net for CIFAR10")
        net = LRNet_CIFAR10()

        if args.load_pre_trained:
            print("Loading Parameters for CIFAR10")
            test_model = FPNet_CIFAR10().to(device)
            test_model.load_state_dict(torch.load('saved_models/cifar10_fp.pt'))
            alpha1, betta1 = find_sigm_weights(test_model.conv1.weight, False)
            alpha2, betta2 = find_sigm_weights(test_model.conv2.weight, False)
            alpha3, betta3 = find_sigm_weights(test_model.conv3.weight, False)
            alpha4, betta4 = find_sigm_weights(test_model.conv4.weight, False)
            alpha5, betta5 = find_sigm_weights(test_model.conv5.weight, False)
            alpha6, betta6 = find_sigm_weights(test_model.conv6.weight, False)

            net.conv1.initialize_weights(alpha1, betta1)
            net.conv2.initialize_weights(alpha2, betta2)
            net.conv3.initialize_weights(alpha3, betta3)
            net.conv4.initialize_weights(alpha4, betta4)
            net.conv5.initialize_weights(alpha5, betta5)
            net.conv6.initialize_weights(alpha6, betta6)

            state_dict = test_model.state_dict()
            with torch.no_grad():
                net.conv1.bias.copy_(state_dict['conv1.bias'])
                net.conv2.bias.copy_(state_dict['conv2.bias'])
                net.conv3.bias.copy_(state_dict['conv3.bias'])
                net.conv4.bias.copy_(state_dict['conv4.bias'])
                net.conv5.bias.copy_(state_dict['conv5.bias'])
                net.conv6.bias.copy_(state_dict['conv6.bias'])
                net.fc1.weight.copy_(state_dict['fc1.weight'])
                net.fc1.bias.copy_(state_dict['fc1.bias'])
                net.fc2.weight.copy_(state_dict['fc2.weight'])
                net.fc2.bias.copy_(state_dict['fc2.bias'])
    elif args.mnist:
            print("Training LR-Net for MNIST")
            net = LRNet().to(device)

            if args.load_pre_trained:
                print("Loading Parameters for MNIST")
                test_model = FPNet().to(device)
                test_model.load_state_dict(torch.load('saved_models/mnist_fp.pt'))

                alpha1, betta1 = find_sigm_weights(test_model.conv1.weight, False, args.binary_mode)
                alpha2, betta2 = find_sigm_weights(test_model.conv2.weight, False, args.binary_mode)

                net.conv1.initialize_weights(alpha1, betta1)
                net.conv2.initialize_weights(alpha2, betta2)

                state_dict = test_model.state_dict()
                with torch.no_grad():
                    net.conv1.bias.copy_(state_dict['conv1.bias'])
                    net.conv2.bias.copy_(state_dict['conv2.bias'])
                    net.fc1.weight.copy_(state_dict['fc1.weight'])
                    net.fc1.bias.copy_(state_dict['fc1.bias'])
                    net.fc2.weight.copy_(state_dict['fc2.weight'])
                    net.fc2.bias.copy_(state_dict['fc2.bias'])

    if device == 'cuda':
        if args.cudnn:
            print('==> Using cudnn.benchmark = True')
            cudnn.benchmark = True
        else:
            print('==> Using cudnn.benchmark = False && torch.backends.cudnn.deterministic = True')
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    bn_decay = 10 ** ((-1) * args.bn_wd)
    weight_decay = 10 ** ((-1) * args.wd)
    probability_decay = 10 ** ((-1) * args.pd)

    wd_decay = set()
    bias_decay = set()
    prob_decay = set()
    no_decay = set()
    for m in net.modules():
        if isinstance(m, lrnet_nn.LRnetConv2d):
            prob_decay.add(m.alpha)
            prob_decay.add(m.betta)
            bias_decay.add(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            # no_decay.add(m.weight)
            # no_decay.add(m.bias)
            wd_decay.add(m.weight)
            bias_decay.add(m.bias)
        elif isinstance(m, nn.Linear):
            wd_decay.add(m.weight)
            bias_decay.add(m.bias)

    optimizer = optim.Adam(
        [
            {"params": list(prob_decay), "weight_decay": probability_decay},
            {"params": list(wd_decay),   "weight_decay": weight_decay},
            {"params": list(bias_decay), "weight_decay": bn_decay}, # TODO
            {"params": list(no_decay),   "weight_decay": 0}
        ],
        args.lr)

    if args.annealing_sched:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    criterion = nn.CrossEntropyLoss()
    net = net.to(device)

    if args.save_file != 'no_need_to_save':
        file_name = "tmp_logs/" + str(args.save_file) + ".log"
        f = open(file_name, "w")
        print(args, file=f)
    else:
        print(args)
        f = None

    for epoch in range(start_epoch, start_epoch + args.epochs):

        net.train_mode_switch()
        train_acc = train(net, criterion, epoch, device, trainloader, optimizer, args, f, writer)
        writer.add_scalar("acc/train", train_acc, epoch)
        best_acc, best_epoch, test_acc = test(net, criterion, epoch, device, testloader, args, best_acc, best_epoch,
                                              test_mode=False, f=f, eval_mode=True,
                                              dont_save=True)  # note: model is saved only in test method below
        writer.add_scalar("cont acc/test", test_acc, epoch)
        net.test_mode_switch(1, 1)
        best_sampled_acc, best_sampled_epoch, sampled_acc = test(net, criterion, epoch, device, testloader,
                                                                 args, best_sampled_acc, best_sampled_epoch,
                                                                 test_mode=False, f=f, eval_mode=True,
                                                                 dont_save=False)
        print_summary(train_acc, best_acc, best_sampled_acc, sampled_acc, f)
        writer.add_scalar("sampled_acc/test", sampled_acc, epoch)
        scheduler.step()

    writer.flush()
    writer.close()

    if args.save_file != 'no_need_to_save':
        f.close()

if __name__ == '__main__':
    main_train()

