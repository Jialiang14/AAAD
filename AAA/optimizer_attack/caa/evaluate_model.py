import argparse
import csv
import os
import random
import time
import builtins
from typing import Dict, List
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
# import torch.nn.parallel
from AAA.optimizer_attack.caa.composite_adv.attacks import *
from AAA.optimizer_attack.caa.composite_adv.utilities import make_dataloader
from math import pi
import warnings
from retrain.PGD_advtrain.models import *

# warnings.filterwarnings('ignore')

# def list_type(s):
#     return tuple(sorted(map(int, s.split(','))))


parser = argparse.ArgumentParser(
    description='Model Robustness Evaluation')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='checkpoint path')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'imagenet', 'svhn'],
                    help='dataset name')
parser.add_argument('--dataset-path', type=str, default='/mnt/jfs/sunjialiang/data',
                    help='path to datasets directory')
parser.add_argument('--batch-size', type=int, default=100,
                    help='number of examples/minibatch')
parser.add_argument('--num-batches', type=int, required=False,
                    help='number of batches (default entire dataset)')
parser.add_argument('--message', type=str, default="",
                    help='csv message before result')
parser.add_argument('--seed', type=int, default=0, help='RNG seed')
parser.add_argument('--output', type=str, help='output CSV')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


def main():
    # settings
    args = parser.parse_args()

    # if args.seed is not None:
    #     # Make sure we can reproduce the testing result.
    #     torch.manual_seed(args.seed)
    #     np.random.seed(args.seed)
    #     random.seed(args.seed)
    #     cudnn.deterministic = True
    #
    # if args.gpu is not None:
    #     warnings.warn('You have chosen a specific GPU. This will completely '
    #                   'disable data parallelism.')
    #
    # if args.dist_url == "env://" and args.world_size == -1:
    #     args.world_size = int(os.environ["WORLD_SIZE"])
    #
    # args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    # ngpus_per_node = torch.cuda.device_count()
    # if args.multiprocessing_distributed:
    #     args.world_size = ngpus_per_node * args.world_size
    #     mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    # else:
        # Simply call main_worker function
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {}".format(args.gpu))

    # if args.distributed:
    #     if args.dist_url == "env://" and args.rank == -1:
    #         args.rank = int(os.environ["RANK"])
    #     if args.multiprocessing_distributed:
    #         # For multiprocessing distributed training, rank needs to be the
    #         # global rank among all the processes
    #         args.rank = args.rank * ngpus_per_node + gpu
    #     dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                             world_size=args.world_size, rank=args.rank)

    # if args.multiprocessing_distributed and args.rank % ngpus_per_node != 0:
    #     def print_pass(*args):
    #         pass
    #     builtins.print = print_pass

    args.model = 'MobileNetV2'
    if args.model == 'VGG':
        model = VGG('VGG19')
    elif args.model == 'ResNet18':
        model = ResNet18()
    elif args.model == 'GoogLeNet':
        model = GoogLeNet()
    elif args.model == 'DenseNet121':
        model = DenseNet121()
    elif args.model == 'DenseNet201':
        model = DenseNet201()
    elif args.model == 'ResNeXt29':
        model = ResNeXt29_2x64d()
    elif args.model == 'ResNeXt29L':
        model = ResNeXt29_32x4d()
    elif args.model == 'MobileNet':
        model = MobileNet()
    elif args.model == 'MobileNetV2':
        model = MobileNetV2()
    elif args.model == 'DPN26':
        model = DPN26()
    elif args.model == 'DPN92':
        model = DPN92()
    elif args.model == 'ShuffleNetG2':
        model = ShuffleNetG2()
    elif args.model == 'SENet18':
        model = SENet18()
    elif args.model == 'ShuffleNetV2':
        model = ShuffleNetV2(1)
    elif args.model == 'EfficientNetB0':
        model = EfficientNetB0()
    elif args.model == 'PNASNetA':
        model = PNASNetA()
    elif args.model == 'RegNetX':
        model = RegNetX_200MF()
    elif args.model == 'RegNetLX':
        model = RegNetX_400MF()
    elif args.model == 'PreActResNet50':
        model = PreActResNet50()

    # model.load_state_dict(torch.load(
    #     '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp_110/eval-MobileNetV2-20230531-031341/weights.pt'))
    model.load_state_dict(torch.load(
        '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/MobileNetV2-20230326-111231/weights.pt'))
    model.cuda()

    # attack_names: List[str] = args.attacks
    # attacks = []
    # for attack_name in attack_names:
    #     tmp = eval(attack_name)
    #     attacks.append(tmp)

    attacks = []
    # for attack_name in attack_names:
    attack_names: List[str] = [
        # "NoAttack()"]
        # "CompositeAttack(model, enabled_attack=(0,), order_schedule='fixed', inner_iter_num=10)" ,
        # "CompositeAttack(model, enabled_attack=(1,), order_schedule='fixed', inner_iter_num=10)" ,
        # "CompositeAttack(model, enabled_attack=(2,), order_schedule='fixed', inner_iter_num=10)" ,
        # "CompositeAttack(model, enabled_attack=(3,), order_schedule='fixed', inner_iter_num=10)" ,
        # "CompositeAttack(model, enabled_attack=(4,), order_schedule='fixed', inner_iter_num=10)" ,
        # "CompositeAttack(model, enabled_attack=(5,), order_schedule='fixed', inner_iter_num=20)" ,
        # "CompositeAttack(model, enabled_attack=(0,1,5), order_schedule='random', inner_iter_num=10)" ,
        # "CompositeAttack(model, enabled_attack=(0,1,5), order_schedule='scheduled', inner_iter_num=10)" ,
        # "CompositeAttack(model, enabled_attack=(3,4,5), order_schedule='random', inner_iter_num=10)" ,
        # "CompositeAttack(model, enabled_attack=(3,4,5), order_schedule='scheduled', inner_iter_num=10)",
        # "CompositeAttack(model, enabled_attack=(0,2,5), order_schedule='random', inner_iter_num=10)" ,
        # "CompositeAttack(model, enabled_attack=(0,2,5), order_schedule='scheduled', inner_iter_num=10)",
        # "CompositeAttack(model, enabled_attack=(0,1,2,3,4), order_schedule='random', inner_iter_num=10)" ,
        # "CompositeAttack(model, enabled_attack=(0,1,2,3,4), order_schedule='scheduled', inner_iter_num=10)" ,
        #                            "CompositeAttack(model, enabled_attack=(1,0,2,0,1,1), order_schedule='fixed', inner_iter_num=1)"]
        "CompositeAttack(model, enabled_attack=(4,), order_schedule='fixed', inner_iter_num=1)"]
    # "CompositeAttack(model, enabled_attack=(0,1,2), order_schedule='scheduled', inner_iter_num=5)" ,
    # "CompositeAttack(model, enabled_attack=(1,0,2,0,1, 1), order_schedule='fixed', inner_iter_num=5)"]
    # tmp = eval("CompositeAttack(model, enabled_attack=(5,), order_schedule='fixed', inner_iter_num=10)")
    for attack_name in attack_names:
        print(attack_name)
        tmp = eval(attack_name)
        attacks.append(tmp)

    # Send to GPU
    # if not torch.cuda.is_available():
    #     print('using CPU, this will be slow')
    # elif args.distributed:
    #     if args.gpu is not None:
    #         torch.cuda.set_device(args.gpu)
    #         model.cuda(args.gpu)
    #         args.batch_size = int(args.batch_size / ngpus_per_node)
    #         args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    #         if args.arch == 'wideresnet':
    #             model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
    #                                                               find_unused_parameters=True)
    #         else:
    #             model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     else:
    #         model.cuda()
    #         model = torch.nn.parallel.DistributedDataParallel(model)
    # elif args.gpu is not None:
    #     torch.cuda.set_device(args.gpu)
    #     model = model.cuda(args.gpu)
    # else:
    #     # DataParallel will divide and allocate batch_size to all available GPUs
    #     if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    #         model.features = torch.nn.DataParallel(model.features)
    #         model.cuda()
    #     else:
    #         model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    test_loader = make_dataloader(args.dataset_path, args.dataset, args.batch_size,
                                  train=False, distributed=False)

    evaluate(model, test_loader, attack_names, attacks, args)


def evaluate(model, val_loader, attack_names, attacks, args):
    model.eval()

    batches_correct: Dict[str, List[torch.Tensor]] = \
        {attack_name: [] for attack_name in attack_names}
    batches_ori_correct: Dict[str, List[torch.Tensor]] = \
        {attack_name: [] for attack_name in attack_names}
    batches_time_used: Dict[str, List[torch.Tensor]] = \
        {attack_name: [] for attack_name in attack_names}

    for batch_index, (inputs, labels) in enumerate(val_loader):
        print(f'BATCH {batch_index:05d}')

        if (
                args.num_batches is not None and
                batch_index >= args.num_batches
        ):
            break

        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)
        else:
            inputs = inputs.cuda()
            labels = labels.cuda()

        for attack_name, attack in zip(attack_names, attacks):
            batch_tic = time.perf_counter()
            adv_inputs = attack(inputs, labels)
            with torch.no_grad():
                ori_logits = model(inputs)
                adv_logits = model(adv_inputs)
            batch_ori_correct = (ori_logits.argmax(1) == labels).detach()
            batch_correct = (adv_logits.argmax(1) == labels).detach()

            batch_accuracy = batch_correct.float().mean().item()
            batch_attack_success_rate = 1.0 - batch_correct[batch_ori_correct].float().mean().item()
            batch_toc = time.perf_counter()
            time_used = torch.tensor(batch_toc - batch_tic)
            print(f'ATTACK {attack_name}',
                  f'accuracy = {batch_accuracy * 100:.1f}',
                  f'attack_success_rate = {batch_attack_success_rate * 100:.1f}',
                  f'time_usage = {time_used:0.2f} s',
                  sep='\t')
            batches_ori_correct[attack_name].append(batch_ori_correct)
            batches_correct[attack_name].append(batch_correct)
            batches_time_used[attack_name].append(time_used)

    print('OVERALL')
    accuracies = []
    attack_success_rates = []
    total_time_used = []
    ori_correct: Dict[str, torch.Tensor] = {}
    attacks_correct: Dict[str, torch.Tensor] = {}
    for attack_name in attack_names:
        ori_correct[attack_name] = torch.cat(batches_ori_correct[attack_name])
        attacks_correct[attack_name] = torch.cat(batches_correct[attack_name])
        accuracy = attacks_correct[attack_name].float().mean().item()
        attack_success_rate = 1.0 - attacks_correct[attack_name][ori_correct[attack_name]].float().mean().item()
        time_used = sum(batches_time_used[attack_name]).item()
        print(f'ATTACK {attack_name}',
              f'accuracy = {accuracy * 100:.1f}',
              f'attack_success_rate = {attack_success_rate * 100:.1f}',
              f'time_usage = {time_used:0.2f} s',
              sep='\t')
        accuracies.append(accuracy)
        attack_success_rates.append(attack_success_rate)
        total_time_used.append(time_used)

    with open(args.output, 'a+') as out_file:
        out_csv = csv.writer(out_file)
        out_csv.writerow([args.message])
        out_csv.writerow(['attack_setting'] + attack_names)
        out_csv.writerow(['accuracies'] + accuracies)
        out_csv.writerow(['attack_success_rates'] + attack_success_rates)
        out_csv.writerow(['time_usage'] + total_time_used)
        out_csv.writerow(['batch_size', args.batch_size])
        out_csv.writerow(['num_batches', args.num_batches])
        out_csv.writerow([''])


if __name__ == '__main__':
    main()