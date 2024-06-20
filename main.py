import utils
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.distributed as dist
from torch import multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm import tqdm
from multiprocessing import reduction
import os
import argparse

from models import TVSL
from datasets import get_ac_test_dataset, get_ac_train_dataset, inverse_normalize
import cv2

import builtins
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints',
                        help='path to save trained model weights')
    parser.add_argument('--mode', type=str, default='train',
                        help='train/test')

    # Data params
    parser.add_argument('--trainset', default='vggsound',
                        type=str, help='trainset')
    parser.add_argument('--testset', default='vggsound',
                        type=str, help='testset')
    parser.add_argument('--train_data_path', default='',
                        type=str, help='Root directory path of train data')
    parser.add_argument('--test_data_path', default='',
                        type=str, help='Root directory path of test data')
    parser.add_argument('--test_gt_path', default='', type=str)
    parser.add_argument('--num_test_samples', default=-1, type=int)
    parser.add_argument('--num_class', default=221, type=int)

    parser.add_argument('--model', default='movsl')
    parser.add_argument('--imgnet_type', default='vitb8')
    parser.add_argument('--audnet_type', default='vitb8')

    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--num_negs', default=None, type=int)
    parser.add_argument('--tau', default=0.03, type=float, help='tau')

    parser.add_argument('--attn_assign', type=str, default='soft',
                        help="type of audio grouping assignment")
    parser.add_argument('--dim', type=int, default=512,
                        help='dimensionality of features')
    parser.add_argument('--depth_aud', type=int, default=3,
                        help='depth of audio transformers')
    parser.add_argument('--depth_vis', type=int, default=3,
                        help='depth of visual transformers')

    # training/evaluation parameters
    parser.add_argument("--epochs", type=int, default=20,
                        help="number of epochs")
    parser.add_argument('--batch_size', default=128,
                        type=int, help='Batch Size')
    parser.add_argument("--optimizer", default='adam',
                        help="training optimizer")
    parser.add_argument("--lr_schedule", default='cte',
                        help="learning rate schedule")
    parser.add_argument("--init_lr", type=float,
                        default=0.0001, help="initial learning rate")
    parser.add_argument("--warmup_epochs", type=int,
                        default=0, help="warmup epochs")
    parser.add_argument("--seed", type=int, default=12345, help="random seed")
    parser.add_argument('--weight_decay', type=float,
                        default=0, help='Weight Decay')
    parser.add_argument("--clip_norm", type=float,
                        default=0, help="gradient clip norm")
    parser.add_argument("--dropout_img", type=float,
                        default=0, help="dropout for image")
    parser.add_argument("--dropout_aud", type=float,
                        default=0, help="dropout for audio")

    parser.add_argument('--iou_thr', default=0.3, type=float)
    parser.add_argument('--ciou_thr', default=0.1, type=float)

    # Distributed params
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--ngpu', type=int, default=None)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str,
                        default='tcp://localhost:12345')
    parser.add_argument('--multiprocessing_distributed', action='store_true')

    # additional params
    parser.add_argument('--id', default='',
                        help="a name for identifying the model")
    parser.add_argument('--vis_encoder_type', type=str,
                        default='vit', help="type of transformer backbone")
    parser.add_argument('--vit_type', type=str, default="base",
                        help="type of transformer backbone")
    parser.add_argument("--load", type=str,
                        default='', help="model path")
    parser.add_argument("--audioclip_ckpt_path", type=str,
                        default='/path/to/audioclip', help="audioclip pretrained model path")
    parser.add_argument("--output_dir", type=str,
                        default='outputs/', help="model save dir")
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--num_vis', type=int, default=20)
    parser.add_argument('--save_visualizations', action="store_true")

    return parser.parse_args()


def main(args):
    args.id += '-{}'.format(args.model)
    args.id += '-mode-{}'.format(args.mode)
    args.id += '-epoch{}'.format(args.epochs)
    args.id += '-batch{}'.format(args.batch_size)
    args.id += '-lr{}'.format(args.init_lr)
    args.id += '-ngpu{}'.format(args.ngpu)
    args.id += '-seed{}'.format(args.seed)

    print('Model ID: {}'.format(args.id))

    # paths to save/load output
    args.output_dir = os.path.join(args.output_dir, args.id)

    args.vis = os.path.join(args.output_dir, 'visualization')
    args.ckpt = os.path.join(args.output_dir, "checkpoints")

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.isdir(args.vis):
        os.makedirs(args.vis)

        if not os.path.isdir(os.path.join(args.vis, "val")):
            os.makedirs(os.path.join(args.vis, "val"))

        if not os.path.isdir(os.path.join(args.vis, "test")):
            os.makedirs(os.path.join(args.vis, "test"))

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    args.log_fn = f"{args.output_dir}/train.log"

    if os.path.isfile(args.log_fn):
        os.remove(args.log_fn)

    # Create model dir
    utils.save_json(vars(args), os.path.join(args.output_dir,
                    'configs.json'), sort_keys=True, save_pretty=True)

    mp.set_start_method('spawn')
    args.dist_url = f'tcp://{args.node}:{args.port}'
    print('Using url {}'.format(args.dist_url))

    ngpus_per_node = args.ngpu if args.ngpu else torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node
        mp.spawn(main_worker,
                 nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))

    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # Setup distributed environment
    if args.multiprocessing_distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # Create model dir
    model_dir = args.ckpt
    # model_dir = os.path.join(args.model_dir, args.experiment_name)
    os.makedirs(model_dir, exist_ok=True)
    utils.save_json(vars(args), os.path.join(model_dir, 'configs.json'), sort_keys=True, save_pretty=True)

    # logger
    def print_and_log(*content, **kwargs):
        # suppress printing if not first GPU on each node
        if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
            return
        msg = ' '.join([str(ct) for ct in content])
        sys.stdout.write(msg+'\n')
        sys.stdout.flush()
        with open(args.log_fn, 'a') as f:
            f.write(msg+'\n')

    builtins.print = print_and_log

    model = TVSL(
        pretrained=args.audioclip_ckpt_path
    )

    print("Model is loaded!")

    # Count paramters
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(
        f"Total Params: {total_params/1000000: 6.4f} M, or {total_params/1000000: .4e} M")
    print(
        f"Trainable Params: {trainable_params/1000000: 6.4f} M or {trainable_params/1000000: .4e} M")
   
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.multiprocessing_distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    print(model)

    # Optimizer
    if args.optimizer == "adam":
        optimizer, scheduler = utils.build_optimizer_and_scheduler_adam(
            model, args)
    elif args.optimizer == "sgd":
        optimizer, scheduler = utils.build_optimizer_and_scheduler_sgd(
            model, args)

    args.viz_dir = os.path.join(args.vis, f"test_{args.testset}") 

    # History of performance
    history = {
        'train': {'epoch': [], 'loss': [], 'mcl_loss': [], 'audio_loss': [], 'image_loss': [], 'cnt_loss': []},
        'test': {'epoch': [], 'ciou': [], 'auc': [], 'ap': []}}

    # Resume if possible
    start_epoch, best_ciou, best_auc, best_ap = 0, 0., 0., 0.

    if args.resume:
        if os.path.exists(os.path.join(model_dir, 'latest.pth')):
            ckp = torch.load(os.path.join(
                model_dir, 'latest.pth'), map_location='cpu')
            start_epoch, best_precision, best_ap, best_f1 = ckp['epoch'], ckp[
                'best_Precision'], ckp['best_AP'], ckp['best_F1']
            model.load_state_dict(ckp['model'], strict=False)
            optimizer.load_state_dict(ckp['optimizer'])
            history = ckp['history']
            print(f'loaded from {os.path.join(model_dir, "latest.pth")}')

    if args.load:
        ckp = torch.load(args.load, map_location='cpu')
        # start_epoch, best_precision, best_ap, best_f1 = ckp['epoch'], ckp[
        #     'best_Precision'], ckp['best_AP'], ckp['best_F1']
        model.load_state_dict(ckp['model'], strict=False)
        # optimizer.load_state_dict(ckp['optimizer'])
        # history = ckp['history']
        print(f'loaded from {os.path.join(model_dir, "latest.pth")}')

    torch.cuda.empty_cache()

    # Dataloaders
    traindataset = get_ac_train_dataset(args)
    train_sampler = None
    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            traindataset)
    train_loader = torch.utils.data.DataLoader(
        traindataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True,
        persistent_workers=args.workers > 0)

    testdataset = get_ac_test_dataset(args)

    if args.multiprocessing_distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            testdataset)

    test_loader = torch.utils.data.DataLoader(
        testdataset, batch_size=32, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=test_sampler, drop_last=False,
        persistent_workers=args.workers > 0)
    print("Loaded dataloader.")

    print(f"Size of Train dataset: {len(traindataset)}")
    print(f"Size of Test dataset: {len(testdataset)}")

    args.epoch_iters = len(train_loader)
    print('1 Epoch = {} iters'.format(args.epoch_iters))

    # =============================================================== #
    # # Training loop
    if args.testset in {'vgginstruments_multi', 'music_duet', 'vggsound_duet'}:
        ciou, auc, ap = validate_multi(
            test_loader, model, 0, history, args)

        print(f'cIoU@50 (epoch {0}): {ciou}')
        print(f'AUC@50 (epoch {0}): {auc}')
        print(f'AP@50 (epoch {0}): {ap}')

    else:
        precision, ap, f1 = validate(test_loader, model, 0, history, args)

        print(f'Precision@30 (epoch {start_epoch}): {precision}')
        print(f'AP@30 (epoch {start_epoch}): {ap}')
        print(f'F1@30 (epoch {start_epoch}): {f1}')
        print(f'PIAP (epoch {start_epoch}): {ap}')
        
        print(f'best_Precision@30: {best_precision}')
        print(f'best_AP@30: {best_ap}')
        print(f'best_F1@30: {best_f1}')
        print(f'best_PIAP: {best_piap}')
    
    if args.mode == "test":
        return

    metric_list = [[] for _ in range(3)]

    for epoch in range(start_epoch+1, args.epochs+1):
        if args.multiprocessing_distributed:
            train_loader.sampler.set_epoch(epoch)

        # Train
        train(train_loader, test_loader, model, optimizer, epoch, history, args)
        torch.cuda.empty_cache()

        # Evaluate
        if args.testset in {'vgginstruments_multi', 'vggsound_duet', 'music_duet'}:
            ciou, auc, ap = validate_multi(
                test_loader, model, epoch, history, args)
        else:
            precision, ap, f1 = validate(
                test_loader, model, epoch, history, args)
        
        if ciou >= best_ciou:
            best_ciou = ciou
        if auc >= best_auc:
            best_auc = auc
        if ap >= best_ap:
            best_ap = ap

        print(f'best_cIoU@50: {best_ciou}')
        print(f'best_AUC@50: {best_auc}')
        print(f'best_AP@50: {best_ap}')

        metric_list[0].append(ciou)
        metric_list[1].append(auc)
        metric_list[2].append(ap)

        # Checkpoint
        if args.rank == 0:
            ckp = {'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   'epoch': epoch+1,
                   'AP': ap,
                   'AUC': auc,
                   'CIoU': ciou,
                   'history': history}

            torch.save(ckp, os.path.join(args.ckpt, 'model_latest.pth'))
            torch.save(history, os.path.join(args.ckpt, 'history_latest.pth'))

            if ap == best_ap:
                torch.save(ckp, os.path.join(args.ckpt, 'model_best.pth'))

            print(f"Model saved to {model_dir}")

        torch.distributed.barrier()

    np.save(os.path.join(args.ckpt, 'metrics.npy'), np.array(metric_list))


def train(train_loader, test_loader, model, optimizer, epoch, history, args):
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_mtr = AverageMeter('Loss', ':.3f')
    loss_mcl_mtr = AverageMeter('Multi Cls Loss', ':.3f')
    loss_aud_mtr = AverageMeter('Audio Cls Loss', ':.3f')
    loss_img_mtr = AverageMeter('Img Cls Loss', ':.3f')
    loss_cnt_mtr = AverageMeter('Cnt Loss', ':.3f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, loss_mtr, loss_mcl_mtr, loss_aud_mtr,
            loss_img_mtr, loss_cnt_mtr],
        prefix="Epoch: [{}]".format(epoch),
    )

    end = time.time()
    for i, (image, audio, anno, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        global_step = i + len(train_loader) * epoch
        utils.adjust_learning_rate(
            optimizer, epoch + i / len(train_loader), args)

        if args.gpu is not None:
            audio = audio.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)
            anno['label1'] = anno['label1'].cuda(args.gpu, non_blocking=True)
            anno['label2'] = anno['label2'].cuda(args.gpu, non_blocking=True)        

        loss, loss_dict = model(
            image.float(), audio.float(), anno)
        
        loss_mcl, loss_aud_cls = loss_dict["mcl_loss"], loss_dict["audio_cls_loss"]
        loss_img_cls, cnt_loss = loss_dict["image_cls_loss"], loss_dict["contrastive_loss"]

        loss_mtr.update(loss.item(), image.shape[0])
        loss_mcl_mtr.update(loss_mcl.item(), image.shape[0])
        loss_aud_mtr.update(loss_aud_cls.item(), image.shape[0])
        loss_img_mtr.update(loss_img_cls.item(), image.shape[0])
        loss_cnt_mtr.update(cnt_loss.item(), image.shape[0])

        optimizer.zero_grad()
        loss.backward()

        # gradient clip
        if args.clip_norm != 0:
            nn.utils.clip_grad_norm_(
                model.parameters(), args.clip_norm)  # clip gradient

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        fractional_epoch = epoch + 1. * i / args.epoch_iters
        history['train']['epoch'].append(fractional_epoch)
        history['train']['loss'].append(loss_mtr.avg)
        history['train']['mcl_loss'].append(loss_mcl_mtr.avg)
        history['train']['audio_loss'].append(loss_aud_mtr.avg)
        history['train']['image_loss'].append(loss_img_mtr.avg)
        history['train']['cnt_loss'].append(loss_cnt_mtr.avg)

        if i % 10 == 0 or i == len(train_loader) - 1:
            progress.display(i)
        
        if args.rank == 0:
            if i == 150:
                ckp = {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch+1,
                    'AP': ap,
                    'AUC': auc,
                    'CIoU': ciou,
                    'history': history}

                torch.save(ckp, os.path.join(args.ckpt, f'model_iter{i}_ep_{epoch}.pth'))
                torch.save(history, os.path.join(args.ckpt, f'history_iter{i}_ep_{epoch}.pth'))
                
                dist.barrier()

        if i % 50 == 0 and i != 0:
            if args.testset in {'vgginstruments_multi', 'vggsound_duet', 'music_duet'}:
                ciou, auc, ap = validate_multi(
                    test_loader, model, epoch, history, args)
            
                print(f'cIoU@50 (epoch {epoch} - Iter {i}): {ciou}')
                print(f'AUC@50 (epoch {epoch} - Iter {i}): {auc}')
                print(f'AP@50 (epoch {epoch} - Iter {i}): {ap}')

            else:
                precision, ap, f1 = validate(
                    test_loader, model, epoch, history, args)
            
            model.train()
            
        del loss

@torch.no_grad()
def validate_multi(test_loader, model, epoch, history, args):
    model.train(False)

    evaluator = utils.EvaluatorNew(iou_thr=args.iou_thr, ciou_thr=args.ciou_thr, results_dir=f"{args.viz_dir}")

    k_viz = 0

    for step, (image, audio, anno, name) in enumerate(test_loader):
        if args.gpu is not None:
            audio = audio.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)

            m, n = image.shape[2] // 224, image.shape[3] // 224
            
            heat_map_1, heat_map_2 = model(image.float(), audio.float(), anno)
            # heat_map_1 = model(image.float(), audio.float(), anno['class1'])
            heat_map_1 = F.interpolate(heat_map_1, size=(
                224*m, 224*n), mode='bicubic', align_corners=False)
            heat_map_1 = heat_map_1.data.cpu().numpy()
            
            # heat_map_2 = model(image.float(), audio.float(), anno['class2'])
            heat_map_2 = F.interpolate(heat_map_2, size=(
                224*m, 224*n), mode='bicubic', align_corners=False)
            heat_map_2 = heat_map_2.data.cpu().numpy()

            for i in range(image.shape[0]):
                # predicting for first class
                gt_map = anno['gt_map1'][i].data.cpu().numpy()
                x = np.zeros((224, 224))
                gt_map = np.concatenate([gt_map, x], axis=-1)
                
                bb = anno['bboxes1'][i]
                bb = bb[bb[:, 0] >= 0].numpy().tolist()

                scores = heat_map_1[i, 0]

                scores = utils.min_max_norm(heat_map_1[i, 0], heat_map_1[i, 0].min(), heat_map_1[i, 0].max())

                scores = utils.inv_normalize_img(scores)

                evaluator.cal_CIOU(scores, gt_map, m=2)
                evaluator.cal_IOU(scores, gt_map, m=2)
                evaluator.calc_AP(scores, gt_map)

                if args.save_visualizations and k_viz < args.num_vis and args.gpu == 0:
                    evaluator.save_viz_ac(image[i], bb, scores, name[i], query=anno['class1'][i])

                # predicting for second class
                gt_map = anno['gt_map2'][i].data.cpu().numpy()
                x = np.zeros((224, 224))
                gt_map = np.concatenate([x, gt_map], axis=-1)
                bb = anno['bboxes2'][i]
                bb = bb[bb[:, 0] >= 0].numpy().tolist()

                scores = heat_map_2[i, 0]

                scores = utils.min_max_norm(heat_map_2[i, 0], heat_map_2[i, 0].min(), heat_map_2[i, 0].max())

                scores = utils.inv_normalize_img(scores)

                evaluator.cal_CIOU(scores, gt_map, m=2)
                evaluator.cal_IOU(scores, gt_map, m=2)
                evaluator.calc_AP(scores, gt_map)

                if args.save_visualizations and k_viz < args.num_vis and args.gpu == 0:
                    evaluator.save_viz_ac(image[i], bb, scores, name[i], query=anno['class2'][i])
                    k_viz += 1

            if step % 10 == 0:
                print(f'{step+1}/{len(test_loader)}')

    print('='*20 + 'AudioCLIP ' + '='*20)
    ciou, iou, auc, ap = evaluator.finalize_results()

    print(f"Epoch {epoch}: CIoU@{args.ciou_thr}: {ciou}, IoU@{args.iou_thr}: {iou}, AUC: {auc}, AP: {ap}")


    history['test']['epoch'].append(epoch)
    history['test']['ciou'].append(ciou)
    history['test']['auc'].append(auc)
    history['test']['ap'].append(ap)
    
    
    for mode in history.keys():
        for key in history[mode]:
            val = torch.tensor(history[mode][key], dtype=torch.float32).cuda(
                args.gpu, non_blocking=True)
            dist.all_reduce(val, dist.ReduceOp.SUM, async_op=False)
            val = (val / args.world_size).tolist()
            history[mode][key] = val

    return ciou, auc, ap


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", fp=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.fp = fp

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        msg = '\t'.join(entries)
        print(msg, flush=True)
        if self.fp is not None:
            self.fp.write(msg+'\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == "__main__":
    main(get_arguments())
