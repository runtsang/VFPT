"""
    Calculate and visualize the loss surface.
    Usage example:
    >>  python plot_surface.py --x=-1:1:101 --y=-1:1:101 --model resnet56 --cuda
"""
import argparse
import copy
import h5py
import torch
import time
import socket
import os
import sys
import numpy as np
import torchvision
import torch.nn as nn
import ls_projection as proj
import ls_net_plotter
import ls_scheduler
import ls_mpi4pytorch as mpi

def name_surface_file(args, dir_file):
    # skip if surf_file is specified in args
    if args.surf_file:
        return args.surf_file

    # use args.dir_file as the perfix
    surf_file = dir_file

    # resolution
    surf_file += '_[%s,%s,%d]' % (str(args.xmin), str(args.xmax), int(args.xnum))
    if args.y:
        surf_file += 'x[%s,%s,%d]' % (str(args.ymin), str(args.ymax), int(args.ynum))

    # dataloder parameters
    if args.raw_data: # without data normalization
        surf_file += '_rawdata'
    if args.data_split > 1:
        surf_file += '_datasplit=' + str(args.data_split) + '_splitidx=' + str(args.split_idx)

    return surf_file + ".h5"


def setup_surface_file(args, surf_file, dir_file):
    print(surf_file)
    # skip if the direction file already exists
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r')
        if (args.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            print ("%s is already set up" % surf_file)
            return
        f.close()

    with h5py.File(surf_file, 'a') as f:
        f['dir_file'] = dir_file

        # Create the coordinates(resolutions) at which the function is evaluated
        xcoordinates = np.linspace(args.xmin, args.xmax, num=int(args.xnum))
        f['xcoordinates'] = xcoordinates

        if args.y:
            ycoordinates = np.linspace(args.ymin, args.ymax, num=int(args.xnum))
            f['ycoordinates'] = ycoordinates
        f.close()

    return surf_file


def crunch(surf_file, net, w, s, d, dataloader, loss_key, acc_key, comm, rank, args, criterion=None, cur_device=None):
    """
        Calculate the loss values and accuracies of modified models in parallel
        using MPI reduce.
    """

    f = h5py.File(surf_file, 'r+' if rank == 0 else 'r')
    losses, accuracies = [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    if loss_key not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        if rank == 0:
            f[loss_key] = losses
            f[acc_key] = accuracies
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]

    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    inds, coords, inds_nums = ls_scheduler.get_job_indices(losses, xcoordinates, ycoordinates, comm)

    print('Computing %d values for rank %d'% (len(inds), rank))
    start_time = time.time()
    total_sync = 0.0

    # Loop over all uncalculated loss values
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]

        # Load the weights corresponding to those coordinates into the net
        if args.dir_type == 'weights':
            ls_net_plotter.set_weights(net.module if args.ngpu > 1 else net, w, d, coord)
        elif args.dir_type == 'states':
            ls_net_plotter.set_states(net.module if args.ngpu > 1 else net, s, d, coord)

        # Record the time to compute the loss value
        loss_start = time.time()
        
        ############# Mine #############
        total = 0
        total_loss = 0
        correct = 0
        for idx, input_data in enumerate(dataloader):
            X, targets = get_input(input_data)
            batch_size = X.size(0)
            total += batch_size
            
            cls_weights = dataloader.dataset.get_class_weights("none")
            net = net.cuda()
            train_loss, out = forward_one_batch(net, cur_device, criterion, cls_weights, X, targets, True)
            correct = compute_acc_auc(out, targets)
            total_loss += train_loss.item()*batch_size
        
        loss, acc = total_loss/total, 100.*correct/total
        # loss, acc = evaluation.eval_loss(net, criterion, dataloader, args.cuda)
        
        ############# Mine #############
        
        loss_compute_time = time.time() - loss_start

        # Record the result in the local array
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc

        # Send updated plot data to the master node
        syc_start = time.time()
        losses     = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)
        syc_time = time.time() - syc_start
        total_sync += syc_time

        # Only the master node writes to the file - this avoids write conflicts
        if rank == 0:
            f[loss_key][:] = losses
            f[acc_key][:] = accuracies
            f.flush()

        print('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f' % (
                rank, count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss,
                acc_key, acc, loss_compute_time, syc_time))

    # This is only needed to make MPI run smoothly. If this process has less work than
    # the rank0 process, then we need to keep calling reduce so the rank0 process doesn't block
    for i in range(max(inds_nums) - len(inds)):
        losses = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)

    total_time = time.time() - start_time
    print('Rank %d done!  Total time: %.2f Sync: %.2f' % (rank, total_time, total_sync))

    f.close()

from src.utils.file_io import PathManager
import os
from random import randint
from time import sleep
from src.configs.config import get_cfg
def setup_ablation(args, lr, wd, final_runs, run_idx=None, seed=None):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg.SEED = seed
    # create the clsemb_path for this dataset, only support vitb-sup experiments
    if cfg.DATA.FEATURE == "sup_vitb16_imagenet21k":
        cfg.MODEL.PROMPT.CLSEMB_PATH = os.path.join(
            cfg.MODEL.PROMPT.CLSEMB_FOLDER, "{}.npy".format(cfg.DATA.NAME))

    if not final_runs:
        cfg.RUN_N_TIMES = 1
        cfg.MODEL.SAVE_CKPT = False
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "_val"
        lr = lr / 256 * cfg.DATA.BATCH_SIZE  # update lr based on the batchsize
        cfg.SOLVER.BASE_LR = lr
        cfg.SOLVER.WEIGHT_DECAY = wd

    else:
        cfg.RUN_N_TIMES = 10
        cfg.MODEL.SAVE_CKPT = False
        cfg.MODEL.SAVE_CKPT_FINALRUNS = False # set here to True to enable final saving 
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "_ablation"
        cfg.SOLVER.BASE_LR = lr
        cfg.SOLVER.WEIGHT_DECAY = wd

    # setup output dir
    # output_dir / data_name / feature_name / lr_wd / run1
    output_dir = cfg.OUTPUT_DIR
    output_folder = os.path.join(
        cfg.DATA.NAME, cfg.DATA.FEATURE, f"lr{lr}_wd{wd}"
    )
    # train cfg.RUN_N_TIMES times
    if run_idx is None:
        count = 1
        while count <= 100:
            output_path = os.path.join(output_dir, output_folder, f"run{count}")
            # pause for a random time, so concurrent process with same setting won't interfere with each other. # noqa
            sleep(randint(1, 5))
            if not PathManager.exists(output_path):
                PathManager.mkdirs(output_path)
                cfg.OUTPUT_DIR = output_path
                break
            else:
                count += 1
        if count > 100:
            raise ValueError(
                f"Already run {cfg.RUN_N_TIMES} times for {output_folder}, no need to run more")
    else:
        output_path = os.path.join(output_dir, output_folder, f"run{run_idx}")
        if not PathManager.exists(output_path):
            PathManager.mkdirs(output_path)
            cfg.OUTPUT_DIR = output_path
        else:
            cfg.OUTPUT_DIR = output_path


    cfg.freeze()
    return cfg

def get_input(data):
    if not isinstance(data["image"], torch.Tensor):
        for k, v in data.items():
            data[k] = torch.from_numpy(v)

    inputs = data["image"].float()
    labels = data["label"]
    return inputs, labels

def forward_one_batch(model, device, cls_criterion, cls_weights, inputs, targets, is_train):
    """Train a single (full) epoch on the model using the given
    data loader.

    Args:
        X: input dict
        targets
        is_train: bool
    Returns:
        loss
        outputs: output logits
    """
    # move data to device
    inputs = inputs.to(device, non_blocking=True)    # (batchsize, 2048)
    targets = targets.to(device, non_blocking=True)  # (batchsize, )
    
    # forward
    with torch.set_grad_enabled(is_train):
        outputs = model(inputs)  # (batchsize, num_cls)

        if cls_criterion.is_local() and is_train:
            model.eval()
            loss = cls_criterion(
                outputs, targets, cls_weights,
                model, inputs
            )
        elif cls_criterion.is_local():
            return torch.tensor(1), outputs
        else:
            loss = cls_criterion(
                outputs, targets, cls_weights)


    return loss, outputs

def accuracy(y_probs, y_true):
    from sklearn.metrics import accuracy_score
    # y_prob: (num_images, num_classes)
    y_preds = np.argmax(y_probs.cpu().detach(), axis=1)
    accuracy = accuracy_score(y_true, y_preds, normalize=False)
    error = 1.0 - accuracy
    return accuracy, error


def compute_acc_auc(y_probs, y_true_ids):
    onehot_tgts = np.zeros_like(y_probs.cpu().detach())
    for idx, t in enumerate(y_true_ids):
        onehot_tgts[idx, t] = 1.

    num_classes = y_probs.shape[1]
    if num_classes == 2:
        top1, _ = accuracy(y_probs, y_true_ids)

    top1, _ = accuracy(y_probs, y_true_ids)
    return top1

###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':
    from launch import default_argument_parser_landscape, logging_train_setup
    args = default_argument_parser_landscape().parse_args()
    cfg = setup_ablation(args,  args.lr,  args.wd, final_runs=True, run_idx=1, seed=42)
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # main training / eval actions here

    import random
    import logging
    # fix the seed for reproducibility
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(0)

    # setup training env including loggers
    logging_train_setup(args, cfg)

    from src.data import loader as data_loader
    train_loader = data_loader.construct_train_loader(cfg)
    
    from src.models.build_model import build_model
    model, cur_device = build_model(cfg, vis=True)
    model = model.cuda()
    
    from src.engine.evaluator import Evaluator
    evaluator = Evaluator()
    
    from src.solver.losses import build_loss
    criterion = build_loss(cfg)
    #--------------------------------------------------------------------------
    # Environment setup
    #--------------------------------------------------------------------------
    if args.mpi:
        comm = mpi.setup_MPI()
        rank, nproc = comm.Get_rank(), comm.Get_size()
    else:
        comm, rank, nproc = None, 0, 1

    # in case of multiple GPUs per node, set the GPU to use for each rank
    if args.cuda:
        if not torch.cuda.is_available():
            raise Exception('User selected cuda option, but cuda is not available on this machine')
        gpu_count = torch.cuda.device_count()
        torch.cuda.set_device(rank % gpu_count)
        print('Rank %d use GPU %d of %d GPUs on %s' %
              (rank, torch.cuda.current_device(), gpu_count, socket.gethostname()))

    #--------------------------------------------------------------------------
    # Check plotting resolution
    #--------------------------------------------------------------------------
    try:
        args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
        args.ymin, args.ymax, args.ynum = (None, None, None)
        if args.y:
            args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
            assert args.ymin and args.ymax and args.ynum, \
            'You specified some arguments for the y axis, but not all'
    except:
        raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')

    #--------------------------------------------------------------------------
    # Load models and extract parameters
    #--------------------------------------------------------------------------
    net = model
    w = ls_net_plotter.get_weights(net) # initial parameters
    s = copy.deepcopy(net.state_dict()) # deepcopy since state_dict are references
    if args.ngpu > 1:
        # data parallel with multiple GPUs on a single node
        net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    #--------------------------------------------------------------------------
    # Setup the direction file and the surface file
    #--------------------------------------------------------------------------
    dir_file = ls_net_plotter.name_direction_file(args) # name the direction file
    if rank == 0:
        ls_net_plotter.setup_direction(args, dir_file, net)

    surf_file = name_surface_file(args, dir_file)
    if rank == 0:
        setup_surface_file(args, surf_file, dir_file)

    # wait until master has setup the direction file and surface file
    mpi.barrier(comm)

    # load directions
    d = ls_net_plotter.load_directions(dir_file)
    # calculate the consine similarity of the two directions
    if len(d) == 2 and rank == 0:
        similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
        print('cosine similarity between x-axis and y-axis: %f' % similarity)

    #--------------------------------------------------------------------------
    # Setup dataloader
    #--------------------------------------------------------------------------
    mpi.barrier(comm)

    trainloader = train_loader

    #--------------------------------------------------------------------------
    # Start the computation
    #--------------------------------------------------------------------------
    crunch(surf_file, net, w, s, d, trainloader, 'train_loss', 'train_acc', comm, rank, args, criterion=criterion, cur_device=cur_device)
    # crunch(surf_file, net, w, s, d, testloader, 'test_loss', 'test_acc', comm, rank, args)

    #--------------------------------------------------------------------------
    # Plot figures
    #--------------------------------------------------------------------------
    # if args.plot and rank == 0:
    #     if args.y and args.proj_file:
    #         plot_2D.plot_contour_trajectory(surf_file, dir_file, args.proj_file, 'train_loss', args.show)
    #     elif args.y:
    #         plot_2D.plot_2d_contour(surf_file, 'train_loss', args.vmin, args.vmax, args.vlevel, args.show)
    #     else:
    #         plot_1D.plot_1d_loss_err(surf_file, args.xmin, args.xmax, args.loss_max, args.log, args.show)