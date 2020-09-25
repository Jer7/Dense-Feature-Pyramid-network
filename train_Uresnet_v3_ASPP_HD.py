import argparse
import random
import cv2
from PIL import Image 

import torch
torch.multiprocessing.set_start_method("spawn", force=True)
from torch.utils import data
import numpy as np
import torch.optim as optim
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import os
import os.path as osp
from networks.Uresnet_v3_ASPP_HD import Res_Deeplab
from dataset.datasets import cartoonDataSet
import torchvision.transforms as transforms
import timeit
from tensorboardX import SummaryWriter
from utils.utils import decode_parsing, inv_preprocess
from utils.criterion_2 import CriterionAll
from utils.encoding import DataParallelModel, DataParallelCriterion 
from utils.miou import compute_mean_ioU
from evaluate import valid

start = timeit.default_timer()


BATCH_SIZE = 8
DATA_DIRECTORY = './dataset/Cartoon_sketches/Dog/TrainVal'
DATA_LIST_PATH = './dataset/list/cityscapes/train.lst'
IGNORE_LABEL = 255
INPUT_SIZE = '384,384' 
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
NUM_CLASSES = 8
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './Trained_Models/resnet101-imagenet.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 10000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

# BATCH_SIZE = 8
# DATA_DIRECTORY = 'cityscapes'
# DATA_LIST_PATH = './dataset/list/cityscapes/train.lst'
# IGNORE_LABEL = 255
# INPUT_SIZE = '769,769'
# LEARNING_RATE = 1e-2
# MOMENTUM = 0.9
# NUM_CLASSES = 20
# POWER = 0.9
# RANDOM_SEED = 1234
# RESTORE_FROM = './dataset/MS_DeepLab_resnet_pretrained_init.pth'
# SAVE_NUM_IMAGES = 2
# SAVE_PRED_EVERY = 10000
# SNAPSHOT_DIR = './snapshots/'
# WEIGHT_DECAY = 0.0005
 
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Dense Feature pyramid Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--dataset", type=str, default='train', choices=['train', 'val', 'trainval', 'test'],
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).") 
    parser.add_argument("--start-iters", type=int, default=0,
                        help="Number of classes to predict (including background).") 
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--gpu", type=str, default='None',
                        help="choose gpu device.")
    parser.add_argument("--start-epoch", type=int, default=1,
                        help="choose the number of recurrence.")
    parser.add_argument("--epochs", type=int, default=150,
                        help="choose the number of recurrence.")
    return parser.parse_args()


args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, total_iters):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, total_iters, args.power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def adjust_learning_rate_pose(optimizer, epoch):
    decay = 0
    if epoch + 1 >= 230:
        decay = 0.05
    elif epoch + 1 >= 200:
        decay = 0.1
    elif epoch + 1 >= 120:
        decay = 0.25
    elif epoch + 1 >= 90:
        decay = 0.5
    else:
        decay = 1

    lr = args.learning_rate * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003


def main():
    """Create the model and start the training."""

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    writer = SummaryWriter(args.snapshot_dir)
    gpus = [int(i) for i in args.gpu.split(',')]
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    h, w = map(int, args.input_size.split(','))
    input_size = [h, w]

    cudnn.enabled = True
    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
 

    deeplab = Res_Deeplab(num_classes=args.num_classes)
    print(type(deeplab))
    

    # dump_input = torch.rand((args.batch_size, 3, input_size[0], input_size[1]))
    # writer.add_graph(deeplab.cuda(), dump_input.cuda(), verbose=False)


    """
    HOW DOES IT LOAD ONLY RESNET101 AND NOT THE RSTE OF THE NET ?
    """
    # UNCOMMENT THE FOLLOWING COMMENTARY TO INITIALYZE THE WEIGHTS
    
    # Load resnet101 weights trained on imagenet and copy it in new_params
    saved_state_dict = torch.load(args.restore_from)
    new_params = deeplab.state_dict().copy()

    # CHECK IF WEIGHTS BELONG OR NOT TO THE MODEL
    # belongs = 0
    # doesnt_b = 0
    # for key in saved_state_dict:
    #     if key in new_params:
    #         belongs+=1 
    #         print('key=', key)
    #     else:
    #         doesnt_b+=1
    #         # print('key=', key)
    # print('belongs = ', belongs, 'doesnt_b=', doesnt_b)
    # print('res101 len',len(saved_state_dict))
    # print('new param len',len(new_params))


    for i in saved_state_dict:
        i_parts = i.split('.')
        # print('i_parts:', i_parts)
        # exp : i_parts: ['layer2', '3', 'bn2', 'running_mean']

        # The deeplab weight modules  have diff name than args.restore_from weight modules
        if i_parts[0] == 'module' and not i_parts[1] == 'fc' :
            if new_params['.'.join(i_parts[1:])].size() == saved_state_dict[i].size():
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        else:
            if not i_parts[0] == 'fc':
                if new_params['.'.join(i_parts[0:])].size() == saved_state_dict[i].size():
                    new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
 
    deeplab.load_state_dict(new_params)
    
    # UNCOMMENT UNTIL HERE

    model = DataParallelModel(deeplab)
    model.cuda()

    criterion = CriterionAll()
    criterion = DataParallelCriterion(criterion)
    criterion.cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    trainloader = data.DataLoader(cartoonDataSet(args.data_dir, args.dataset, crop_size=input_size, transform=transform),
                                  batch_size=args.batch_size * len(gpus), shuffle=True, num_workers=8,
                                  pin_memory=True)

    #mIoU for Val set
    val_dataset = cartoonDataSet(args.data_dir, 'val', crop_size=input_size, transform=transform)
    numVal_samples = len(val_dataset)
    
    valloader = data.DataLoader(val_dataset, batch_size=args.batch_size * len(gpus),
                                shuffle=False, pin_memory=True)

    #mIoU for trainTest set
    trainTest_dataset = cartoonDataSet(args.data_dir, 'trainTest', crop_size=input_size, transform=transform)
    numTest_samples = len(trainTest_dataset)
    
    testloader = data.DataLoader(trainTest_dataset, batch_size=args.batch_size * len(gpus),
                                shuffle=False, pin_memory=True)


    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    optimizer.zero_grad()
    # valBatch_idx = 0
    total_iters = args.epochs * len(trainloader)
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        for i_iter, batch in enumerate(trainloader):
            i_iter += len(trainloader) * epoch
            lr = adjust_learning_rate(optimizer, i_iter, total_iters)
            images, labels, _, _ = batch
            labels = labels.long().cuda(non_blocking=True)
            preds = model(images)
            # print('preds size in batch', len(preds))
            # print('Size of Segmentation1 tensor output:',preds[0][0].size())
            # print('Segmentation2 tensor output:',preds[0][-1].size())
            # print('Size of Edge tensor output:',preds[1][-1].size())
            loss = criterion(preds, [labels])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_iter % 100 == 0:
                writer.add_scalar('learning_rate', lr, i_iter)
                writer.add_scalar('loss', loss.data.cpu().numpy(), i_iter)

            if i_iter % 500 == 0:
                # print('In iter%500 Size of Segmentation2 GT: ', labels.size())
                # print('In iter%500 Size of edges GT: ', edges.size())
                images_inv = inv_preprocess(images, args.save_num_images)
                # print(labels[0])
                labels_colors = decode_parsing(labels, args.save_num_images, args.num_classes, is_pred=False)
               
                # if isinstance(preds, list):
                #     print(len(preds))
                #     preds = preds[0]
                
                # val_images, _ = valloader[valBatch_idx]
                # valBatch_idx += 1
                # val_sampler = torch.utils.data.RandomSampler(val_dataset,replacement=True, num_samples=args.batch_size * len(gpus))
                # sample_valloader = data.DataLoader(val_dataset, batch_size=args.batch_size * len(gpus),
                #                 shuffle=False, sampler=val_sampler , pin_memory=True)
                # val_images, _ = sample_valloader
                # preds_val = model(val_images)

                # With multiple GPU, preds return a list, therefore we extract the tensor in the list
                if len(gpus)>1:
                    preds= preds[0]
                    # preds_val = preds_val[0]

                
                

                # print('In iter%500 Size of Segmentation2 tensor output:',preds[0][0][-1].size())
                # preds[0][-1] cause model returns [[seg1, seg2], [edge]]
                preds_colors = decode_parsing(preds[0][-1], args.save_num_images, args.num_classes, is_pred=True)
                # preds_val_colors = decode_parsing(preds_val[0][-1], args.save_num_images, args.num_classes, is_pred=True)
                # print("preds type:",type(preds)) #list
                # print("preds shape:", len(preds)) #2
                # hello = preds[0][-1]
                # print("preds type [0][-1]:",type(hello)) #<class 'torch.Tensor'>
                # print("preds len [0][-1]:", len(hello)) #12
                # print("preds len [0][-1]:", hello.shape)#torch.Size([12, 8, 96, 96])
                # print("preds color's type:",type(preds_colors))#torch.tensor
                # print("preds color's shape:",preds_colors.shape) #([2,3,96,96])

                # print('IMAGE', images_inv.size())
                img = vutils.make_grid(images_inv, normalize=False, scale_each=True)
                lab = vutils.make_grid(labels_colors, normalize=False, scale_each=True)
                pred = vutils.make_grid(preds_colors, normalize=False, scale_each=True)
                
                
                # print("preD type:",type(pred)) #<class 'torch.Tensor'>
                # print("preD len:", len(pred))# 3
                # print("preD shape:", pred.shape)#torch.Size([3, 100, 198])

                # 1=head red, 2=body green , 3=left_arm yellow, 4=right_arm blue, 5=left_leg pink
                # 6=right_leg skuBlue, 7=tail grey

                writer.add_image('Images/', img, i_iter)
                writer.add_image('Labels/', lab, i_iter)
                writer.add_image('Preds/', pred, i_iter)
                
               
            print('iter = {} of {} completed, loss = {}'.format(i_iter, total_iters, loss.data.cpu().numpy()))
        
        print('end epoch:', epoch)
        
        if epoch%99 == 0:
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'DFPnet_epoch_' + str(epoch) + '.pth'))
        
        if epoch%5 == 0 and epoch<500:
            # mIou for Val set
            parsing_preds, scales, centers = valid(model, valloader, input_size,  numVal_samples, len(gpus))
            '''
            Insert a sample of prediction of a val image on tensorboard
            '''
            # generqte a rand number between len(parsing_preds)
            sample = random.randint(0, len(parsing_preds)-1)
            
            #loader resize and convert to tensor the image
            loader = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor()
            ])

            # get val segmentation path and open the file
            list_path = os.path.join(args.data_dir, 'val' + '_id.txt')
            val_id = [i_id.strip() for i_id in open(list_path)]
            gt_path = os.path.join(args.data_dir, 'val' + '_segmentations', val_id[sample] + '.png')
            gt =Image.open(gt_path)
            gt = loader(gt)
            #put gt back from 0 to 255
            gt = (gt*255).int()
            # convert pred from ndarray to PIL image then to tensor
            display_preds = Image.fromarray(parsing_preds[sample])
            tensor_display_preds = transforms.ToTensor()(display_preds)
            #put gt back from 0 to 255
            tensor_display_preds = (tensor_display_preds*255).int()
            # color them 
            val_preds_colors = decode_parsing(tensor_display_preds, num_images=1, num_classes=args.num_classes, is_pred=False)
            gt_color = decode_parsing(gt, num_images=1, num_classes=args.num_classes, is_pred=False)
            # put in grid 
            pred_val = vutils.make_grid(val_preds_colors, normalize=False, scale_each=True)
            gt_val = vutils.make_grid(gt_color, normalize=False, scale_each=True)
            writer.add_image('Preds_val/', pred_val, epoch)
            writer.add_image('Gt_val/', gt_val, epoch)

            mIoUval = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size, 'val')

            print('For val set', mIoUval)
            writer.add_scalars('mIoUval', mIoUval, epoch)

            # mIou for trainTest set
            parsing_preds, scales, centers = valid(model, testloader, input_size,  numTest_samples, len(gpus))

            mIoUtest = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size, 'trainTest')

            print('For trainTest set', mIoUtest)
            writer.add_scalars('mIoUtest', mIoUtest, epoch)

        else:
            if epoch%20 == 0 and epoch>=500:
                # mIou for Val set
                parsing_preds, scales, centers = valid(model, valloader, input_size,  numVal_samples, len(gpus))
                '''
                Insert a sample of prediction of a val image on tensorboard
                '''
                # generqte a rand number between len(parsing_preds)
                sample = random.randint(0, len(parsing_preds)-1)
                
                #loader resize and convert to tensor the image
                loader = transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.ToTensor()
                ])

                # get val segmentation path and open the file
                list_path = os.path.join(args.data_dir, 'val' + '_id.txt')
                val_id = [i_id.strip() for i_id in open(list_path)]
                gt_path = os.path.join(args.data_dir, 'val' + '_segmentations', val_id[sample] + '.png')
                gt =Image.open(gt_path)
                gt = loader(gt)
                #put gt back from 0 to 255
                gt = (gt*255).int()
                # convert pred from ndarray to PIL image then to tensor
                display_preds = Image.fromarray(parsing_preds[sample])
                tensor_display_preds = transforms.ToTensor()(display_preds)
                #put gt back from 0 to 255
                tensor_display_preds = (tensor_display_preds*255).int()
                # color them 
                val_preds_colors = decode_parsing(tensor_display_preds, num_images=1, num_classes=args.num_classes, is_pred=False)
                gt_color = decode_parsing(gt, num_images=1, num_classes=args.num_classes, is_pred=False)
                # put in grid 
                pred_val = vutils.make_grid(val_preds_colors, normalize=False, scale_each=True)
                gt_val = vutils.make_grid(gt_color, normalize=False, scale_each=True)
                writer.add_image('Preds_val/', pred_val, epoch)
                writer.add_image('Gt_val/', gt_val, epoch)

                mIoUval = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size, 'val')

                print('For val set', mIoUval)
                writer.add_scalars('mIoUval', mIoUval, epoch)

                # mIou for trainTest set
                parsing_preds, scales, centers = valid(model, testloader, input_size,  numTest_samples, len(gpus))

                mIoUtest = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size, 'trainTest')

                print('For trainTest set', mIoUtest)
                writer.add_scalars('mIoUtest', mIoUtest, epoch)

    end = timeit.default_timer()
    print(end - start, 'seconds')
 

if __name__ == '__main__':
    main()
