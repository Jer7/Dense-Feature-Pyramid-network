import argparse
import numpy as np
import torch
torch.multiprocessing.set_start_method("spawn", force=True)
from torch.utils import data
from networks.Uresnet_v3_ASPP_HD import Res_Deeplab
from dataset.datasets import cartoonDataSet
import os
import torchvision.transforms as transforms
from utils.miou import compute_mean_ioU
from copy import deepcopy
import skimage.draw

'''
CHANGE THE NETWORK 
'''
DATA_DIRECTORY = './dataset/Cartoon_sketches/Dog/TrainVal'
DATA_LIST_PATH = './dataset/Cartoon_sketches/Dog/val_id.txt'
SAVE_PATH_DIR='/home/jeromewan/SJTU_Thesis/Datasets/Cartoon_sketches/Dog/'
IGNORE_LABEL = 255
NUM_CLASSES = 8
SNAPSHOT_DIR = './snapshots/'
INPUT_SIZE = (473,473)

# colour map
COLORS = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=head red, 2=body green , 3=left_arm yellow, 4=right_arm blue, 5=left_leg pink
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=right_leg skuBlue, 7=tail grey, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CE2P Network")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--dataset", type=str, default='val',
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu(s) device(s).")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--save-path", type=str, default=SAVE_PATH_DIR,
                        help="Path to the directory that will contain the preds")                    

    return parser.parse_args()

def valid(model, valloader, input_size, num_samples, gpus):
    model.eval()

    parsing_preds = np.zeros((num_samples, input_size[0], input_size[1]),
                             dtype=np.uint8)

    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)

    idx = 0
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    with torch.no_grad():
        for index, batch in enumerate(valloader):
            image, meta = batch
            num_images = image.size(0)
            if index % 10 == 0:
                print('%d  processd' % (index * num_images))

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            scales[idx:idx + num_images, :] = s[:, :]
            centers[idx:idx + num_images, :] = c[:, :]

            outputs = model(image.cuda())
            if gpus > 1:
                for output in outputs:
                    parsing = output[0][-1]
                    nums = len(parsing)
                    parsing = interp(parsing).data.cpu().numpy()
                    parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                    parsing_preds[idx:idx + nums, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)

                    idx += nums
            else:
                parsing = outputs[0][-1]
                parsing = interp(parsing).data.cpu().numpy()
                parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                parsing_preds[idx:idx + num_images, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)

                idx += num_images

    parsing_preds = parsing_preds[:num_samples, :, :]


    return parsing_preds, scales, centers

def color_parsing(pred):
    
    labels_color=skimage.color.gray2rgb(pred)
    for i, c in enumerate(COLORS):
        c0 = labels_color[:, :, 0]
        c1 = labels_color[:, :, 1]
        c2 = labels_color[:, :, 2]

        c0[pred == i] = c[0]
        c1[pred == i] = c[1]
        c2[pred == i] = c[2]

    # h, w = pred.shape
    # labels_color = np.zeros([3, h, w], dtype=np.uint8)
    # # print('labl clr size: ', labels_color.size())
    # for i, c in enumerate(COLORS):
    #     c0 = labels_color[0, :, :]
    #     c1 = labels_color[1, :, :]
    #     c2 = labels_color[2, :, :]

    #     c0[pred == i] = c[0]
    #     c1[pred == i] = c[1]
    #     c2[pred == i] = c[2]

    return labels_color

def save_img(mask, img_name):
    args = get_arguments()
    file_name = "{}{}.png".format(args.save_path, img_name)
    skimage.io.imsave(file_name, mask)
    

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    gpus = [int(i) for i in args.gpu.split(',')]


    h, w = map(int, args.input_size.split(','))
    # h, w = args.input_size

    
    input_size = (h, w)

    model = Res_Deeplab(num_classes=args.num_classes)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    lip_dataset = cartoonDataSet(args.data_dir, args.dataset, crop_size=input_size, transform=transform)
    # num_samples = 10
    num_samples = len(lip_dataset)

    #print(lip_dataset[0], num_samples)

    valloader = data.DataLoader(lip_dataset, batch_size=args.batch_size * len(gpus),
                                shuffle=False, pin_memory=True)

    restore_from = args.restore_from

    state_dict = model.state_dict().copy()
    state_dict_old = torch.load(restore_from)

    for key, nkey in zip(state_dict_old.keys(), state_dict.keys()):
        if key != nkey:
            # remove the 'module.' in the 'key'
            state_dict[key[7:]] = deepcopy(state_dict_old[key])
        else:
            state_dict[key] = deepcopy(state_dict_old[key])

    model.load_state_dict(state_dict)

    model.eval()
    model.cuda()

    parsing_preds, scales, centers = valid(model, valloader, input_size, num_samples, len(gpus))
    for i, mask in enumerate(parsing_preds):
        mask = color_parsing(mask)
        print(type(mask), mask.shape)
        save_img(mask,i)

    # print(type(parsing_preds), parsing_preds.shape)


    # mIoU = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size, dataset=args.dataset)
    # print(mIoU)

if __name__ == '__main__':
    main()
