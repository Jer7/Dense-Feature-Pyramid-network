from PIL import Image
import numpy as np
import torchvision
import torch

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


def decode_parsing(labels, num_images=1, num_classes=21, is_pred=False):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    # print('labels',labels.size())
    pred_labels = labels[:num_images].clone().cpu().data
    
    """
    argmax returns the index of the highest value of the matrix. if dim=1, return idx of highest value of each line.
    if dim=0 ... each column.
    """
    if is_pred:
        pred_labels = torch.argmax(pred_labels, dim=1)
    if is_pred == False:
        torch.set_printoptions(threshold=512*512)
        # print('IT IS FALSE', pred_labels)
    # print('pred after', pred_labels.shape)
    n, h, w = pred_labels.size()
    labels_color = torch.zeros([n, 3, h, w], dtype=torch.uint8)
    # print('labl clr size: ', labels_color.size())
    for i, c in enumerate(COLORS):
        c0 = labels_color[:, 0, :, :]
        c1 = labels_color[:, 1, :, :]
        c2 = labels_color[:, 2, :, :]

        c0[pred_labels == i] = c[0]
        c1[pred_labels == i] = c[1]
        c2[pred_labels == i] = c[2]
    
    if is_pred == False:
        torch.set_printoptions(threshold=512*512)
        # print('labels_color mtrx', labels_color)
    return labels_color

def inv_preprocess(imgs, num_images):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    rev_imgs = imgs[:num_images].clone().cpu().data
    rev_normalize = NormalizeInverse(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    for i in range(num_images):
        rev_imgs[i] = rev_normalize(rev_imgs[i])

    return rev_imgs

class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)
