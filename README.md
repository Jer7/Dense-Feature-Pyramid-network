# DFPnet

This respository is the PyTorch implementation of [DFPnet] https://doi.org/10.1007/s00371-020-01887-5

The code was based on a PyTorch implementation of [CE2P](https://arxiv.org/abs/1809.05996) and upon [https://github.com/speedinghzl/pytorch-segmentation-toolbox](https://github.com/speedinghzl/pytorch-segmentation-toolbox), and the data processing is based upon [https://github.com/Microsoft/human-pose-estimation.pytorch](https://github.com/Microsoft/human-pose-estimation.pytorch)

### Requirements

python 3.6   

PyTorch 0.4.1  

To install PyTorch, please refer to https://github.com/pytorch/pytorch#installation.  


Or using anaconda (For linux):  conda env create -f environment.yaml  


Or to use Pytorch 1.0, just replace 'libs' with 'modules' in [https://github.com/mapillary/inplace_abn](https://github.com/mapillary/inplace_abn), and rename it to 'libs'. 

### Compiling

Some parts of InPlace-ABN have a native CUDA implementation, which must be compiled with the following commands:
```bash
cd libs
sh build.sh
python build.py
``` 
The `build.sh` script assumes that the `nvcc` compiler is available in the current system search path.
The CUDA kernels are compiled for `sm_50`, `sm_52` and `sm_61` by default.
To change this (_e.g._ if you are using a Kepler GPU), please edit the `CUDA_GENCODE` variable in `build.sh`.

### Dataset and pretrained model
**Note** that the left and right label should be swapped when the label file is flipped. 

Plesae download [Dog](https://drive.google.com/drive/folders/1dQt98cTkpP6omZ7zKEokNP5Xql5zunf9?usp=sharing) dataset, and put into dataset/Cartoon_sketches directory
or
create symbolic links:
ln -s YOUR_CARTOON_DATASET_DIR dataset/Cartoon_sketches
  
The Cartoon Dataset includes: 

├── train_images   

├── train_segmentations  

├── val_images  

├── val_segmentations  

├── trainTest_images   

├── trainTest_segmentations 

├── train_id.txt  

├── val_id.txt  

├── trainTest_id.txt  

 
Please download imagenet pretrained resent-101 from [Google drive](https://drive.google.com/drive/folders/1dQt98cTkpP6omZ7zKEokNP5Xql5zunf9?usp=sharing), and put it into Trained_Models folder.

### Training and Evaluation
```bash
./run.sh
```
To evaluate the results, please download 'DFPnet_epoch_495.pth' from [Google drive](https://drive.google.com/drive/folders/1dQt98cTkpP6omZ7zKEokNP5Xql5zunf9?usp=sharing), and put into snapshots directory. 
```
./run_evaluate.sh
``` 
The parsing result of the provided 'DFPnet_epoch_495.pth' is 53.88 without any bells and whistles,[68.39](DFPnet_results_epoch490.png)

**Note** that we keep the model at epoch 495 because of its superiority in mIoU over epoch 490. In our paper, we compare every model at epoch 490 for fairness. To see the accuracy evolution of DFPnet in details, download 'Tensorboard_DFPnet_acc_evolution.rar' with [Google drive](https://drive.google.com/drive/folders/1dQt98cTkpP6omZ7zKEokNP5Xql5zunf9?usp=sharing) and use TensorboardX to read files. 

If this code is helpful for your research, please cite the following paper:

    @article{DFPnet, 
	  author = {Wan, Jerome and Mougeot, Guillaume and Yang, Xubo}, 
	  title = {{Dense feature pyramid network for cartoon dog parsing}}, 
	  issn = {0178-2789}, 
	  doi = {10.1007/s00371-020-01887-5}, 
	  abstract = {{While traditional cartoon character drawings are simple for humans to create, it remains a highly challenging task for machines to interpret. Parsing is a way to alleviate the issue with fine-grained semantic segmentation of images. Although well studied on naturalistic images, research toward cartoon parsing is very sparse. Due to the lack of available dataset and the diversity of artwork styles, the difficulty of the cartoon character parsing task is greater than the well-known human parsing task. In this paper, we study one type of cartoon instance: cartoon dogs. We introduce a novel dataset toward cartoon dog parsing and create a new deep convolutional neural network (DCNN) to tackle the problem. Our dataset contains 965 precisely annotated cartoon dog images with seven semantic part labels. Our new model, called dense feature pyramid network (DFPnet), makes use of recent popular techniques on semantic segmentation to efficiently handle cartoon dog parsing. We achieve a mIoU of 68.39\%, a Mean Accuracy of 79.4\% and a Pixel Accuracy of 93.5\% on our cartoon dog validation set. Our method outperforms state-of-the-art models of similar tasks trained on our dataset: CE2P for single human parsing and Mask R-CNN for instance segmentation. We hope this work can be used as a starting point for future research toward digital artwork understanding with DCNN. Our DFPnet and dataset will be publicly available.}}, 
	  pages = {1--13}, 
	  journal = {The Visual Computer}, 
	  year = {2020}
	}
