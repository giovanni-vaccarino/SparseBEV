## Setup

We have provided a .yml support file to quickly setup the virtual environment. It can be found in the root directory as `sparsebev.yml`.

To create a new environment that contains the correct dependencies with miniconda:

```bash
conda env create -f sparsebev.yml
```

## Inference on nuScenes

To perform the inference on nuScenes dataset you can follow the instructions provided in the `README.md` in the root directory. A script `val.py` is used for that.

## Inference on custom dataset

In order to make inference it is required to:
1. Including the dataset, populating the `data/` folder, as explained in the original `README.md`
2. Craft our own .pkl file, that embeds the metadata required.
3. Perform the inference

### Including the dataset

The instructions for this part are the same that for the inference on nuScenes. The only constraint is to be coherent with the config variable `dataset_root` in the config file used (under `configs/`).

After loading the dataset, we have to go under a preprocessing phase to adhere the expected format.

In particular, we have to:
1. Ensure that our dataset folder contains all 6 cameras folders
2. Ensure that the 6 folders are named correctly
3. Ensure that the images have been resized to match the format requested (1600 x 900)

#### Populating all the folders

SparseBEV have been trained on nuScenes, that includes 6 cameras(3 for the front part and 3 for the rear). In our case we have only 3 cameras for the front view, however the model architecture still expects to have also the other three cameras. Hence, we use the `/generatePkl/genBlackImages.py` to generate black images for the 3 missing cameras. Before running it make sure to have created in the dataset folder the missing 3 directories: `CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT`.

You can run it with:
```bash
python genBlackImages.py
```

#### Renaming the folders

At this point ensure all the folders have the expected names: `'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'`.

#### Resizing

Finally, we need to match also the requested size of the images, that is 1600x900. To match this we've created a script `generatePkl/resizeImages.py` that helps us with that. 

It will overwrite our images, resizing them to the desired measure.

You can run it with:
```bash
python resizeImages.py
```


### Creation of the pkl

The script we used to generate the .pkl can be found under `generatePkl/genTestPkl.py`.

The scripts will read

### Perform the inference

At this point we're going to run the `inference.py` script as showed:

```bash
export CUDA_VISIBLE_DEVICES=0
python inference.py --config configs/r50_nuimg_704x256.py --weights checkpoints/r50_nuimg_704x256.pth
```

> *Note:* as for the inference on nuScenes, it is important to have created a directory in the root folder named `checkpoints`, and populate it with the .pth file, that is downloadble via the link provided in the original `README.md`.




## Training

Download pretrained weights and put it in directory `pretrain/`:

```
pretrain
├── cascade_mask_rcnn_r101_fpn_1x_nuim_20201024_134804-45215b1e.pth
├── cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth
```

Train SparseBEV with 8 GPUs:

```
torchrun --nproc_per_node 8 train.py --config configs/r50_nuimg_704x256.py
```

Train SparseBEV with 4 GPUs (i.e the last four GPUs):

```
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node 4 train.py --config configs/r50_nuimg_704x256.py
```

The batch size for each GPU will be scaled automatically. So there is no need to modify the `batch_size` in config files.

## Evaluation

Single-GPU evaluation:

```
export CUDA_VISIBLE_DEVICES=0
python inference.py --config configs/r50_nuimg_704x256.py --weights checkpoints/r50_nuimg_704x256.pth
```

Multi-GPU evaluation:

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node 8 inference.py --config configs/r50_nuimg_704x256.py --weights checkpoints/r50_nuimg_704x256.pth
```

## Timing

FPS is measured with a single GPU:

```
export CUDA_VISIBLE_DEVICES=0
python timing.py --config configs/r50_nuimg_704x256.py --weights checkpoints/r50_nuimg_704x256.pth
```

## Visualization

### Visualize Bounding Box on the image

Convert the result.json obtained from the inference to result.pkl:

```
python viz_bbox_predictions.py --config configs/r50_nuimg_704x256.py --weights checkpoints/r50_nuimg_704x256.pth
```

Convert the result.pkl to bbox:
```
python ./visualization/generateImagesBB.py
```

### Visualize Bounding Box on the point cloud
Put in ./submission/pts_bbox/ the file info.pkl containing the intrisic of the camera and result.pkl containing the coordinates of the bbox
```
python ./visualization/visualizeAllPointCloud.py
```
