# SparseBEV on nuScenes & custom datasets

The purpose of this document is to provide a guide on how to run any necessary experiments, from inference to fine-tuning, to visualization, with the SparseBEV model. We provide instructions for both nuScenes and custom dataset, referencing often to the original documentation for the nuScenes-related part.

## Setup

The given documentation provided instructions for the setup of the environment. However we found some inconsistencies, often due to the missing support and modernization of the libraries. 

To have a faster experience we have provided a .yml config file to quickly setup the virtual environment. It can be found in the root directory as `sparsebev.yml`.

To create a new environment that contains the correct dependencies with miniconda you can run:

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

The scripts will read information from the `data/` directory - so make sure to have filled the folder - and generate the related infos. It reads from config.py the intrinsics and extrinsics of the camera. In case you are working with your own camera you can edit them in `generatePkl/config.py`.

You can run it with:
```bash
python genTestPkl.py
```

It will produce a .pkl file in the same folder.

### Perform the inference

At this point we're going to run the `inference.py` script as showed:

```bash
export CUDA_VISIBLE_DEVICES=0
python inference.py --config configs/r50_nuimg_704x256.py --weights checkpoints/r50_nuimg_704x256.pth
```

The results are going to be saved in a `results.json` folder, in the nuScenes format.

> *Note:* as for the inference on nuScenes, it is important to have created a directory in the root folder named `checkpoints`, and populate it with the .pth file, that is downloadble via the link provided in the original `README.md`.

## Visualizing the results

### nuScenes

For nuScenes you can follow the instructions provided in the original `README.MD`:

```bash
python viz_bbox_predictions.py --config configs/r50_nuimg_704x256.py --weights checkpoints/r50_nuimg_704x256.pth
```

### Custom Dataset

In this section instead we show how to visualize the results on both the image view or a 3D view.

#### Visualize Bounding Box on the image

For our custom dataset we have to first convert the bboxes in the MMDDET format.

You can do that by running:

```bash
python convert_results_to_mmdet.py --config configs/r50_nuimg_704x256.py --weights checkpoints/r50_nuimg_704x256.pth
```

This script will return in output a `results.pkl`, that will now be used to the final display.

Now with this last script we're going to generate the output images with the bboxes
```bash
python ./visualization/generateImagesBB.py
```

#### Visualize Bounding Box on the point cloud

Put in `./submission/pts_bbox/` the file `info.pkl` containing the intrisic of the camera and `results.pkl` containing the coordinates of the bbox
```bash
python ./visualization/visualizeAllPointCloud.py
```

## Metrics

### Generate GT json
To visualize the metrics first of all convert the ground truth to the same format as the result.json of the inference
```bash
python ./evaluation/generateGT.py
```

### Compute metrics
The following script takes in input the result.json of the inference and the previous generate gt file.
```bash
python ./evaluation/eval.py
```


## Fine-Tuning

Firstly, as for the inference, we have to create the .pkl file. The same procedures that have been discussed for the generation of the test one apply. You can then generate the file using `generatePkl/genTrainPkl.py`.

You can run it with:
```bash
python genTrainPkl.py
```

At this point you are ready to perform the effective fine-tuning. We're going to exploit the `fine_tuning.py` script.

It can be run like this:

You can run it with:
```bash
python -m torch.distributed.run --nproc_per_node=1 fine_tuning.py --config configs/r50_nuimg_704x256.py
```
