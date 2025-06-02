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

Convert the result.json obtained from the inference to result.pkl:

```
python viz_bbox_predictions.py --config configs/r50_nuimg_704x256.py --weights checkpoints/r50_nuimg_704x256.pth
```

Convert the result.pkl to bbox:
```
python ./visualization/generateImagesBB.py
```


Visualize the sampling points (like Fig. 6 in the paper):

```
python viz_sample_points.py --config configs/r50_nuimg_704x256.py --weights checkpoints/r50_nuimg_704x256.pth
```
