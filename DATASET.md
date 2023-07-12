# Preparing Dataset
Our dataloader follows [Detectron2](https://github.com/facebookresearch/detectron2) contains (1) A dataset registrator. (2) A dataset mapper. (3) A dataset loader. We modify the dataset registrator and mapper for different datasets.

## SA-1B Training
Please follow [SAM](https://github.com/facebookresearch/segment-anything) to prepare your datasets.
We recommend you to transfer SAM data into the formats of TSV for faster data loading. We also provide a tsv loader for you.
## COCO
Please Refer to [Mask2Former](https://github.com/facebookresearch/Mask2Former/tree/main/datasets).


