# Semantic-SAM: Segment and Recognize Anything at Any Granularity
In this work, we introduce **Semantic-SAM**, a universal image segmentation model to enable segment and recognize anything at any desired granularity.
We have trained on the whole **SA-1B** dataset and our model can **reproduce SAM and beyond it**.

:grapes: \[[Read our arXiv Paper](https://arxiv.org/pdf/2307.04767.pdf)\] &nbsp; 

:apple: \[[Try Auto Generation with Controllable Granularity Demo](http://semantic-sam.xyzou.net:6520/)\] &nbsp; :apple: \[[Try Interactive Multi-Granularity Demo](http://semantic-sam.xyzou.net:6081/)\]  &nbsp; 

### :rocket: Features
:fire: **Reproduce SAM**. SAM training is a sub-task of ours. We have released the training code to reproduce SAM training.
  
:fire: **Beyond SAM**. Our newly proposed model offers the following attributes from instance to part level:
* **Granularity Abundance**. Our model can produce all possible segmentation granularities for a user click with high quality, which enables more **controllable** and **user-friendly** interactive segmentation.
* **Semantic Awareness**. We jointly train SA-1B with semantically labeled datasets to learn the semantics at both object-level and part-level.
* **High Quality**. We base on the DETR-based model to implement both generic and interactive segmentation, and validate that SA-1B helps generic and part segmentation. The mask quality of multi-granularity is high.

### :rocket: **News** 
:fire: We release the **demo code for controllable mask auto-generation with different granularity prompts!**
![levels_dog2](https://github.com/UX-Decoder/Semantic-SAM/assets/34880758/2089bd4a-fd9b-4b09-a615-6b373fe38f91)

Segment everything for one image. We output **controllable granularity** masks from **semantic, instance to part level** when using different granularity prompts.

:fire: We release the **demo code for mask auto-generation!**
![tank_auto](https://github.com/UX-Decoder/Semantic-SAM/assets/34880758/ad634168-ea25-451f-960b-918803305073)

Segment everything for one image. We output more masks with more granularity.

:fire: We release the **demo code for interactive segmentation!**
![character](https://github.com/UX-Decoder/Semantic-SAM/assets/34880758/10554e8c-e7cf-463b-875e-0792e629315e)
One click to output up to 6 granularity masks. Try it in our demo!

:fire: We release the **training and inference code and checkpoints (SwinT, SwinL) trained on SA-1B!**

:fire: We release the **training code to reproduce SAM!**

![teaser_xyz](https://github.com/UX-Decoder/Semantic-SAM/assets/11957155/769a0c28-bcdf-42ac-b418-17961c1f2430)

Our model supports a wide range of segmentation tasks and their related applications, including:

* Generic Segmentation
* Part Segmentation
* Interactive Multi-Granularity Segmentation with Semantics
* Multi-Granularity Image Editing


👉: **Related projects:**

* [Mask DINO](https://github.com/IDEA-Research/MaskDINO): We build upon Mask DINO which is a unified detection and segmentation model to implement our model.
* [OpenSeeD](https://github.com/IDEA-Research/OpenSeeD): Strong open-set segmentation methods based on Mask DINO. We base on it to implement our open-vocabulary segmentation.
* [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once): Segment using a wide range of user prompts.
* [VLPart](https://github.com/facebookresearch/VLPart): Going denser with open-vocabulary part segmentation.
## :unicorn: Getting Started

### :hammer_and_wrench: Installation
```shell
pip3 install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
pip install git+https://github.com/cocodataset/panopticapi.git
git clone https://github.com/UX-Decoder/Semantic-SAM
cd Semantic-SAM
python -m pip install -r requirements.txt

export DATASET=/pth/to/dataset  # path to your coco data
```
### :star: A few lines to get generated results
First download a checkpoint from [model zoo](https://github.com/UX-Decoder/Semantic-SAM/releases/tag/checkpoint).
* For interactive multi-granularity segmentation
```python
from semantic_sam import prepare_image, plot_multi_results, build_semantic_sam, SemanticSAMPredictor
original_image, input_image = prepare_image(image_pth='examples/dog.jpg')  # change the image path to your image
mask_generator = SemanticSAMPredictor(build_semantic_sam(model_type='<model_type>', ckpt='</your/ckpt/path>')) # model_type: 'L' / 'T', depends on your checkpint
iou_sort_masks, area_sort_masks = mask_generator.predict_masks(original_image, input_image, point='<your prompts>') # input point [[w, h]] relative location, i.e, [[0.5, 0.5]] is the center of the image
plot_multi_results(iou_sort_masks, area_sort_masks, original_image, save_path='../vis/')  # results and original images will be saved at save_path
```
* For mask auto generation
```python
from semantic_sam import prepare_image, plot_results, build_semantic_sam, SemanticSamAutomaticMaskGenerator
original_image, input_image = prepare_image(image_pth='examples/dog.jpg')  # change the image path to your image
mask_generator = SemanticSamAutomaticMaskGenerator(build_semantic_sam(model_type='<model_type>', ckpt='</your/ckpt/path>')) # model_type: 'L' / 'T', depends on your checkpint
masks = mask_generator.generate(input_image)
plot_results(masks, original_image, save_path='../vis/')  # results and original images will be saved at save_path
```
**Advanced usage:**
* Level is set to [1,2,3,4,5,6] to use all six prompts by default
* You can change the input prompt for controllable mask auto-generation to get the granularity results you want. An example is shown in [here](https://github.com/UX-Decoder/Semantic-SAM/assets/34880758/2089bd4a-fd9b-4b09-a615-6b373fe38f91)
* Here are some examples of `mask_generator` for generating different granularity results
```python
mask_generator = SemanticSamAutomaticMaskGenerator(semantic_sam, level=[1]) # [1] and [2] for semantic level.
mask_generator = SemanticSamAutomaticMaskGenerator(semantic_sam, level=[3]) # [3] for instance level.
mask_generator = SemanticSamAutomaticMaskGenerator(semantic_sam, level=[6]) # [4], [5], [6] for different part level.
```
### :mosque: Data preparation
Please refer to [prepare SA-1B data](DATASET.md). Let us know if you need more instructions about it.

### :volcano: Model Zoo
The currently released checkpoints are only trained with SA-1B data. 
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Training Dataset</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">1-IoU@Multi-Granularity</th>
<th valign="bottom">1-IoU@COCO(Max|Oracle)</th>
<th valign="bottom">download</th>

 <tr><td align="left">Semantic-SAM | <a href="configs/semantic_sam_only_sa-1b_swinT.yaml">config</a></td>
<td align="center">SA-1B</td>
<td align="center">SwinT</td>
<td align="center">88.1</td>
<td align="center">54.5|73.8</td>
<td align="center"><a href="https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swint_only_sam_many2many.pth">model</a></td>
   
 <tr><td align="left">Semantic-SAM | <a href="configs/semantic_sam_only_sa-1b_swinL.yaml">config</a></td>
<td align="center">SA-1B</td>
<td align="center">SwinL</td>
<td align="center">89.0</td>
<td align="center">55.1|74.1</td>
<td align="center"><a href="https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth">model</a></td>

</tbody></table>

### :arrow_forward: Demo
For interactive segmentation.
```shell
python demo.py --ckpt /your/ckpt/path
```
For mask auto-generation.
```shell
python demo_auto_generation.py --ckpt /your/ckpt/path
```

### :sunflower: Evaluation
We do zero-shot evaluation on COCO val2017.
`$n` is the number of gpus you use

For SwinL backbone
```shell
python train_net.py --eval_only --resume --num-gpus $n --config-file configs/semantic_sam_only_sa-1b_swinL.yaml COCO.TEST.BATCH_SIZE_TOTAL=$n  MODEL.WEIGHTS=/path/to/weights
```
For SwinT backbone
```shell
python train_net.py --eval_only --resume --num-gpus $n --config-file configs/semantic_sam_only_sa-1b_swinT.yaml COCO.TEST.BATCH_SIZE_TOTAL=$n  MODEL.WEIGHTS=/path/to/weights
```
### :star: Training 
We currently release the code of training on SA-1B only. Complete training with semantics will be released later.
`$n` is the number of gpus you use
before running the training code, you need to specify your training data of SA-1B.
```shell
export SAM_DATASET=/pth/to/dataset
export SAM_DATASET_START=$start
export SAM_DATASET_END=$end
```
We convert SA-1B data into 100 tsv files. `start`(int, 0-99) is the start of your SA-1B data index and `end`(int, 0-99) is the end of your data index.
If you are not using the tsv data formats, you can refer to this [json registration for SAM](datasets/registration/register_sam_json.py) for a reference. 

For SwinL backbone
```shell
python train_net.py --resume --num-gpus $n  --config-file configs/semantic_sam_only_sa-1b_swinL.yaml COCO.TEST.BATCH_SIZE_TOTAL=$n  SAM.TEST.BATCH_SIZE_TOTAL=$n  SAM.TRAIN.BATCH_SIZE_TOTAL=$n MODEL.WEIGHTS=/path/to/weights
```
For SwinT backbone
```shell
python train_net.py --resume --num-gpus $n  --config-file configs/semantic_sam_only_sa-1b_swinT.yaml COCO.TEST.BATCH_SIZE_TOTAL=$n  SAM.TEST.BATCH_SIZE_TOTAL=$n  SAM.TRAIN.BATCH_SIZE_TOTAL=$n MODEL.WEIGHTS=/path/to/weights
```
**We also support training to reproduce SAM**
```shell
python train_net.py --resume --num-gpus $n  --config-file configs/semantic_sam_reproduce_sam_swinL.yaml COCO.TEST.BATCH_SIZE_TOTAL=$n  SAM.TEST.BATCH_SIZE_TOTAL=$n  SAM.TRAIN.BATCH_SIZE_TOTAL=$n MODEL.WEIGHTS=/path/to/weights
```
This is a swinL backbone. The only difference of this script is to use many-to-one matching and 3 prompts as in SAM.

## 👀 Comparison with SAM and SA-1B Ground-truth
![compare_sam_v3](https://github.com/UX-Decoder/Semantic-SAM/assets/34880758/6c7b50eb-6fe4-4a4f-b3cb-71920e30193e)

(a)(b) are the output masks of our model and SAM, respectively. The red points on the left-most image of each row are the user clicks. (c) shows the GT masks that contain the user clicks. The outputs of our model have been processed to remove duplicates.
## :deciduous_tree: Learned prompt semantics
![levels](https://github.com/UX-Decoder/Semantic-SAM/assets/34880758/d4c3df78-ba07-4f09-9d4f-e5d4f2fc7f45)

We visualize the prediction of each content prompt embedding of points with a fixed order
for our model. We find all the output masks are from small to large. This indicates each prompt
embedding represents a semantic level. The red point in the first column is the click.

## :sauropod: Method
![method_xyz](https://github.com/UX-Decoder/Semantic-SAM/assets/11957155/8e8150a4-a1de-49a6-a817-3c43cf55871b)

## :medal_military: Experiments
We also show that jointly training SA-1B interactive segmentation and generic segmentation can improve the generic segmentation performance.
![coco](https://github.com/UX-Decoder/Semantic-SAM/assets/34880758/b4963761-ef36-47bb-b960-9884b86dce5b)

We also outperform SAM on both mask quality and granularity completeness, please refer to our paper for more experimental details.
<details open>
<summary> <font size=8><strong>:bookmark_tabs: Todo list</strong></font> </summary>

- [x] Release demo
  
- [x] Release code and checkpoints trained on SA-1B

- [ ] Release demo with semantics
  
- [ ] Release code and checkpoints trained on SA-1B and semantically-labeled datasets

</details>

## :hearts: Acknowledgement

Our model is related to [Mask DINO](https://github.com/IDEA-Research/MaskDINO) and [OpenSeeD](https://github.com/IDEA-Research/OpenSeeD). We also thank [Segment Anything](https://github.com/facebookresearch/segment-anything) for the SA-1B data.


## :black_nib: Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.   

```bibtex
@article{li2023semantic,
  title={Semantic-SAM: Segment and Recognize Anything at Any Granularity},
  author={Li, Feng and Zhang, Hao and Sun, Peize and Zou, Xueyan and Liu, Shilong and Yang, Jianwei and Li, Chunyuan and Zhang, Lei and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2307.04767},
  year={2023}
}
}
