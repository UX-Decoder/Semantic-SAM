# Semantic-SAM: Segment and Recognize Anything at Any Granularity

:grapes: \[[Read our arXiv Paper](https://arxiv.org/pdf/2307.04767.pdf)\] &nbsp; :apple: \[[Try Gradio Demo1](http://exp.xyzou.net:7860/)\]  &nbsp; 🍐: \[[Try Gradio Demo2](http://exp.xyzou.net:7861/)\] 



## Introduction
In this work, we introduce Semantic-SAM, a universal image segmentation model to enable segment and recognize anything at any desired granularity. 
Our model offers the following attributes from instance to part level:
* **Granularity Abundance**. Our model can produce all possible segmentation granularities for a user click with high quality, which enables more **controllable** and **user-friendly** interactive segmentation.
* **Semantic Awareness**. We jointly train SA-1B with semantically labeled datasets to learn the semantics of both instance and part.
* **High Quality**. We base on the DETR-based model to implement both generic and interactive segmentation, and validate that SA-1B helps generic segmentation. The mask quality of multi-granularity is high.

![teaser_xyz](https://github.com/UX-Decoder/Semantic-SAM/assets/11957155/769a0c28-bcdf-42ac-b418-17961c1f2430)

Our model supports a wide range of segmentation tasks and their related applications, including:

* Generic Segmentation
* Part Segmentation
* Interactive Multi-Granularity Segmentation with Semantics
* Multi-Granularity Image Editing

:fire: **Related projects:**

* [Mask DINO](https://github.com/IDEA-Research/MaskDINO): We build upon Mask DINO which is a unified detection and segmentation model to implement our model.
* [OpenSeed](https://github.com/IDEA-Research/OpenSeeD) : Strong open-set segmentation methods based on Mask DINO. We base on it to implement our open-vocabulary segmentation.
* [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once) : Segment using a wide range of user prompts.
* [VLPart](https://github.com/facebookresearch/VLPart) : Going denser with open-vocabulary part segmentation.

## Comparison with SAM and SA-1B Ground-truth
![compare_sam_v3](https://github.com/UX-Decoder/Semantic-SAM/assets/34880758/6c7b50eb-6fe4-4a4f-b3cb-71920e30193e)

(a)(b) are the output masks of our model and SAM, respectively. The red points on the left-most image of each row are the user clicks. (c) shows the GT masks that contain the user clicks. The outputs of our model have been processed to remove duplicates.
## Learned prompt semantics
![levels](https://github.com/UX-Decoder/Semantic-SAM/assets/34880758/d4c3df78-ba07-4f09-9d4f-e5d4f2fc7f45)

We visualize the prediction of each content prompt embedding of points with a fixed order
for our model. We find all the output masks are from small to large. This indicates each prompt
embedding represents a semantic level. The red point in the first column is the click.

## Method
![method_xyz](https://github.com/UX-Decoder/Semantic-SAM/assets/11957155/8e8150a4-a1de-49a6-a817-3c43cf55871b)

## Experiments
We also show that jointly training SA-1B interactive segmentation and generic segmentation can improve the generic segmentation performance.
![coco](https://github.com/UX-Decoder/Semantic-SAM/assets/34880758/b4963761-ef36-47bb-b960-9884b86dce5b)

We also outperform SAM on both mask quality and granularity completeness, please refer to our paper for more experimental details.
