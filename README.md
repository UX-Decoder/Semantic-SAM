# Semantic-SAM: Segment and Recognize Anything at Any Granularity

:grapes: \[[Read our arXiv Paper]()\] &nbsp; :apple: \[[Try Gradio Demo]()\] 


## Introduction
In this work, we introduce Semantic-SAM, a universal image segmentation model to enable segment and recognize anything at any desired granularity. 
Our model offers the following attributes from instance to part level:
* Semantic Awareness
* Granularity Abundance

  
![teaser_xyz](https://github.com/UX-Decoder/Semantic-SAM/assets/11957155/b7ebbef7-fc34-4768-9082-cc110951d403)

## Interactive Multi-Granularity Segmentation

## Multi-Granularity Image Editing

## Method
![method_xyz](https://github.com/UX-Decoder/Semantic-SAM/assets/11957155/e392d5a8-2f65-45a6-b786-3b09af15cd33)
## Comparison with SAM and SA-1B Ground-truth
![compare_sam_v3](https://github.com/UX-Decoder/Semantic-SAM/assets/34880758/c137ce09-e1f5-4584-8e47-21887ab20ad1)
(a)(b) are the output masks of our model and SAM, respectively. The red points on the left-most image of each row are the user clicks. (c) shows the GT masks that contain the user clicks. The outputs of our model have been processed to remove duplicates.
## Learned prompt semantics
![levels](https://github.com/UX-Decoder/Semantic-SAM/assets/34880758/965784dc-fa70-46fd-89d2-0722325f4f69)
We visualize the prediction of each content prompt embedding of points with a fixed order
for our model. We find all the output masks are from small to large. This indicates each prompt
embedding represents a semantic level. The red point in the first column is the click.
## Experiments
We also show that jointly training SA-1B interactive segmentation and generic segmentation can improve the generic segmentation performance.
![coco](https://github.com/UX-Decoder/Semantic-SAM/assets/34880758/6f633fa3-7cb3-4ead-a10b-41a3bea3675a)
We also outperform SAM on both mask quality and granularity completeness, please refer to our paper for more experimental details.
