# Preparing Dataset
Our dataloader follows [Detectron2](https://github.com/facebookresearch/detectron2) contains (1) A dataset registrator. (2) A dataset mapper. (3) A dataset loader. We modify the dataset registrator and mapper for different datasets.

## SA-1B Training
Please follow [SAM](https://github.com/facebookresearch/segment-anything) to prepare your datasets.
We recommend you to transfer SAM data into the formats of TSV for faster data loading. We also provide a [tsv loader](datasets/dataset_mappers/sam_baseline_dataset_mapper.py) for you.
### TSV data preparation

```python
import json
import base64

tsv_file = '/your/save/path'
index_file = '/your/save/path'
f1 = open(tsv_file, 'w')
f2 = open(index_file, 'w')
"""
Example code: write a single image and its json annotation to
    tsv_file: save the image and annotation (forms one data piece)
    index_file: save the tsv index of each data piece
"""
ann_start = 0
json_file = '/your/sam/path/json' 
image_file = '/your/sam/path/image'
ann = json.load(json_file)
anno = json.dumps(ann)
img = open(image_file, 'rb').read()
img = base64.b64encode(img).decode('utf-8')
lent = 0
# save image_file name
length = f1.write("%s\t"%image_file)
lent += length
# save annotation
length = f1.write("%s\t"%anno)
lent += length
# save image
length = f1.write("%s\n"%img)
lent += length
f2.write("%d %d\n"%(ann_start, lent))
ann_start += lent
```
You can refer to this example format to write the original SAM data into tsv format for faster data processing.
### Json file
If you wanna use the original Json format in SA-1B, you can use [this mapper](datasets/dataset_mappers/sam_baseline_dataset_mapper_json.py) we provide.
You can build a `image_list.da` to combine all the json and image file of a directory. Here is the example code.
```python
import torch
import os
sam_path='/your/sam/path/sa_000000'
save_path='/your/sam/path/sa_000000/image_list.da'
f_save = open(save_path, 'wb')
a=[]
files = os.listdir(sam_path)
for f in files:
    if f.split('.')[-1]=='jpg':
        a.append({'img_name': os.path.join(sam_path, f), 'ann_name': os.path.join(sam_path, f.split('.')[0]+'.json')})
torch.save(a, f_save)
f_save.close()
```
## COCO
Please Refer to [MaskDINO](https://github.com/IDEA-Research/MaskDINO/blob/main/README.md).


