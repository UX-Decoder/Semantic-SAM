# --------------------------------------------------------
# Semantic-SAM: Segment and Recognize Anything at Any Granularity
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Hao Zhang (hzhangcx@connect.ust.hk)
# --------------------------------------------------------

import torch
import numpy as np
from torchvision import transforms
from utils.visualizer import Visualizer
from typing import Tuple
from PIL import Image
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
import cv2
import io
from .automatic_mask_generator import SamAutomaticMaskGenerator
metadata = MetadataCatalog.get('coco_2017_train_panoptic')

def interactive_infer_image(model, image,all_classes,all_parts, thresh,text_size,hole_scale,island_scale,semantic, refimg=None, reftxt=None, audio_pth=None, video_pth=None):
    t = []
    t.append(transforms.Resize(int(text_size), interpolation=Image.BICUBIC))
    transform1 = transforms.Compose(t)
    image_ori = transform1(image['image'])
    mask_ori = transform1(image['mask'])
    width = image_ori.size[0]
    height = image_ori.size[1]
    image_ori = np.asarray(image_ori)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()

    mask_ori = np.asarray(mask_ori)[:,:,0:1].copy()
    mask_ori = torch.from_numpy(mask_ori).permute(2,0,1)[0]

    mask_generator = SamAutomaticMaskGenerator(model)
    outputs = mask_generator.generate(images)

    # batch_inputs = [data]
    # masks,ious = model.model.evaluate_demo(batch_inputs,all_classes,all_parts)
    # masks=outputs[0]['segmentation']
    # ious=outputs[0]['predicted_iou']
    plt.figure(figsize=(10, 10))
    plt.imshow(image_ori)
    show_anns(outputs)
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png',bbox_inches='tight')
    im = Image.open(img_buf)
    # import pdb;pdb.set_trace()
    # img_buf.close()
    return im
    # import pdb;pdb.set_trace()
    # pred_masks_poses = masks
    # reses=[]
    # ious=ious[0,0]
    # ids=torch.argsort(ious,descending=True)
    #
    # text_res=''
    # try:
    #     thresh=float(thresh)
    # except Exception:
    #     thresh=0.0
    # mask_ls=[]
    # ious_res=[]
    # areas=[]
    # for i,(pred_masks_pos,iou) in enumerate(zip(pred_masks_poses[ids],ious[ids])):
    #     iou=round(float(iou),2)
    #     texts=f'{iou}'
    #     mask=(pred_masks_pos>0.0).cpu().numpy()
    #     area=mask.sum()
    #     conti=False
    #     if iou<thresh:
    #         conti=True
    #     for m in mask_ls:
    #         if np.logical_and(mask,m).sum()/np.logical_or(mask,m).sum()>0.95:
    #             conti=True
    #             break
    #     if i == len(pred_masks_poses[ids])-1 and mask_ls==[]:
    #         conti=False
    #     if conti:
    #         continue
    #     ious_res.append(iou)
    #     mask_ls.append(mask)
    #     areas.append(area)
    #     mask,_=remove_small_regions(mask,int(hole_scale),mode="holes")
    #     mask,_=remove_small_regions(mask,int(island_scale),mode="islands")
    #     mask=(mask).astype(np.float)
    #     out_txt = texts
    #     visual = Visualizer(image_ori, metadata=metadata)
    #     color=[0.,0.,1.0]
    #     demo = visual.draw_binary_mask(mask, color=color, text=texts)
    #     res = demo.get_image()
    #     point_x0=max(0,int(point_[0, 1])-3)
    #     point_x1=min(mask_ori.shape[1],int(point_[0, 1])+3)
    #     point_y0 = max(0, int(point_[0, 0]) - 3)
    #     point_y1 = min(mask_ori.shape[0], int(point_[0, 0]) + 3)
    #     res[point_y0:point_y1,point_x0:point_x1,0]=255
    #     res[point_y0:point_y1,point_x0:point_x1,1]=0
    #     res[point_y0:point_y1,point_x0:point_x1,2]=0
    #     reses.append(Image.fromarray(res))
    #     text_res=text_res+';'+out_txt
    # ids=list(torch.argsort(torch.tensor(areas),descending=False))
    # ids = [int(i) for i in ids]
    #
    # torch.cuda.empty_cache()

    # return reses,[reses[i] for i in ids]

def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))