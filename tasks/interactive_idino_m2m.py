# --------------------------------------------------------
# SEEM -- Segment Everything Everywhere All At Once
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
# from xdecoder.language.loss import vl_similarity
from utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from typing import Any, Dict, Generator, ItemsView, List, Tuple
import cv2
import os
import glob
import subprocess
from PIL import Image
import random

# t = []
# t.append(transforms.Resize((960,640), interpolation=Image.BICUBIC))
# transform2 = transforms.Compose(t)
from detectron2.data import MetadataCatalog
from utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
metadata = MetadataCatalog.get('coco_2017_train_panoptic')
all_classes = [name.replace('-other','').replace('-merged','') for name in COCO_PANOPTIC_CLASSES]
colors_list = [(np.array(color['color'])/255).tolist() for color in COCO_CATEGORIES] + [[1, 1, 1]]
all_parts=['arm', 'beak', 'body', 'cap', 'door', 'ear', 'eye', 'foot', 'hair', 'hand', 'handlebar', 'head', 'headlight', 'horn', 'leg', 'license plate', 'mirror', 'mouth', 'muzzle', 'neck', 'nose', 'paw', 'plant', 'pot', 'saddle', 'tail', 'torso', 'wheel', 'window', 'wing']
# all_whole=all_classes
def interactive_infer_image(model, image,all_classes,all_parts, thresh,text_size,hole_scale,island_scale,semantic, refimg=None, reftxt=None, audio_pth=None, video_pth=None):
    # if image['image']
    t = []
    t.append(transforms.Resize(int(text_size), interpolation=Image.BICUBIC))
    transform1 = transforms.Compose(t)
    image_ori = transform1(image['image'])
    mask_ori = transform1(image['mask'])
    width = image_ori.size[0]
    height = image_ori.size[1]
    image_ori = np.asarray(image_ori)
    # import pdb;pdb.set_trace()
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()
    all_classes, all_parts=all_classes.strip().strip("\"[]").split(':'),all_parts.strip().strip("\"[]").split(':')
    # stroke_inimg = None
    # stroke_refimg = None

    data = {"image": images, "height": height, "width": width}
    # if len(tasks) == 0:
    #     tasks = ["Panoptic"]
    
    # inistalize task
    # model.model.task_switch['spatial'] = False
    # model.model.task_switch['visual'] = False
    # model.model.task_switch['grounding'] = False
    # model.model.task_switch['audio'] = False

        # overlay = refimg_mask[0,0].float().numpy()[:,:,None] * np.array([0,0,255])
        # x = refimg_ori_np
        # stroke_refimg = x * (1 - refimg_mask[0,0].float().numpy()[:,:,None]) + (x * refimg_mask[0,0].numpy()[:,:,None] * 0.2 + overlay * 0.8)
        # stroke_refimg = Image.fromarray(stroke_refimg.astype(np.uint8))

    # assert 'Stroke' in tasks
    # model.model.task_switch['spatial'] = True
    mask_ori = np.asarray(mask_ori)[:,:,0:1].copy()
    mask_ori = torch.from_numpy(mask_ori).permute(2,0,1)[0]
    points=mask_ori.nonzero().float().to(images.device)
    if len(points)==0:
        point=points.new_tensor([[0.5,0.5,0.006,0.006]])
    else:
        point_=points.mean(0)[None]
        point=point_.clone()
        point[0, 0] = point_[0, 0] / mask_ori.shape[0]
        point[0, 1] = point_[0, 1] / mask_ori.shape[1]
        point = point[:, [1, 0]]
        point=torch.cat([point,points.new_tensor([[0.005,0.005]])],dim=-1)
    # mask_ori = (F.interpolate(mask_ori, (height, width), mode='bilinear') > 0)
    data['targets'] = [dict()]
    data['targets'][0]['points']=point
    data['targets'][0]['pb']=point.new_tensor([0.])


    batch_inputs = [data]
    # if 'Panoptic' in tasks:
    #     model.model.metadata = metadata
    #     results = model.model.evaluate(batch_inputs)
    #     pano_seg = results[-1]['panoptic_seg'][0]
    #     pano_seg_info = results[-1]['panoptic_seg'][1]
    #     demo = visual.draw_panoptic_seg(pano_seg.cpu(), pano_seg_info) # rgb Image
    #     res = demo.get_image()
    #     return Image.fromarray(res), None
    # else:
    masks,ious = model.model.evaluate_demo(batch_inputs,all_classes,all_parts)

    # If contians spatial use spatial:
    # if 'Stroke' in tasks:
    # v_emb = results['pred_maskembs']
    # s_emb = results['pred_pspatials']
    # pred_masks = results['pred_masks']

    # pred_logits = v_emb @ s_emb.transpose(1,2)
    # logits_idx_y = pred_logits[:,:,0].max(dim=1)[1]
    # logits_idx_x = torch.arange(len(logits_idx_y), device=logits_idx_y.device)
    # logits_idx = torch.stack([logits_idx_x, logits_idx_y]).tolist()
    pred_masks_poses = masks
    # pred_classes = logits.max(dim=-1)[1]
    # pred_classes_part = logits_part.max(dim=-1)[1]


    # interpolate mask to ori size
    # pred_masks_pos = (F.interpolate(pred_masks_pos[None,], image_size[-2:], mode='bilinear')[0,:,:data['height'],:data['width']] > 0.0).float().cpu().numpy()
    reses=[]
    # import pdb;pdb.set_trace()
    ious=ious[0,0]
    # if not sort_method=="Area":
    ids=torch.argsort(ious,descending=True)
    # else:
    #     areas=torch.tensor([(pred_masks_pos>0.0).sum() for pred_masks_pos in pred_masks_poses])
    #     ids=list(torch.argsort(areas,descending=False))
    #     ids = [int(i) for i in ids]
    text_res=''
    try:
        thresh=float(thresh)
    except Exception:
        thresh=0.0
    mask_ls=[]
    ious_res=[]
    areas=[]
    # import pdb;pdb.set_trace()
    for i,(pred_masks_pos,iou) in enumerate(zip(pred_masks_poses[ids],ious[ids])):
        # texts = all_classes[pred_class]
        iou=round(float(iou),2)
        # texts = texts+': '+all_parts[pred_class_part]+f' ({iou})'
        # if not semantic:
        texts=f'{iou}'
        # for idx, mask in enumerate(pred_masks_pos):
            # color = random_color(rgb=True, maximum=1).astype(np.int32).tolist()
        mask=(pred_masks_pos>0.0).cpu().numpy()
        area=mask.sum()
        conti=False
        if iou<thresh:
            conti=True
        for m in mask_ls:
            if np.logical_and(mask,m).sum()/np.logical_or(mask,m).sum()>0.95:
                conti=True
                break
        if i == len(pred_masks_poses[ids])-1 and mask_ls==[]:
            conti=False
        if conti:
            continue
        ious_res.append(iou)
        mask_ls.append(mask)
        areas.append(area)
        # import pdb;pdb.set_trace()
        mask,_=remove_small_regions(mask,int(hole_scale),mode="holes")
        mask,_=remove_small_regions(mask,int(island_scale),mode="islands")
        mask=(mask).astype(np.float)
        out_txt = texts
        visual = Visualizer(image_ori, metadata=metadata)
        if semantic:
            color = colors_list[pred_class % 133]
        else:
            color=[0.,0.,1.0]
        demo = visual.draw_binary_mask(mask, color=color, text=texts)
        res = demo.get_image()
        point_x0=max(0,int(point_[0, 1])-3)
        point_x1=min(mask_ori.shape[1],int(point_[0, 1])+3)
        point_y0 = max(0, int(point_[0, 0]) - 3)
        point_y1 = min(mask_ori.shape[0], int(point_[0, 0]) + 3)
        res[point_y0:point_y1,point_x0:point_x1,0]=255
        res[point_y0:point_y1,point_x0:point_x1,1]=0
        res[point_y0:point_y1,point_x0:point_x1,2]=0
        reses.append(Image.fromarray(res))
        text_res=text_res+';'+out_txt
    ids=list(torch.argsort(torch.tensor(areas),descending=False))
    ids = [int(i) for i in ids]
    # ids=[str(id) for id in ids]
    # import pdb;pdb.set_trace()
    torch.cuda.empty_cache()
    # return Image.fromarray(res), stroke_inimg, stroke_refimg
    # image['mask']=Image.fromarray(np.zeros_like(np.asarray(image['mask'])))
    return reses,[reses[i] for i in ids]

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