# --------------------------------------------------------
# SEEM -- Segment Everything Everywhere All At Once
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu), Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------

# import os
# import warnings
# import PIL
# from PIL import Image
# from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import gradio as gr
import torch
import argparse

# from gradio import processing_utils
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from utils.dist import init_distributed_mode
from utils.arguments import load_opt_from_config_file
from utils.constants import COCO_PANOPTIC_CLASSES

from tasks import interactive_infer_image_idino_m2m

def parse_option():
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    parser.add_argument('--conf_files', default="configs/semantic_sam_only_sa-1b_swinL.yaml", metavar="FILE", help='path to config file', )
    parser.add_argument('--ckpt', default="", metavar="FILE", help='path to ckpt', )
    args = parser.parse_args()

    return args

'''
build args
'''
args = parse_option()


# META DATA
cur_model = 'None'
# if 'focalt' in args.conf_files:
#     pretrained_pth = os.path.join("seem_focalt_v2.pt")
#     if not os.path.exists(pretrained_pth):
#         os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focalt_v2.pt"))
#     cur_model = 'Focal-T'
# elif 'focal' in args.conf_files:
#     pretrained_pth = os.path.join("seem_focall_v1.pt")
#     if not os.path.exists(pretrained_pth):
#         os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt"))
#     cur_model = 'Focal-L'

'''
build model
'''
root="/home/t-zhangha/azure_data/output/"
# root="/mnt/output/"
# model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
model=None
model_size=None
ckpt=None
cfgs={'T':"configs/semantic_sam_only_sa-1b_swinT.yaml",
      'L':"configs/semantic_sam_only_sa-1b_swinL.yaml"}
'''
audio
'''
# audio = whisper.load_model("base")
sam_ckpt=root+"fengli/joint_part_idino/train_interactive_all_m2m_swinL_bs16_0.1part9_nohash_bs1_resume_all_local_0.15_onlysa_swinL_4node_mnode/model_0099999.pth"
sam_cfg=cfgs['L']
# semantic_ckpt=root+"fengli/joint_part_idino/train_interactive_all_m2m_swint_bs16_pas_coco_0.1part9_nohash_bs1_resume_all_local_0.1_hcb/model_0229999.pth"
# semantic_cfg=cfgs['T']
opt = load_opt_from_config_file(sam_cfg)
# opt = init_distributed_mode(opt)
# model_sam = BaseModel(opt, build_model(opt)).eval().cuda()
model_sam = BaseModel(opt, build_model(opt)).from_pretrained(sam_ckpt).eval().cuda()
# opt = load_opt_from_config_files(semantic_cfg)
# opt = init_distributed(opt)
# model_semantic = BaseModel(opt, build_model(opt)).eval().cuda()
# model_semantic = BaseModel(opt, build_model(opt)).from_pretrained(semantic_ckpt).eval().cuda()
@torch.no_grad()
def inference(image,text,text_part,text_thresh,*args, **kwargs):
    # global model_size
    # global ckpt
    # global model
    # global cfgs
    # global root
    # changed=False
    # # import pdb;pdb.set_trace()
    # if text_model_size != model_size or ckpt !=text_ckpt:
    #     model_size=text_model_size
    #     ckpt=text_ckpt
    #     opt = load_opt_from_config_files(cfgs[model_size])
    #     opt = init_distributed(opt)
    #     model = BaseModel(opt, build_model(opt)).from_pretrained(root+ckpt).eval().cuda()
    #     changed=True
        # with torch.no_grad():
        #     model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"],
        #                                                                         is_eval=True)
    text_size, hole_scale, island_scale=640,100,100
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        # if semantic:
        #     model=model_semantic
        # else:
        semantic=False
        model=model_sam
        a,b= interactive_infer_image_idino_m2m(model, image,text,text_part,text_thresh,text_size,hole_scale,island_scale,semantic, *args, **kwargs)
        # prefix="changed to:" if changed else "unchanged:"
        return a,b



class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        return super().preprocess(x)




'''
launch app
'''
title = "SEMANTIC-SAM: SEGMENT AND RECOGNIZE ANYTHING AT ANY GRANULARITY"
description = """
<div style="text-align: center; font-weight: bold;">
    <span style="font-size: 18px" id="paper-info">
        [<a href="https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once" target="_blank">GitHub</a>]
        [<a href="https://arxiv.org/pdf/2304.06718.pdf" target="_blank">arXiv</a>]
    </span>
</div>
<div style="text-align: left; font-weight: bold;">
    <br>
    &#x1F32A Note: The current model is run on <span style="color:blue;">SEMANTIC-SAM {}</span>, for <span style="color:blue;">best performance</span> refer to <a href="https://huggingface.co/spaces/xdecoder/SEEM" target="_blank"><span style="color:red;">our demo</span></a>.
    </p>
</div>
""".format(cur_model)

article = "The Demo is Run on SEMANTIC SAM."

from detectron2.data import MetadataCatalog
from utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
metadata = MetadataCatalog.get('coco_2017_train_panoptic')
all_classes = [name.replace('-other','').replace('-merged','') for name in COCO_PANOPTIC_CLASSES]
all_parts=['arm', 'beak', 'body', 'cap', 'door', 'ear', 'eye', 'foot', 'hair', 'hand', 'handlebar', 'head', 'headlight', 'horn', 'leg', 'license plate', 'mirror', 'mouth', 'muzzle', 'neck', 'nose', 'paw', 'plant', 'pot', 'saddle', 'tail', 'torso', 'wheel', 'window', 'wing']

demo = gr.Blocks()
image=ImageMask(label="Click on Image (Please only click one point, or our model will take the center of all points as the clicked location. Remember to clear the click after each interaction, or we will take the center of the current click and previous ones as the clicked location.)",type="pil",brush_radius=15.0).style(height=512)
gallery_output=gr.Gallery(label="Image Gallery sorted by IoU score.",min_width=1536).style(grid=6)
gallery_output2=gr.Gallery(label="Image Gallery sorted by mask area.",min_width=1536).style(grid=6)
text=gr.components.Textbox(label="Categories. (The default is the categories in COCO panoptic segmentation.)",value=":".join(all_classes),visible=False)
text_part=gr.components.Textbox(label="Part Categories. (The default is the categories in PASCAL Part.)",value=":".join(all_parts),visible=False)
text_res=gr.components.Textbox(label="\"class:part(score)\" of all predictions (seperated by ;): ",visible=True)
text_thresh=gr.components.Textbox(label="The threshold to filter masks with low iou score.",value="0.5",visible=True)
text_size=gr.components.Textbox(label="image size (shortest edge)",value="640",visible=True)
hole_scale=gr.components.Textbox(label="holes scale",value="100",visible=True)
island_scale=gr.components.Textbox(label="island scale",value="100",visible=True)
text_model_size=gr.components.Textbox(label="model size (L or T)",value="L",visible=True)
text_ckpt=gr.components.Textbox(label="ckpt path (relative to /mnt/output/)",value="fengli/joint_part_idino/train_interactive_all_m2m_swinL_bs16_0.1part9_nohash_bs1_resume_all_local_0.15_onlysa_swinL_4node_mnode/model_0099999.pth",visible=True)
text_ckpt_now=gr.components.Textbox(label="current ckpt path (relative to /mnt/output/)",value="",visible=True)
semantic=gr.Checkbox(label="Semantics", info="Do you use semantic? (The semantic model in the demo is trained on SA-1B, COCO and PASCAL Part.)")
# sort_method=gr.Radio(["IoU score", "Area"], label="Sort by", info="")
# def edit():
#     text.update(visible=True)
#     text_part.update(visible=True)
#     text.render()
#     text_part.render()
# checkbox=gr.inputs.CheckboxGroup(choices=["Stroke"], type="value", label="Interative Mode")

title='''
# Semantic-SAM: Segment and Recognize Anything at Any Granularity

# [[Read our arXiv Paper](https://arxiv.org/pdf/2307.04767.pdf)\] &nbsp; \[[Github page](https://github.com/UX-Decoder/Semantic-SAM)\] 

# Please only click one point, or our model will take the center of all points as the clicked location. Remember to clear the click after each interaction, or we will take the center of the current click and previous ones as the clicked location.
'''
def change_vocab(choice):
    if choice:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


with demo:
    with gr.Row():
        with gr.Column(scale=9.0):
            generation_tittle = gr.Markdown(title)
            # generation_tittle.render()
            with gr.Row(scale=20.0):
                image.render()
            example = gr.Examples(
                examples=[
                    ["examples/tank.png"],
                    ["examples/castle.png"],
                    ["examples/fries1.png"],
                    ["examples/4.png"],
                    ["examples/5.png"],
                    ["examples/corgi2.jpg"],
                    ["examples/minecraft2.png"],
                    ["examples/ref_cat.jpeg"],
                    ["examples/img.png"],
                    # ["examples/truck.jpg"],
                    # ["examples/placeholder.png", ["Stroke"]],
                ],
                inputs=image,
                # outputs=gr.Gallery(label="Image Gallery.", min_width=1536).style(grid=6),

                cache_examples=False,
            )
            with gr.Row(scale=1.0):
                # semantic.render()
                with gr.Column():
                    text_thresh.render()
                # with gr.Column(scale=1.0):
                #     text_size.render()
                # with gr.Column(scale=1.0):
                #     text_model_size.render()
                # with gr.Column(scale=1.0):
                #     hole_scale.render()
                # with gr.Column(scale=1.0):
                #     island_scale.render()
                # with gr.Column(scale=10.0):
                #     text_ckpt.render()
                # with gr.Column(scale=10.0):
                #     text_ckpt_now.render()
            with gr.Row(scale=2.0):
                clearBtn = gr.ClearButton(
                    components=[image])
                runBtn = gr.Button("Run")
            with gr.Row(scale=6.0):
                text.render()
            with gr.Row(scale=1.0):
                text_part.render()
            # with gr.Row(scale=1.0):
            #     sort_method.render()
            gallery_tittle = gr.Markdown("# The masks sorted by IoU scores (masks with low score may have low quality).")
            with gr.Row(scale=9.0):
                gallery_output.render()
            gallery_tittle1 = gr.Markdown("# The masks sorted by mask areas.")
            with gr.Row(scale=9.0):
                gallery_output2.render()
            # with gr.Row(scale=1.0):
            #     text_res.render()

    title = title,
    description = description,
    article = article,
    allow_flagging = 'never',

    runBtn.click(inference, inputs=[image, text, text_part,text_thresh],
              outputs = [gallery_output,gallery_output2])
    # semantic.change(fn=change_vocab, inputs=semantic, outputs=text_part)
    # semantic.change(fn=change_vocab, inputs=semantic, outputs=text)
    # semantic.change(fn=change_vocab, inputs=semantic, outputs=text_res)
    # vocabBtn.click(edit)


demo.queue().launch(share=True,server_port=6081)

