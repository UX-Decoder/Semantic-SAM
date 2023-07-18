# --------------------------------------------------------
# Semantic-SAM: Segment and Recognize Anything at Any Granularity
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Hao Zhang (hzhangcx@connect.ust.hk)
# --------------------------------------------------------


import gradio as gr
import torch
import argparse

# from gradio import processing_utils
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from utils.dist import init_distributed_mode
from utils.arguments import load_opt_from_config_file
from utils.constants import COCO_PANOPTIC_CLASSES

from tasks import interactive_infer_image_idino_m2m_auto

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


cur_model = 'None'

'''
build model
'''

model=None
model_size=None
ckpt=None
cfgs={'T':"configs/semantic_sam_only_sa-1b_swinT.yaml",
      'L':"configs/semantic_sam_only_sa-1b_swinL.yaml"}

# audio = whisper.load_model("base")
sam_cfg=cfgs['L']
args.ckpt="/home/t-zhangha/azure_data/output/fengli/joint_part_idino/train_interactive_all_m2m_swinL_bs16_0.1part9_nohash_bs1_resume_all_local_0.15_onlysa_swinL_4node_mnode/model_0099999.pth"
opt = load_opt_from_config_file(sam_cfg)
model_sam = BaseModel(opt, build_model(opt)).from_pretrained(args.ckpt).eval().cuda()
# model_sam = BaseModel(opt, build_model(opt)).eval().cuda()
@torch.no_grad()
def inference(image,*args, **kwargs):
    text_size, hole_scale, island_scale=640,100,100
    text, text_part, text_thresh='','','0.0'
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        semantic=False
        model=model_sam
        a= interactive_infer_image_idino_m2m_auto(model, image,text,text_part,text_thresh,text_size,hole_scale,island_scale,semantic, *args, **kwargs)
        return a



class ImageMask(gr.components.Image):

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", **kwargs)

    def preprocess(self, x):
        return super().preprocess(x)




'''
launch app
'''
title = "SEMANTIC-SAM: SEGMENT AND RECOGNIZE ANYTHING AT ANY GRANULARITY"

article = "The Demo is Run on SEMANTIC SAM."

from detectron2.data import MetadataCatalog
from utils.constants import COCO_PANOPTIC_CLASSES
metadata = MetadataCatalog.get('coco_2017_train_panoptic')
all_classes = [name.replace('-other','').replace('-merged','') for name in COCO_PANOPTIC_CLASSES]
all_parts=['arm', 'beak', 'body', 'cap', 'door', 'ear', 'eye', 'foot', 'hair', 'hand', 'handlebar', 'head', 'headlight', 'horn', 'leg', 'license plate', 'mirror', 'mouth', 'muzzle', 'neck', 'nose', 'paw', 'plant', 'pot', 'saddle', 'tail', 'torso', 'wheel', 'window', 'wing']

demo = gr.Blocks()
image=ImageMask(label="Click on Image (Please only click one point, or our model will take the center of all points as the clicked location. Remember to clear the click after each interaction, or we will take the center of the current click and previous ones as the clicked location.)",type="pil",brush_radius=15.0).style(height=512)
image_out=gr.components.Image(label="Auto generation",type="pil",brush_radius=15.0).style(height=512)

title='''
# Semantic-SAM: Segment and Recognize Anything at Any Granularity

# [[Read our arXiv Paper](https://arxiv.org/pdf/2307.04767.pdf)\] &nbsp; \[[Github page](https://github.com/UX-Decoder/Semantic-SAM)\] 

# Auto generation demo. The demo is a little slow. Please wait for a few minutes.
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

                ],
                inputs=image,

                cache_examples=False,
            )
            with gr.Row(scale=2.0):
                clearBtn = gr.ClearButton(
                    components=[image])
                runBtn = gr.Button("Run")

            with gr.Row(scale=9.0):
                image_out.render()


    title = title,
    article = article,
    allow_flagging = 'never',

    runBtn.click(inference, inputs=[image],
              outputs = image_out)



demo.queue().launch(share=True,server_port=6081)

