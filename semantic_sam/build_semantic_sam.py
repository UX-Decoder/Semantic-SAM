import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import os

from utils.arguments import load_opt_from_config_file
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from tasks.automatic_mask_generator import SemanticSamAutomaticMaskGenerator
from tasks.interactive_idino_m2m_auto import show_anns


def prepare_image(image_pth):
    """
    apply transformation to the image. crop the image ot 640 short edge by default
    """
    image = Image.open(image_pth).convert('RGB')
    t = []
    t.append(transforms.Resize(640, interpolation=Image.BICUBIC))
    transform1 = transforms.Compose(t)
    image_ori = transform1(image)

    image_ori = np.asarray(image_ori)
    images = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()

    return image_ori, images


def build_semantic_sam(model_type, ckpt):
    """
    build model
    """
    cfgs={'T':"configs/semantic_sam_only_sa-1b_swinT.yaml",
          'L':"configs/semantic_sam_only_sa-1b_swinL.yaml"}

    sam_cfg=cfgs[model_type]
    opt = load_opt_from_config_file(sam_cfg)
    model_semantic_sam = BaseModel(opt, build_model(opt)).from_pretrained(ckpt).eval().cuda()
    return model_semantic_sam


def plot_results(outputs, image_ori, save_path='../vis/'):
    """
    plot input image and its reuslts
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    fig = plt.figure(figsize=(10, 10))
    fig = plt.figure()
    plt.imshow(image_ori)
    plt.savefig('../vis/input.png')
    show_anns(outputs)
    fig.canvas.draw()
    im = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.savefig('../vis/example.png')
    return im