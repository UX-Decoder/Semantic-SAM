import os
import logging

import torch
import torch.nn as nn

from utils.model import align_and_update_state_dicts

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self, opt, module: nn.Module):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.model = module

    def forward(self, *inputs, **kwargs):
        outputs = self.model(*inputs, **kwargs)
        return outputs

    def save_pretrained(self, save_dir):
        torch.save(self.model.state_dict(), save_path)

    def from_pretrained(self, load_dir):
        state_dict = torch.load(load_dir, map_location='cpu')
        if 'model' in state_dict:
            state_dict=state_dict['model']
        state_dict={k[6:]:v for k,v in state_dict.items() if k.startswith('model.')}
        # for k in self.model.state_dict():
        #     if k not in state_dict:
        #         assert k[:-2] in state_dict
        #         state_dict[k]=state_dict.pop(k[:-2])
        state_dict = align_and_update_state_dicts(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        return self