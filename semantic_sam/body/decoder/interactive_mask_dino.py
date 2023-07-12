# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
import logging
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d
from detectron2.utils.registry import Registry
from detectron2.structures import BitMasks
from timm.models.layers import trunc_normal_

from .registry import register_decoder
from .utils.dino_decoder import TransformerDecoder, DeformableTransformerDecoderLayer
from .utils import MLP, gen_encoder_output_proposals, inverse_sigmoid
from ...utils import box_ops
from ...utils import configurable

class IMaskDINODecoder(nn.Module):
    @configurable
    def __init__(
            self,
            lang_encoder: nn.Module,
            in_channels,
            mask_classification=True,
            *,
            num_classes: int,
            hidden_dim: int,
            dim_proj: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            mask_dim: int,
            enforce_input_project: bool,
            two_stage: bool,
            dn: str,
            noise_scale:float,
            dn_num:int,
            initialize_box_type:bool,
            initial_pred:bool,
            learn_tgt: bool,
            total_num_feature_levels: int = 4,
            dropout: float = 0.0,
            activation: str = 'relu',
            nhead: int = 8,
            dec_n_points: int = 4,
            return_intermediate_dec: bool = True,
            query_dim: int = 4,
            dec_layer_share: bool = False,
            semantic_ce_loss: bool = False,
            num_mask_tokens: int = 3,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
            d_model: transformer dimension
            dropout: dropout rate
            activation: activation function
            nhead: num heads in multi-head attention
            dec_n_points: number of sampling points in decoder
            return_intermediate_dec: return the intermediate results of decoder
            query_dim: 4 -> (x, y, w, h)
            dec_layer_share: whether to share each decoder layer
            semantic_ce_loss: use ce loss for semantic segmentation
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        self.num_feature_levels = total_num_feature_levels
        self.initial_pred = initial_pred

        # define Transformer decoder here
        self.dn=dn
        self.learn_tgt = learn_tgt
        self.noise_scale=noise_scale
        self.dn_num=dn_num
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.two_stage=two_stage
        self.initialize_box_type = initialize_box_type
        self.total_num_feature_levels = total_num_feature_levels

        self.num_queries = num_queries
        
        self.semantic_ce_loss = semantic_ce_loss
        interactive_only = True
        # learnable query features
        if num_queries>0 and not interactive_only:
            if not two_stage or self.learn_tgt:
                self.query_feat = nn.Embedding(num_queries, hidden_dim)
            if not two_stage and initialize_box_type == 'no':
                self.query_embed = nn.Embedding(num_queries, 4)
        if two_stage:
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        self.num_classes=num_classes
        # output FFNs
        assert self.mask_classification, "why not class embedding?"
        # self.label_enc=nn.Embedding(505, hidden_dim)  # this is a hack for o365+coco (365+133=498)
        self.dim_proj = dim_proj
        self.lang_encoder = lang_encoder
        # if lang_encoder is not None:
        self.lang_mapper = nn.Parameter(torch.empty(dim_proj, hidden_dim))
        trunc_normal_(self.lang_mapper, std=.02)

        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # init decoder
        self.decoder_norm = decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = DeformableTransformerDecoderLayer(hidden_dim, dim_feedforward,
                                                          dropout, activation,
                                                          self.num_feature_levels, nhead, dec_n_points)
        self.decoder = TransformerDecoder(decoder_layer, self.num_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=hidden_dim, query_dim=query_dim,
                                          num_feature_levels=self.num_feature_levels,
                                          dec_layer_share=dec_layer_share,
                                          )

        self.hidden_dim = hidden_dim
        self._bbox_embed = _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        box_embed_layerlist = [_bbox_embed for i in range(self.num_layers)]  # share box prediction each layer
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.decoder.bbox_embed = self.bbox_embed
        
        # whole category classification
        self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        trunc_normal_(self.class_embed, std=.02)
        # part category classification
        self.class_embed_part = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        trunc_normal_(self.class_embed_part, std=.02)

        # FIXME iou head; iou prediction: 1. iou token to predict 3 score. 2. predict each iou score from query tokens
        # FIXME seems we only need to stack these tokens in batch dimension to reduce self attention burden.
        self.num_mask_tokens = num_mask_tokens  # sam uses 4 to handle multi prompts
        self.iou_token = 0   # FIXME hack to remove iou token
        self.num_all_tokens = self.num_mask_tokens + self.iou_token  # sam uses 4 to handle multi prompts
        self.iou_prediction_head = MLP(hidden_dim, hidden_dim, 1, 3)
        # self.iou_token = nn.Embedding(self.iou_token, hidden_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, hidden_dim)
        self.pb_embedding=nn.Embedding(2,hidden_dim)
        self.label_enc=nn.Embedding(2,hidden_dim)
        
        self.prediction_switch = None


    @classmethod
    def from_config(cls, cfg, in_channels, lang_encoder, mask_classification, extra):
        ret = {}
        ret["in_channels"] = in_channels
        ret["lang_encoder"] = lang_encoder
        ret["mask_classification"] = mask_classification

        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']

        ret["num_classes"] = enc_cfg['NUM_CLASSES']
        ret["hidden_dim"] = dec_cfg['HIDDEN_DIM']
        ret["dim_proj"] = cfg['MODEL']['DIM_PROJ']
        ret["num_queries"] = dec_cfg['NUM_OBJECT_QUERIES']

        # Transformer parameters:
        ret["num_mask_tokens"] = dec_cfg.get('NUM_MASK_TOKENS', 3)
        
        ret["nheads"] = dec_cfg['NHEADS']
        ret["dim_feedforward"] = dec_cfg['DIM_FEEDFORWARD']
        ret["dec_layers"] = dec_cfg['DEC_LAYERS']
        ret["enforce_input_project"] = dec_cfg['ENFORCE_INPUT_PROJ']
        ret["mask_dim"] = enc_cfg['MASK_DIM']
        ret["two_stage"] = dec_cfg['TWO_STAGE']
        ret["initialize_box_type"] = dec_cfg['INITIALIZE_BOX_TYPE']  # ['no', 'bitmask', 'mask2box']
        ret["dn"] = dec_cfg['DN']
        ret["noise_scale"] = dec_cfg['DN_NOISE_SCALE']
        ret["dn_num"] = dec_cfg['DN_NUM']
        ret["initial_pred"] = dec_cfg['INITIAL_PRED']
        ret["learn_tgt"] = dec_cfg['LEARN_TGT']
        ret["total_num_feature_levels"] = dec_cfg['TOTAL_NUM_FEATURE_LEVELS']
        ret["num_mask_tokens"] = dec_cfg.get('NUM_INTERACTIVE_TOKENS', 3)
        ret["semantic_ce_loss"] = dec_cfg['TEST']['SEMANTIC_ON'] and dec_cfg['SEMANTIC_CE_LOSS'] and not dec_cfg['TEST']['PANOPTIC_ON']

        return ret

    def prepare_for_dn(self, targets, tgt, refpoint_emb, batch_size):
        """
        modified from dn-detr. You can refer to dn-detr
        https://github.com/IDEA-Research/DN-DETR/blob/main/models/dn_dab_deformable_detr/dn_components.py
        for more details
            :param dn_args: scalar, noise_scale
            :param tgt: original tgt (content) in the matching part
            :param refpoint_emb: positional anchor queries in the matching part
            :param batch_size: bs
            """
        if self.training:
            scalar, noise_scale = self.dn_num, self.noise_scale

            known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
            know_idx = [torch.nonzero(t) for t in known]
            known_num = [sum(k) for k in known]

            # use fix number of dn queries
            if max(known_num) > 0:
                scalar = scalar // (int(max(known_num)))
            else:
                scalar = 0
            if scalar == 0:
                input_query_label = None
                input_query_bbox = None
                attn_mask = None
                mask_dict = None
                return input_query_label, input_query_bbox, attn_mask, mask_dict

            # can be modified to selectively denosie some label or boxes; also known label prediction
            unmask_bbox = unmask_label = torch.cat(known)
            labels = torch.cat([t['labels'] for t in targets])
            # use languge as denosing content queries.
            # if task == 'det':
            #     labels = labels  # o365 start from 133 class
            boxes = torch.cat([t['boxes'] for t in targets])
            batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
            # known
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)

            # noise
            known_indice = known_indice.repeat(scalar, 1).view(-1)
            known_labels = labels.repeat(scalar, 1).view(-1)
            known_bid = batch_idx.repeat(scalar, 1).view(-1)
            known_bboxs = boxes.repeat(scalar, 1)
            known_labels_expaned = known_labels.clone()
            known_bbox_expand = known_bboxs.clone()

            if noise_scale > 0:
                diff = torch.zeros_like(known_bbox_expand)
                diff[:, :2] = known_bbox_expand[:, 2:] / 2
                diff[:, 2:] = known_bbox_expand[:, 2:]
                known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
                                               diff).cuda() * noise_scale
                known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

            m = known_labels_expaned.long().to('cuda')
            # import ipdb; ipdb.set_trace()
            input_label_embed = torch.gather(self.lang_encoder.default_text_embeddings, 0,
                                             m[:, None].repeat(1, self.dim_proj)) @ self.lang_mapper

            input_bbox_embed = inverse_sigmoid(known_bbox_expand)
            single_pad = int(max(known_num))
            pad_size = int(single_pad * scalar)

            padding_label = input_label_embed.new_zeros(pad_size, self.hidden_dim)
            padding_bbox = input_bbox_embed.new_zeros(pad_size, 4)

            if not refpoint_emb is None:
                input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
                input_query_bbox = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
            else:
                input_query_label = padding_label.repeat(batch_size, 1, 1)
                input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

            # map
            map_known_indice = input_label_embed.new_tensor([])
            if len(known_num):
                map_known_indice = torch.cat(
                    [input_label_embed.new_tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
            if len(known_bid):
                input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
                input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

            tgt_size = pad_size + self.num_queries
            attn_mask = input_label_embed.new_ones(tgt_size, tgt_size) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size,
                'scalar': scalar,
            }
        else:
            if not refpoint_emb is None:
                input_query_label = tgt.repeat(batch_size, 1, 1)
                input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
            else:
                input_query_label = None
                input_query_bbox = None
            attn_mask = None
            mask_dict = None

        # 100*batch*256
        if not input_query_bbox is None:
            input_query_label = input_query_label
            input_query_bbox = input_query_bbox

        return input_query_label, input_query_bbox, attn_mask, mask_dict

    def prepare_for_dn_mo(self, targets, tgt, refpoint_emb, batch_size):
        """
        Train SA-1B data with point input.
        This training can be regarded as a multi-granularity denoising process
        modified from dn-detr. You can refer to dn-detr
        https://github.com/IDEA-Research/DN-DETR/blob/main/models/dn_dab_deformable_detr/dn_components.py
        for more details
            :param dn_args: scalar, noise_scale
            :param tgt: original tgt (content) in the matching part
            :param refpoint_emb: positional anchor queries in the matching part
            :param batch_size: bs
            """
        scalar, noise_scale = self.dn_num, self.noise_scale


        pb_labels = torch.stack([t['pb'] for t in targets])
        # FIXME this is for future content-based interaction; pool content features as label embedding
        labels = torch.zeros_like(pb_labels).long()
        boxes = torch.stack([t['boxes_dn'] for t in targets])
        box_start = [t['box_start'] for t in targets]


        known_labels = labels
        known_pb_labels = pb_labels

        known_bboxs = boxes
        known_labels_expaned = known_labels.clone()
        known_pb_labels_expaned = known_pb_labels.clone()
        known_bbox_expand = known_bboxs.clone()
        if noise_scale > 0 and self.training:
            diff = torch.zeros_like(known_bbox_expand)
            diff[:, :, :2] = known_bbox_expand[:, :, 2:] / 2
            diff[:, :, 2:] = known_bbox_expand[:, :, 2:]
            # add very small noise to input points
            sc = 0.01
            for i, st in enumerate(box_start):
                diff[i, :st] = diff[i, :st] * sc
            known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
                                           diff).cuda() * noise_scale
            known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

        m = known_labels_expaned.long().to('cuda')
        m_pb = known_pb_labels_expaned.long().to('cuda')
        input_label_embed = self.label_enc(m)+self.pb_embedding(m_pb)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        input_label_embed = input_label_embed.repeat_interleave(self.num_all_tokens,1) + self.mask_tokens.weight.unsqueeze(0).repeat(input_label_embed.shape[0], input_label_embed.shape[1], 1)
        input_bbox_embed = input_bbox_embed.repeat_interleave(self.num_all_tokens,1)

        single_pad = self.num_all_tokens

        # NOTE scalar is modified to 100, each click cannot see each other
        scalar = int(input_label_embed.shape[1]/self.num_all_tokens)

        pad_size = input_label_embed.shape[1]

        if input_label_embed.shape[1]>0:
            input_query_label = input_label_embed
            input_query_bbox = input_bbox_embed

        tgt_size = pad_size
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        mask_dict = {
            'known_lbs_bboxes': (known_labels, known_bboxs),
            # 'know_idx': know_idx,
            'pad_size': pad_size,
            'scalar': scalar,
        }


        return input_query_label,input_query_bbox,attn_mask,mask_dict

    def prepare_for_dn_mo_infer(self, targets, tgt, refpoint_emb, batch_size):

        known = [(torch.ones_like(t['points'])).cuda() for t in targets]
        known_num = [k.sum() for k in known]

        assert max(known_num)>0

        pb_labels = torch.stack([t['pb'] for t in targets])
        # FIXME this is for future content-based interaction; pool content features as label embedding
        labels = torch.zeros_like(pb_labels).long()
        boxes = torch.stack([t['points'] for t in targets])


        known_labels = labels
        known_pb_labels = pb_labels

        known_bboxs = boxes
        known_labels_expaned = known_labels.clone()
        known_pb_labels_expaned = known_pb_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        m = known_labels_expaned.long().to('cuda')
        m_pb = known_pb_labels_expaned.long().to('cuda')
        input_label_embed = self.label_enc(m)+self.pb_embedding(m_pb)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        input_label_embed = input_label_embed.repeat_interleave(self.num_all_tokens,1) + self.mask_tokens.weight.unsqueeze(0).repeat(input_label_embed.shape[0], input_label_embed.shape[1], 1)
        input_bbox_embed = input_bbox_embed.repeat_interleave(self.num_all_tokens,1)


        scalar = int(input_label_embed.shape[1]/self.num_all_tokens)

        pad_size = input_label_embed.shape[1]

        if input_label_embed.shape[1]>0:
            input_query_label = input_label_embed
            input_query_bbox = input_bbox_embed

        attn_mask = None
        mask_dict = {
            'known_lbs_bboxes': (known_labels, known_bboxs),
            # 'know_idx': know_idx,
            'pad_size': pad_size,
            'scalar': scalar,
        }


        return input_query_label,input_query_bbox,attn_mask,mask_dict

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def pred_box(self, reference, hs, ref0=None):
        """
        :param reference: reference box coordinates from each decoder layer
        :param hs: content
        :param ref0: whether there are prediction from the first layer
        """
        if ref0 is None:
            outputs_coord_list = []
        else:
            outputs_coord_list = [ref0]
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            # layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            new_layer_ref_sig = layer_ref_sig.view(layer_ref_sig.shape[0], -1, self.num_all_tokens, layer_ref_sig.shape[-1])
            new_layer_ref_sig = new_layer_ref_sig[:, :, :self.num_mask_tokens].reshape(new_layer_ref_sig.shape[0], -1, new_layer_ref_sig.shape[-1])
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(new_layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)
        return outputs_coord_list

    def forward(self, x, mask_features, masks, targets=None, target_queries=None, target_vlp=None, task='seg', extra={}):
        """
        task: seg/det TODO add sam
        """
        prediction_switch = extra
        self.prediction_switch = prediction_switch
        assert len(x) == self.num_feature_levels
        do_seg = True  # if task is det, not do segmentation training (for O365)
        size_list = []
        # disable mask, it does not affect performance
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in x]
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for i in range(self.num_feature_levels):
            idx=self.num_feature_levels-1-i
            bs, c , h, w=x[idx].shape
            size_list.append(x[i].shape[-2:])
            spatial_shapes.append(x[idx].shape[-2:])
            src_flatten.append(self.input_proj[idx](x[idx]).flatten(2).transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        predictions_class = []
        predictions_class_part = []
        predictions_mask = []
        predictions_iou_score = []

        tgt_mask = None
        mask_dict = None
        if self.dn != "no":
            assert targets is not None
            if task=='demo':
                input_query_label, input_query_bbox, tgt_mask, mask_dict = \
                    self.prepare_for_dn_mo_infer(targets, None, None, x[0].shape[0])
            else:
                input_query_label, input_query_bbox, tgt_mask, mask_dict = \
                    self.prepare_for_dn_mo(targets, None, None, x[0].shape[0])
            tgt=input_query_label
            refpoint_embed=input_query_bbox
            if tgt is None:
                tgt = torch.zeros(bs, self.num_queries, self.hidden_dim).cuda()
                refpoint_embed = torch.zeros(bs, self.num_queries, 4).cuda()

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=None,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask
        )

        new_hs = []
        for i, output in enumerate(hs):
            outputs_class, outputs_mask, iou_score, decoder_output_mask = self.interactive_forward_prediction_heads(output.transpose(0, 1), mask_features, (self.training or (i == len(hs)-1)) and do_seg)
            outputs_class_whole, outputs_class_part = outputs_class
            predictions_class.append(outputs_class_whole)
            predictions_class_part.append(outputs_class_part)
            predictions_mask.append(outputs_mask)
            if iou_score is not None:
                predictions_iou_score.append(iou_score)
                new_hs.append(decoder_output_mask)
        if new_hs is not None:
            hs = new_hs
        # iteratively box prediction
        out_boxes = self.pred_box(references, hs)
        out_boxes[-1] = out_boxes[-1] + 0.0 * (self.label_enc.weight.sum() + self.pb_embedding.weight.sum() 
                                                               + self.mask_tokens.weight.sum() + self.lang_mapper.sum())
        if mask_dict is not None:
            if predictions_mask is None:
                predictions_class[-1] = predictions_class[-1]
                for i in range(self.mask_embed.num_layers):
                    predictions_class[-1] = predictions_class[-1] + 0.0 * (self.mask_embed.layers[i].weight[0][0] + self.mask_embed.layers[i].bias[0])  # avoid no mask loss
                predictions_class[-1] = predictions_class[-1] + 0.0 * mask_features[0][0][0][0]  # avoid no mask loss

            if do_seg:
                predictions_mask = list(predictions_mask)
        elif self.training:  # this is to insure self.label_enc participate in the model
            for i in range(self.mask_embed.num_layers):
                predictions_class[-1] = predictions_class[-1] + 0.0 * (
                            self.mask_embed.layers[i].weight[0][0] + self.mask_embed.layers[i].bias[
                        0])  # avoid no mask loss
            predictions_class[-1] = predictions_class[-1] + 0.0 * mask_features[0][0][0][0]  # avoid no mask loss

        out = {
            'pred_logits': predictions_class[-1],
            'pred_logits_part': predictions_class_part[-1],
            'pred_masks': None if not do_seg else predictions_mask[-1],
            'pred_boxes':out_boxes[-1],
            'pred_ious': predictions_iou_score[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask, out_boxes, predictions_iou_score, predictions_class_part
            )
        }

        return out, mask_dict

    def interactive_forward_prediction_heads(self, output, mask_features, pred_mask=True):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        
        decoder_output = decoder_output + 0.0 * (self.class_embed_part.sum() + self.class_embed.sum())

        out = decoder_output.view(decoder_output.shape[0], -1, self.num_all_tokens, decoder_output.shape[-1])
        decoder_output_mask = out[:, :, :self.num_mask_tokens].reshape(decoder_output.shape[0], -1, decoder_output.shape[-1])
        # decoder_output_iou = out[:, :, -1].view(decoder_output.shape[0], -1, decoder_output.shape[-1])
        decoder_output_iou = decoder_output_mask

        outputs_mask = outputs_class_whole = outputs_class_part = None
        if self.prediction_switch['whole']:
            class_embed_whole = decoder_output @ self.class_embed
            outputs_class_whole = self.lang_encoder.compute_similarity(class_embed_whole, name='whole')
        if self.prediction_switch['part']:
            class_embed_part = decoder_output @ self.class_embed_part
            outputs_class_part = self.lang_encoder.compute_similarity(class_embed_part, name='part')
        
        outputs_class = (outputs_class_whole, outputs_class_part)
        if self.prediction_switch['seg']:
            mask_embed = self.mask_embed(decoder_output_mask)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        iou_score = self.iou_prediction_head(decoder_output_iou).squeeze(-1).view(decoder_output.shape[0], -1, self.num_mask_tokens)
        # outputs_mask = outputs_mask + 0.0 * iou_score.sum()  # TODO add iou prediction head

        return outputs_class, outputs_mask, iou_score, decoder_output_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class=None, outputs_seg_masks=None, out_boxes=None, predictions_iou_score=None, predictions_class_part=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # if self.mask_classification:
        if out_boxes is None:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        elif outputs_seg_masks is None:
            return [
                {"pred_logits": a, "pred_boxes": c}
                for a, c in zip(outputs_class[:-1], out_boxes[:-1])
            ]
        elif predictions_iou_score is None:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes":c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1])
            ]
        else:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes":c, "pred_ious":d, "pred_logits_part": e}
                for a, b, c, d, e in zip(outputs_class[:-1], outputs_seg_masks[:-1],out_boxes[:-1], predictions_iou_score[:-1], predictions_class_part[:-1])
            ]

@register_decoder
def get_interactive_maskdino_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
    return IMaskDINODecoder(cfg, in_channels, lang_encoder, mask_classification, extra)
