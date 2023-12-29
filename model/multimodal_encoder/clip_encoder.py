import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.feature_select_strategy = getattr(args, 'feature_select_strategy', '')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

        if self.feature_select_strategy == 'mfm':
            self.layer_norm = nn.LayerNorm(self.hidden_size)
            self.mlm_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def multi_features_merging(self, image_forward_outs):
        """
        MFM：
        采用LLN-Layerscale: z = w1* LLN(z1) + ... + wn * LLN(zn)
        """
        # 第一层embeding层不要，其他transformer层stack到一起过layner norm
        multi_layer_features = torch.stack(list(image_forward_outs.hidden_states)[1:], dim=0)  # 24, bs, 577, 1024
        # multi_layer_features = multi_layer_features.squeeze(1)  # 24, 577, 1024
        if self.select_feature == 'patch':
            multi_layer_features = multi_layer_features[:, :, 1:]  # 24, bs, 576, 1024
        elif self.select_feature == 'cls_patch':
            multi_layer_features = multi_layer_features  # 24, bs, 577, 1024
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        multi_layer_features = self.layer_norm(multi_layer_features)
        out = self.mlm_proj(multi_layer_features)  # 24, bs, 576, 1024
        out = torch.sum(out, dim=0)  # bs, 576, 1024
        return out

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                if self.feature_select_strategy == 'mfm':
                    # print('use mfm@@@@@@@@')
                    image_feature = self.multi_features_merging(image_forward_out).to(image.dtype)
                else:
                    # print('use feature_select   @@@@@@@@')
                    image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)

            if self.feature_select_strategy == 'mfm':
                # print('use mfm!!!!!!!')
                image_features = self.multi_features_merging(image_forward_outs).to(images.dtype)
            else:
                # print('use feature_select   !!!!!!!')
                image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
