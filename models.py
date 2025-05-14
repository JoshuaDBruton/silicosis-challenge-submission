import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model_and_transforms
from modules.q_former import QFormer
from timm.models.swin_transformer_v2 import SwinTransformerV2
from torchvision import transforms
from modules.ptuned_text_encoder import PTunedTextEncoder


class BiomedCLIPWithSwin(nn.Module):
    def __init__(self, biomedclip_model, tokenizer,  pretrained_path="checkpoints/open_clip_pytorch_model.bin", image_size=2048, prompt_length=10, use_qformer=False, use_swin=False):
        super().__init__()
        # Store the original BiomedCLIP model
        self.biomedclip = biomedclip_model

        self.ptuned_text_encoder = PTunedTextEncoder(
            base_text_encoder=self.biomedclip.text,
            tokenizer=tokenizer,
            prompt_length=prompt_length
        )

        # Working for image size 2048
        # self.swin = SwinTransformer(
        #     img_size=image_size,
        #     patch_size=8,
        #     in_chans=1,
        #     embed_dim=96,
        #     depths=[2, 4],
        #     num_heads=[6, 12],
        #     window_size=16,
        #     mlp_ratio=4.0,
        #     qkv_bias=True,
        #     drop_rate=0,
        #     attn_drop_rate=0,
        #     drop_path_rate=0.1,
        #     norm_layer=nn.LayerNorm,
        #     patch_norm=True,
        #     use_checkpoint=True,  # Enable checkpointing for memory efficiency
        #     num_classes=0  # No classification head, just feature extraction
        # )

        # self.adapter = nn.Conv2d(192, 3, kernel_size=1)

        # self.swin = SwinTransformer(
        #     img_size=image_size,
        #     patch_size=4,
        #     in_chans=1,
        #     embed_dim=48,
        #     depths=[2, 2],
        #     num_heads=[2, 4],
        #     window_size=4,
        #     mlp_ratio=4.0,
        #     qkv_bias=True,
        #     drop_rate=0,
        #     attn_drop_rate=0,
        #     drop_path_rate=0.1,
        #     norm_layer=nn.LayerNorm,
        #     patch_norm=True,
        #     use_checkpoint=True,  # Enable checkpointing for memory efficiency
        #     num_classes=0  # No classification head, just feature extraction
        # )

        if use_swin:
            self.swin = SwinTransformerV2(
                img_size=image_size,
                patch_size=4,
                in_chans=1,
                embed_dim=48,
                depths=[2, 2],
                num_heads=[2, 4],
                window_size=4,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=0,
                attn_drop_rate=0,
                drop_path_rate=0.1,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
                use_checkpoint=True,
                num_classes=0
            )
        else:
            self.swin = None

        self.adapter = nn.Conv2d(96, 3, kernel_size=1)

        if pretrained_path:
            state_dict = torch.load(pretrained_path)
            biomedclip_state_dict = {k: v for k, v in state_dict.items() if k.startswith('biomedclip.')}
            self.load_state_dict(biomedclip_state_dict, strict=False)

        self.use_qformer = use_qformer
        if use_qformer:
            qformer_kwargs = {
                "num_query_tokens": 32,
                "query_dim": 256,
                "encoder_dim": 512,
                "num_layers": 6,
                "num_heads": 8
            }
            self.qformer = QFormer(**qformer_kwargs)

            self.qformer_head = nn.Linear(
                qformer_kwargs.get("query_dim", 256),
                512
            )

    @property
    def logit_scale(self):
        return self.biomedclip.logit_scale

    def encode_image(self, x):
        if self.swin is not None:
            features = self.swin.forward_features(x)

            features = features.permute(0, 3, 1, 2)

            x = self.adapter(features)

        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.biomedclip.encode_image(x)

        if self.use_qformer:
            x = x.unsqueeze(1)
            x = self.qformer(x)
            x = x.mean(dim=1)
            x = self.qformer_head(x)

        return x

    def swin_pass(self, x):
        features = self.swin.forward_features(x)

        features = features.permute(0, 3, 1, 2)

        x = self.adapter(features)

        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        single_image_features = x[0]
        o_image_tensor = torch.mean(single_image_features, dim=0)
        o_image_tensor = torch.abs(o_image_tensor)

        return o_image_tensor.cpu().detach().numpy()

    def encode_text(self, text):
        # return self.biomedclip.encode_text(text)
        return self.ptuned_text_encoder(text)

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.biomedclip.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


def create_custom_preprocess(image_size):
    return transforms.Compose([
        # transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
        # transforms.Grayscale(), # this transform from torchvisions makes things look quite bad
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def create_combined_model(tokenizer, model_name="biomedclip_local", pretrained_path="checkpoints/open_clip_pytorch_model.bin", image_size=2049, use_qformer=False, use_swin=False):
    model, _, _ = create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained_path
    )

    combined_model = BiomedCLIPWithSwin(model, tokenizer, pretrained_path, image_size=image_size, use_qformer=use_qformer, use_swin=use_swin)

    preprocess = create_custom_preprocess(image_size=image_size)

    return combined_model, preprocess

