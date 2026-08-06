from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch, torch.nn as nn

@dataclass(frozen=True)
class B00Output:
    """B00 forward output. No placeholder regional/prompt fields."""
    spoof_logit:torch.Tensor              # [B], finite
    feature_embedding:torch.Tensor|None=None   # debug only
class B00ConvNeXtBinaryClassifier(nn.Module):
    """ConvNeXt V2 backbone + single-logit spoof head (live=0, spoof=1).

    The model applies the pretrained weights' normalization internally, so the
    M4 canonical [0,1] RGB tensor is normalized exactly once.
    """
    def __init__(self,model_name:str,*,pretrained:bool=True,dropout:float=0.,
                 mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225),weight_file:str|None=None):
        super().__init__()
        import timm
        self.model_name=model_name
        # weight_file loads the pinned local checkpoint instead of reaching the
        # Hub, which is required in offline containers.
        overlay=dict(file=str(weight_file)) if weight_file else None
        self.backbone=timm.create_model(model_name,pretrained=pretrained,num_classes=0,
                                        **({"pretrained_cfg_overlay":overlay} if overlay else {}))
        self.feature_dim=int(self.backbone.num_features)
        self.dropout=nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.head=nn.Linear(self.feature_dim,1)
        nn.init.zeros_(self.head.bias); nn.init.trunc_normal_(self.head.weight,std=0.01)
        self.register_buffer("norm_mean",torch.tensor(mean,dtype=torch.float32).view(1,3,1,1),persistent=False)
        self.register_buffer("norm_std",torch.tensor(std,dtype=torch.float32).view(1,3,1,1),persistent=False)
    def normalize(self,image:torch.Tensor)->torch.Tensor:
        if image.ndim!=4 or image.shape[1]!=3: raise ValueError(f"expected [B,3,H,W], got {tuple(image.shape)}")
        return (image-self.norm_mean)/self.norm_std
    def forward(self,image:torch.Tensor,*,return_features:bool=False)->B00Output:
        features=self.backbone(self.normalize(image))
        logit=self.head(self.dropout(features)).squeeze(1)
        return B00Output(spoof_logit=logit,feature_embedding=features if return_features else None)
    def parameter_groups(self,backbone_lr:float,head_lr:float,weight_decay:float)->list[dict]:
        return [{"params":list(self.backbone.parameters()),"lr":backbone_lr,"weight_decay":weight_decay,"name":"backbone"},
                {"params":list(self.head.parameters()),"lr":head_lr,"weight_decay":weight_decay,"name":"head"}]
def build_b00_model(spec,weight_file:str|None=None)->B00ConvNeXtBinaryClassifier:
    return B00ConvNeXtBinaryClassifier(spec.model_name,pretrained=spec.pretrained,dropout=spec.dropout,
                                       mean=spec.normalization_mean,std=spec.normalization_std,
                                       weight_file=weight_file)
