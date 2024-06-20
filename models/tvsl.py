import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
import torch.distributed as dist
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import json
from .audioclip import AudioCLIP
from models.audioclip.clip.clip import tokenize
from .conditioning import CondBlock

class TVSL(AudioCLIP):
    def __init__(self, *args, **kwargs):
        super(TVSL, self).__init__(*args, **kwargs)

        for name, params in self.named_parameters():
            params.requires_grad = False

        self.logit_scale_at.requires_grad_(False)
        self.logit_scale.requires_grad_(False)

        self.logit_scale_ml = torch.nn.Parameter(torch.log(torch.ones([]) * 100))

        # extract all class text tokens
        all_class_names = []
        with open('metadata/vggss.json') as json_file:
            annotations = json.load(json_file)
            for annotation in annotations:
                class_name = annotation["class"]
                if class_name not in all_class_names:
                    all_class_names.append(class_name)
        all_class_names = sorted(set(all_class_names))
        with torch.no_grad():
            self.all_text_class = self.encode_text(all_class_names)
            self.all_text_class = self.all_text_class / self.all_text_class.norm(
                dim=-1, keepdim=True
            )

        self.audio_proj = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024)
        )
        self.ap_w = torch.nn.Parameter(torch.ones([]))

        self.aud_cls_proj =  nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024)
        )
        self.acp_w = torch.nn.Parameter(torch.ones([]))

        self.visual_proj = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024)
        )
        self.vp_w = torch.nn.Parameter(torch.ones([]))

        self.img_cls_proj = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024)
        )
        self.icp_w = torch.nn.Parameter(torch.ones([]))

        self.fusion_proj = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048)
        )

        self.aud_img_proj = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024)
        )
        self.aip_w = torch.nn.Parameter(torch.ones([]))

        self.img_aud_proj = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024)
        )
        self.iap_w = torch.nn.Parameter(torch.ones([]))

        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_bce = nn.BCEWithLogitsLoss()


    def encode_text(self, text, base_str: str = '{}', return_tokens=False):
        batch_indices = torch.arange(len(text), dtype=torch.int64, device=self.device)

        if batch_indices is not None:
            text = [text[idx] for idx in batch_indices]

        text_tokens = torch.cat([
            tokenize(base_str.format(entities)) for entities in text
        ])

        text_tokens = text_tokens.to(self.device)

        x = self.token_embedding(text_tokens).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if return_tokens:
            tokens = x @ self.text_projection
            x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.text_projection
            return x, tokens

        x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.text_projection

        return x

    def audio_class_loss(self, feature1, label1, feature2, label2):
        ## check audio clip classification parts
        logits_audio_text = torch.clamp(self.logit_scale_at.exp(), min=1.0, max=100.0)
        pred1 = logits_audio_text * (feature1 @ self.all_text_class.T)
        pred2 = logits_audio_text * (feature2 @ self.all_text_class.T)        
        
        loss = (self.loss_ce(pred1, label1) + self.loss_ce(pred2, label2))/2

        return loss

    def image_class_loss(self, feature1, label1, feature2, label2):
        logits_image_text = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)
        pred1 = logits_image_text * (feature1 @ self.all_text_class.T)
        pred2 = logits_image_text * (feature2 @ self.all_text_class.T)        
        
        loss = (self.loss_ce(pred1, label1) + self.loss_ce(pred2, label2))/2

        return loss

    def multi_label_loss(self, feature, label):
        logits_multilabel = torch.clamp(self.logit_scale_ml.exp(), min=1.0, max=100.0)

        all_text_feat = torch.concat([
            self.all_text_class, self.all_text_class
        ], dim=-1)

        all_text_feat = all_text_feat / all_text_feat.norm(
            dim=-1, keepdim=True
        )

        pred = logits_multilabel * (feature @ all_text_feat.T)
          
        loss = self.loss_bce(pred, label)

        return loss

    def contrastive_loss(self, image_features, audio_features):

        batch_size = image_features.shape[0]

        reference = torch.arange(
            batch_size,
            dtype=torch.int64,
            device=self.device
        )

        logit_scale_ai = torch.clamp(self.logit_scale_ai.exp(), min=1.0, max=100.0)
        logits_audio_image = logit_scale_ai * audio_features @ image_features.T

        loss = F.cross_entropy(
            logits_audio_image, reference
        ) + F.cross_entropy(
            logits_audio_image.transpose(-1, -2), reference
        )

        return loss / 2

    def image_token_feats(self, image):
        all_tokens = self.encode_image(image, return_tokens=True)
        all_tokens = all_tokens / all_tokens.norm(dim=-1, keepdim=True)
        
        return all_tokens

    def forward(self, image, audio, anno):
        batch, x, y = image.shape[0], image.shape[2]//224, image.shape[3]//224

        class1, label1, class2, label2 = anno['class1'], anno['label1'], anno['class2'], anno['label2']
        label = anno['class']

        self.all_text_class = self.all_text_class.to(device=self.device)
        label1 = torch.argmax(label1, dim=-1)
        label2 = torch.argmax(label2, dim=-1)

        # extract fine-grained text class features
        with torch.no_grad():
            class1_text_features = self.encode_text(class1, return_tokens=False)
            class2_text_features = self.encode_text(class2, return_tokens=False)
        
        class1_text_features = class1_text_features.contiguous().view(batch, 1024)
        class1_text_features = class1_text_features / torch.linalg.norm(
            class1_text_features, dim=-1, keepdim=True
        )
        class1_text_features = class1_text_features.unsqueeze(-1)

        class2_text_features = class2_text_features.contiguous().view(batch, 1024)
        class2_text_features = class2_text_features / torch.linalg.norm(
            class2_text_features, dim=-1, keepdim=True
        )
        class2_text_features = class2_text_features.unsqueeze(-1)

        # extract image features
        batch = image.shape[0]
        image_feature = self.image_token_feats(image).transpose(0, 1)
        image_feature = image_feature.contiguous().view(batch, -1, 1024)
        image_feature = self.vp_w * image_feature + (1-self.vp_w) * self.visual_proj(image_feature)
        mean_img_feature = image_feature.mean(dim=1)
        image_feature = image_feature / torch.linalg.norm(
            image_feature, dim=-1, keepdim=True
        )

        # extract audio features
        _, audio_features = self.audio(audio, return_feats=True)
        audio_features = audio_features.contiguous().view(batch, -1, 1024)
        audio_features = self.ap_w * audio_features + (1 - self.ap_w) * self.audio_proj(audio_features)
        mean_aud_feature = audio_features.mean(dim=1)
        audio_features = audio_features / audio_features.norm(
            dim=-1, keepdim=True
        )

        # apply multi-label class loss
        aud_img_feature = torch.concat([
            mean_img_feature, mean_aud_feature
        ], dim=-1)

        aud_img_feature = self.fusion_proj(aud_img_feature)
        aud_img_feature = aud_img_feature / aud_img_feature.norm(
            dim=-1, keepdim=True
        )

        loss_mcl = self.multi_label_loss(aud_img_feature, label)

        # Extract text-conditional image features
        ## class 1
        class1_mask = (image_feature @ class1_text_features).squeeze()
        class1_mask = F.normalize(class1_mask, dim=-1)
        class1_img_feats = class1_mask.unsqueeze(-1) * image_feature
        mean_class1_img_feats =  class1_img_feats.mean(dim=1)
        mean_class1_img_feats = self.icp_w * mean_class1_img_feats + (1-self.icp_w)* self.img_cls_proj(mean_class1_img_feats)
        mean_class1_img_feats = mean_class1_img_feats / mean_class1_img_feats.norm(dim=-1, keepdim=True)

        ## class 2
        class2_mask = (image_feature @ class2_text_features).squeeze()
        class2_mask = F.normalize(class2_mask, dim=-1)
        class2_img_feats = class2_mask.unsqueeze(-1) * image_feature
        mean_class2_img_feats =  class2_img_feats.mean(dim=1)
        mean_class2_img_feats = self.icp_w * mean_class2_img_feats + (1-self.icp_w)* self.img_cls_proj(mean_class2_img_feats)

        mean_class2_img_feats = mean_class2_img_feats / mean_class2_img_feats.norm(dim=-1, keepdim=True)

        # apply class img loss
        loss_img_cls = self.image_class_loss(mean_class1_img_feats, label1, mean_class2_img_feats, label2)


        # Extract text-conditional audio features
        ## class 1
        class1_mask = (audio_features @ class1_text_features).squeeze()
        class1_mask = F.normalize(class1_mask, dim=-1)
        class1_aud_feats = class1_mask.unsqueeze(-1) * audio_features
        mean_class1_aud_feats =  class1_aud_feats.mean(dim=1)
        mean_class1_aud_feats = self.acp_w * mean_class1_aud_feats + (1 - self.acp_w) * self.aud_cls_proj(mean_class1_aud_feats)
        mean_class1_aud_feats = mean_class1_aud_feats / mean_class1_aud_feats.norm(dim=-1, keepdim=True)

        ## class 2
        class2_mask = (audio_features @ class2_text_features).squeeze()
        class2_mask = F.normalize(class2_mask, dim=-1)
        class2_aud_feats = class2_mask.unsqueeze(-1) * audio_features
        mean_class2_aud_feats =  class2_aud_feats.mean(dim=1)
        mean_class2_aud_feats = self.acp_w * mean_class2_aud_feats + (1- self.acp_w) * self.aud_cls_proj(mean_class2_aud_feats)
        mean_class2_aud_feats = mean_class2_aud_feats / mean_class2_aud_feats.norm(dim=-1, keepdim=True)

        # apply class aud loss
        loss_aud_cls = self.audio_class_loss(mean_class1_aud_feats, label1, mean_class2_aud_feats, label2)

        # extract audio-conditional image features
        class1_aud_query = class1_aud_feats.mean(dim=1)
        class2_aud_query = class2_aud_feats.mean(dim=1)

        class1_aud_query = self.aip_w * class1_aud_query + (1-self.aip_w) * self.aud_img_proj(class1_aud_query)
        class2_aud_query = self.aip_w * class2_aud_query + (1 - self.aip_w) * self.aud_img_proj(class2_aud_query)

        class1_aud_query = class1_aud_query / class1_aud_query.norm(
            dim=-1, keepdim=True
        )
        class2_aud_query = class2_aud_query / class2_aud_query.norm(
            dim=-1, keepdim=True
        )

        if not self.training:
            class1_img_feats = self.iap_w * class1_img_feats + (1 - self.iap_w) * self.img_aud_proj(class1_img_feats)
            class2_img_feats = self.iap_w * class2_img_feats + (1 - self.iap_w) * self.img_aud_proj(class2_img_feats)

            class1_img_feats = class1_img_feats / torch.linalg.norm(
                class1_img_feats, dim=-1, keepdim=True
            )
            class2_img_feats = class2_img_feats / torch.linalg.norm(
                class2_img_feats, dim=-1, keepdim=True
            )

            class1_image_output = class1_img_feats @ class1_aud_query.unsqueeze(-1)
            class1_image_output = class1_image_output.contiguous().view(batch, -1)
            class1_image_output = class1_image_output / class1_image_output.norm(dim=-1, keepdim=True)
            class1_image_output = class1_image_output.contiguous().view(batch, 1, 7*x, 7*y)

            class2_image_output = class2_img_feats @ class2_aud_query.unsqueeze(-1)
            class2_image_output = class2_image_output.contiguous().view(batch, -1)
            class2_image_output = class2_image_output / class2_image_output.norm(dim=-1, keepdim=True)
            class2_image_output = class2_image_output.contiguous().view(batch, 1, 7*x, 7*y)

            return class1_image_output, class2_image_output

        else:            
            class1_img_feats = class1_img_feats.mean(dim=1)
            class2_img_feats = class2_img_feats.mean(dim=1)

            class1_img_feats = self.iap_w * class1_img_feats + (1-self.iap_w) * self.img_aud_proj(class1_img_feats)
            class2_img_feats = self.iap_w * class2_img_feats + (1 - self.iap_w) * self.img_aud_proj(class2_img_feats)

            class1_img_feats = class1_img_feats / class1_img_feats.norm(
                dim=-1, keepdim=True
            )
            class2_img_feats = class2_img_feats / class2_img_feats.norm(
                dim=-1, keepdim=True
            )

            aud_feats = torch.concat([class1_aud_query, class2_aud_query], dim=0)
            img_feats = torch.concat([class1_img_feats, class2_img_feats], dim=0)
            cnt_loss = self.contrastive_loss(img_feats, aud_feats)

            total_loss = loss_img_cls + loss_aud_cls + cnt_loss + loss_mcl

            loss_dict = {
                "mcl_loss": loss_mcl,
                "audio_cls_loss": loss_aud_cls,
                "image_cls_loss": loss_img_cls,
                "contrastive_loss": cnt_loss
            }

            return total_loss, loss_dict