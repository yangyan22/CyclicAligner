import copy
import torch
import torch.nn as nn
from modules.encoder_decoder import EncoderDecoder  # encoder_decoder: Shared decoder
import torchvision.models as models
import math
import torch.nn.functional as F
from info_nce import InfoNCE
import random
import torchvision
from einops import repeat, rearrange

import random

from timm.models.swin_transformer import SwinTransformer
import timm.models.swin_transformer as swin

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class Memory(nn.Module):  # input, memory, memory
    def __init__(self, h, d_model, dropout=0.1):
        super(Memory, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, layer_past=None):
        nbatches = query.size(0)
        query, key, value =  [l(x) for l, x in zip(self.linears, (query, key, value))] 
        query, key, value =  [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for x in [query, key, value]]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linear(x)

class Model(nn.Module):
    def __init__(self, args, tokenizer):
        super(Model, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        self.mse_loss = torch.nn.MSELoss(reduction="mean")
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)

        # text encoder
        self.word_embd = nn.Embedding(self.encoder_decoder.vocab_size + 1, args.d_model)
        self.word_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=args.d_model, nhead=8), num_layers=3)
        self.word_mlp = nn.Sequential(nn.Linear(args.d_model, args.d_model), nn.Tanh(), nn.Linear(args.d_model, 9))
        self.att_embed_report = nn.Sequential(nn.Linear(args.d_model, args.d_model), nn.ReLU(), nn.Linear(args.d_model, args.d_model), nn.Dropout(args.drop_prob_lm))

        pe = torch.zeros(120, args.d_model)
        position = torch.arange(0, 120).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, args.d_model, 2).float() * -(math.log(10000.0) / args.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        # image encoder
        if args.visual_extractor == "densenet121":
            model = getattr(models, args.visual_extractor)(weights=args.pretrained)
            self.vision = model.features
            self.att_feat_size = 1024

        elif args.visual_extractor == "resnet18":
            model = getattr(models, args.visual_extractor)(weights=args.pretrained)
            modules = nn.ModuleList(model.children())[:-2]
            self.vision = nn.Sequential(*modules)
            self.att_feat_size = 512

        elif args.visual_extractor == "resnet50":
            model = torchvision.models.resnet50(weights=False)
            model.fc = nn.Linear(model.fc.in_features, 512, bias=False) # projection head
            state_dict = torch.load("pytorch_model.bin")
            diction = {}
            for key in state_dict:
                if key.split(".")[0]== "vision_model":
                    diction_key = key.replace("vision_model.model.","")
                    diction[diction_key] = state_dict[key]
            model.load_state_dict(diction, strict=False)
            modules = nn.ModuleList(model.children())[:-2]
            self.vision = nn.Sequential(*modules) 
            self.att_feat_size = 2048
       
        d_middle = 512
        self.cnn = nn.Conv2d(self.att_feat_size, d_middle, 5, stride=1)
        self.att_embed_image = nn.Sequential(nn.Linear(d_middle, args.d_model), nn.ReLU(),nn.Linear(args.d_model, args.d_model), nn.Dropout(args.drop_prob_lm))

        # Ark+ (only load if we don't have precomputed features)
        self.use_precomputed_features = getattr(args, 'use_precomputed_features', False)
        if not self.use_precomputed_features:
            model = OmniSwinTransformer([14,14,14,3,6,1], projector_features=1376, use_mlp=False, img_size = 768, patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
            checkpoint = torch.load('./Ark6_swinLarge768_ep50.pth.tar')
            state_dict = checkpoint['teacher']
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            k_del = [k for k in state_dict.keys() if "attn_mask" in k] + ['head.weight', 'head.bias']
            print(f"Removing key(s) {k_del} from pretrained checkpoint for scaled input size")
            for k in k_del:
                if k in state_dict:
                    del state_dict[k]
            
            load_result = model.load_state_dict(state_dict, strict=False)
            print(load_result)
            self.msg = model
        else:
            self.msg = None
            print("Using precomputed Ark+ features - Ark+ model not loaded")
        
        # Memory
        self.Memory = Memory(args.num_heads, args.d_model)
        self.dictionary = nn.Linear(86, 512)
        self.w = nn.Linear(args.d_model, args.d_model)

        
    def forward(self, images, imageData, targets, tok, mode='train', tags=1, epoch_id=0, 
                ark_embeddings=None, ark_predictions=None, stage=None):
    
        if mode == 'train':
            patch_feats = self.cnn(self.vision(images))
            batch_size, feat_size, _, _ = patch_feats.shape
            patch_feats_f = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)  # [16, 9, 512]
            att_feats_0 = self.att_embed_image(patch_feats_f)
            # Text branch
            if tags == 1:
                word_embeddings = self.word_embd(tok)  # targets or tok
            else:
                word_embeddings = self.word_embd(targets)  # targets or tok
            word_embeddings = word_embeddings + self.pe[:, : word_embeddings.size(1)]  # x = x + self.pe[:, : x.size(1)]
            H = self.word_encoder(word_embeddings)
            mid = self.word_mlp(H)  # BS * n * r
            p_attn = F.softmax(mid.transpose(-2, -1), dim=-1)
            sturctured_emb_0 = self.att_embed_report(torch.matmul(p_attn, H))  # Text Tokens [16, 9, 512]
            sturctured_emb_text = sturctured_emb_0

            # Ark+ - use precomputed features if available
            if ark_embeddings is not None and ark_predictions is not None:
                # Use precomputed features
                embeddings = ark_embeddings.reshape(batch_size, 16, 86)
                predictions = ark_predictions.cpu().numpy()
            else:
                # Compute features on-the-fly (original method)
                embeddings = torch.FloatTensor().cuda()
                predictions = torch.FloatTensor().cuda()
                with torch.no_grad():
                    pre_logits = self.msg(imageData)
                    preds = [torch.sigmoid(out) for out in pre_logits]
                    preds = torch.cat(preds, dim=1)
                    predictions = torch.cat((predictions, preds), dim=0)  
                    embed = self.msg.generate_embeddings(imageData) 
                    embeddings = torch.cat((embeddings, embed), 0)
                    embeddings = embeddings.reshape(batch_size, 16, 86)
                    predictions = predictions.cpu().numpy()
                            
            # Memory
            memory = self.dictionary(embeddings)  # (batch_size, 16, 512)
            memory_out = self.Memory(att_feats_0, memory, memory)
            att_feats_0 = self.w(att_feats_0 + memory_out) 
            memory_out = self.Memory(sturctured_emb_0, memory, memory)
            sturctured_emb_0 = self.w(sturctured_emb_0 + memory_out)  # (batch_size, 9, 512)
            
            # Mix branch logic - Stage 4 training
            if stage == 4:
                # Mix branch: randomly replace image features with text features
                a = 5  # number of features to replace
                feats = att_feats_0.clone()
                replace = random.sample(range(0, 9), a)
                for i in replace:
                    feats[:, i, :] = sturctured_emb_0[:, i, :]
                output_mix, _ = self.encoder_decoder(feats, targets, mode='forward')
                return output_mix, sturctured_emb_text  # output_mix: Mix branch output, sturctured_emb_text: original text embedding
            else:
                # Normal training - return both text and image branches
                output_t, logit_t = self.encoder_decoder(sturctured_emb_0, targets, mode='forward')
                output_v, logit_v = self.encoder_decoder(att_feats_0, targets, mode='forward')
                return output_t, output_v, sturctured_emb_text, logit_t, logit_v      # output_t: Text Tokens branch output, output_v: Image branch output, sturctured_emb_0: original text embedding

        elif mode == 'sample_cycle':
            # if tags == 1:
            #     word_embeddings = self.word_embd(tok)  # targets or tok
            # else:
            #     word_embeddings = self.word_embd(targets)  # targets or tok
            patch_feats = self.cnn(self.vision(images))
            batch_size, feat_size, _, _ = patch_feats.shape
            word_embeddings = targets
            word_embeddings = word_embeddings + self.pe[:, : word_embeddings.size(1)]  # x = x + self.pe[:, : x.size(1)]
            H = self.word_encoder(word_embeddings)
            mid = self.word_mlp(H)  # BS * n * r
            p_attn = F.softmax(mid.transpose(-2, -1), dim=-1)
            sturctured_emb_0 = self.att_embed_report(torch.matmul(p_attn, H))  # Text Tokens [16, 9, 512]
            sturctured_emb_text = sturctured_emb_0

            # Ark+ - use precomputed features if available
            if ark_embeddings is not None and ark_predictions is not None:
                # Use precomputed features
                embeddings = ark_embeddings.reshape(batch_size, 16, 86)
                predictions = ark_predictions.cpu().numpy()
            else:
                # Compute features on-the-fly (original method)
                embeddings = torch.FloatTensor().cuda()
                predictions = torch.FloatTensor().cuda()
                with torch.no_grad():
                    pre_logits = self.msg(imageData)
                    preds = [torch.sigmoid(out) for out in pre_logits]
                    preds = torch.cat(preds, dim=1)
                    predictions = torch.cat((predictions, preds), dim=0)  
                    embed = self.msg.generate_embeddings(imageData) 
                    embeddings = torch.cat((embeddings, embed), 0)
                    embeddings = embeddings.reshape(batch_size, 16, 86)
                    predictions = predictions.cpu().numpy()
                            
            # Memory
            memory = self.dictionary(embeddings)  # (batch_size, 16, 512)
            memory_out = self.Memory(sturctured_emb_0, memory, memory)
            sturctured_emb_0 = self.w(sturctured_emb_0 + memory_out)  # (batch_size, 9, 512)

            output_t, logit_t = self.encoder_decoder(sturctured_emb_0, tok, mode='forward')

            return sturctured_emb_text, output_t
            
        elif mode == 'sample_v':
            patch_feats = self.cnn(self.vision(images))
            batch_size, feat_size, _, _ = patch_feats.shape
            patch_feats_f = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)  # [16, 9, 512]
            att_feats_0 = self.att_embed_image(patch_feats_f)

            # Ark+ - use precomputed features if available
            if ark_embeddings is not None and ark_predictions is not None:
                # Use precomputed features
                embeddings = ark_embeddings.reshape(batch_size, 16, 86)
                predictions = ark_predictions.cpu().numpy()
            else:
                # Compute features on-the-fly (original method)
                embeddings = torch.FloatTensor().cuda()
                predictions = torch.FloatTensor().cuda()
                with torch.no_grad():
                    pre_logits = self.msg(imageData)
                    preds = [torch.sigmoid(out) for out in pre_logits]
                    preds = torch.cat(preds, dim=1)
                    predictions = torch.cat((predictions, preds), dim=0)  
                    embed = self.msg.generate_embeddings(imageData) 
                    embeddings = torch.cat((embeddings, embed), 0)
                    embeddings = embeddings.reshape(batch_size, 16, 86)
                    predictions = predictions.cpu().numpy()
                            
            # Memory
            memory = self.dictionary(embeddings)  # (batch_size, 16, 512)
            memory_out = self.Memory(att_feats_0, memory, memory)
            att_feats_0 = self.w(att_feats_0 + memory_out)
            
            output_v, probabilities = self.encoder_decoder(att_feats_0, att_feats_0, mode='sample')
            return output_v  # Image branch output

        elif mode == 'sample_t':
            batch_size = imageData.size(0) if imageData is not None else ark_embeddings.size(0)
            if tags == 1:
                word_embeddings = self.word_embd(tok)  # targets or tok
            else:
                word_embeddings = self.word_embd(targets)  # targets or tok
            word_embeddings = word_embeddings + self.pe[:, : word_embeddings.size(1)]  # x = x + self.pe[:, : x.size(1)]
            H = self.word_encoder(word_embeddings)
            mid = self.word_mlp(H)  # BS * n * r
            p_attn = F.softmax(mid.transpose(-2, -1), dim=-1)
            sturctured_emb_0 = self.att_embed_report(torch.matmul(p_attn, H))

            # Ark+ - use precomputed features if available
            if ark_embeddings is not None and ark_predictions is not None:
                # Use precomputed features
                embeddings = ark_embeddings.reshape(batch_size, 16, 86)
                predictions = ark_predictions.cpu().numpy()
            else:
                # Compute features on-the-fly (original method)
                embeddings = torch.FloatTensor().cuda()
                predictions = torch.FloatTensor().cuda()
                with torch.no_grad():
                    pre_logits = self.msg(imageData)
                    preds = [torch.sigmoid(out) for out in pre_logits]
                    preds = torch.cat(preds, dim=1)
                    predictions = torch.cat((predictions, preds), dim=0)  
                    embed = self.msg.generate_embeddings(imageData) 
                    embeddings = torch.cat((embeddings, embed), 0)
                    embeddings = embeddings.reshape(batch_size, 16, 86)
                    predictions = predictions.cpu().numpy()
                            
            # Memory
            memory = self.dictionary(embeddings)  # (batch_size, 16, 512)
            memory_out = self.Memory(sturctured_emb_0, memory, memory)
            sturctured_emb_0 = self.w(sturctured_emb_0 + memory_out)  # (batch_size, 9, 512)

            output_t, probabilities = self.encoder_decoder(sturctured_emb_0, sturctured_emb_0, mode='sample')
            return output_t  # Text branch output

        elif mode == 'sample_mix':
            # Sample from Mix branch
            patch_feats = self.cnn(self.vision(images))
            batch_size, feat_size, _, _ = patch_feats.shape
            patch_feats_f = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)  # [16, 9, 512]
            att_feats_0 = self.att_embed_image(patch_feats_f)
            
            if tags == 1:
                word_embeddings = self.word_embd(tok)  # targets or tok
            else:
                word_embeddings = self.word_embd(targets)  # targets or tok
            word_embeddings = word_embeddings + self.pe[:, : word_embeddings.size(1)]  # x = x + self.pe[:, : x.size(1)]
            H = self.word_encoder(word_embeddings)
            mid = self.word_mlp(H)  # BS * n * r
            p_attn = F.softmax(mid.transpose(-2, -1), dim=-1)
            sturctured_emb_0 = self.att_embed_report(torch.matmul(p_attn, H))

            # Ark+ - use precomputed features if available
            if ark_embeddings is not None and ark_predictions is not None:
                # Use precomputed features
                embeddings = ark_embeddings.reshape(batch_size, 16, 86)
                predictions = ark_predictions.cpu().numpy()
            else:
                # Compute features on-the-fly (original method)
                embeddings = torch.FloatTensor().cuda()
                predictions = torch.FloatTensor().cuda()
                with torch.no_grad():
                    pre_logits = self.msg(imageData)
                    preds = [torch.sigmoid(out) for out in pre_logits]
                    preds = torch.cat(preds, dim=1)
                    predictions = torch.cat((predictions, preds), dim=0)  
                    embed = self.msg.generate_embeddings(imageData) 
                    embeddings = torch.cat((embeddings, embed), 0)
                    embeddings = embeddings.reshape(batch_size, 16, 86)
                    predictions = predictions.cpu().numpy()
                            
            # Memory
            memory = self.dictionary(embeddings)  # (batch_size, 16, 512)
            memory_out = self.Memory(att_feats_0, memory, memory)
            att_feats_0 = self.w(att_feats_0 + memory_out)
            memory_out = self.Memory(sturctured_emb_0, memory, memory)
            sturctured_emb_0 = self.w(sturctured_emb_0 + memory_out)  # (batch_size, 9, 512)
            
            # Mix features for sampling
            a = 5  # number of features to replace
            feats = att_feats_0.clone()
            replace = random.sample(range(0, 9), a)
            for i in replace:
                feats[:, i, :] = sturctured_emb_0[:, i, :]
            
            output_mix, probabilities = self.encoder_decoder(feats, feats, mode='sample')
            return output_mix  # Mix branch output
        
class OmniSwinTransformer(swin.SwinTransformer):
    def __init__(self, num_classes_list, projector_features = None, use_mlp=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_classes_list is not None
        
        self.projector = None 
        if projector_features:
            encoder_features = self.num_features
            self.num_features = projector_features
            if use_mlp:
                self.projector = nn.Sequential(nn.Linear(encoder_features, self.num_features), nn.ReLU(inplace=True), nn.Linear(self.num_features, self.num_features))
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)

        self.omni_heads = []
        for num_classes in num_classes_list:
            self.omni_heads.append(nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        self.omni_heads = nn.ModuleList(self.omni_heads)

    def forward(self, x, head_n=None):
        x = self.forward_features(x)
        if self.projector:
            x = self.projector(x)
        if head_n is not None:
            return x, self.omni_heads[head_n](x)
        else:
            return [head(x) for head in self.omni_heads]
    
    def generate_embeddings(self, x, after_proj = True):
        x = self.forward_features(x)
        if after_proj:
            x = self.projector(x)
        return x 