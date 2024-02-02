import os
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchcrf import CRF
from transformers import AutoModel, AutoConfig
from transformers import ViTModel, CLIPVisionModel
from transformers.modeling_utils import apply_chunking_to_forward
from transformers.activations import ACT2FN

from .model_loss import ContrastiveLoss
from utils import get_extended_attention_mask, get_extended_attention_mask_v2
import model.resnet as resnet
from model.resnet_utils import myResnet

logger = logging.getLogger(__name__)

class MMNER(nn.Module):
    def __init__(self, args, label2idx):
        super(MMNER, self).__init__()
        self.args = args

        if args.use_image:
            hidden_size = 768
            if args.vit_model == 'google/vit-base-patch16-224':
                self.vit = ViTModel.from_pretrained(args.vit_model)
                self.image_dense_cl = nn.Linear(197*hidden_size, hidden_size)
            elif args.vit_model == 'openai/clip-vit-base-patch32':
                self.vit = CLIPVisionModel.from_pretrained(args.vit_model)
                self.image_dense_cl = nn.Linear(50*hidden_size, hidden_size)
            elif args.vit_model == 'openai/clip-vit-base-patch16': # 暂不可用
                self.vit = CLIPVisionModel.from_pretrained(args.vit_model)
                self.image_dense_cl = nn.Linear(50*hidden_size, hidden_size)
            else:
                exit('need to add model {}'.format(args.vit_model))
            self.text_dense_cl = nn.Linear(hidden_size, hidden_size)
            self.relu = nn.ReLU()
            self.text_output_cl = nn.Linear(hidden_size, hidden_size)
            self.image_output_cl = nn.Linear(hidden_size, hidden_size)
            self.image_dense_cl_resnet = nn.Linear(2048, hidden_size)
            self.image_output_cl_resnet = nn.Linear(hidden_size, hidden_size)

            net = getattr(resnet, 'resnet152')()
            net.load_state_dict(torch.load(os.path.join(args.resnet_root, 'resnet152.pth')))
            self.my_resnet = myResnet(net, False, args.device)

            self.dense_resnet = nn.Linear(2048, hidden_size)
            self.dense_cat = nn.Linear(hidden_size * 2, hidden_size)
            self.w_txt = nn.Linear(hidden_size, hidden_size)
            self.w_img = nn.Linear(hidden_size, hidden_size)

        self.emb = AutoModel.from_pretrained(args.bert_model)
        # self.emb = AutoModel.from_pretrained(args.bert_model, add_pooling_layer=False) # add_pooling_layer default True

        config = AutoConfig.from_pretrained(args.bert_model)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if args.use_image:
            # self.interaction_module = nn.ModuleList([BertLayer(config) for _ in range(self.args.interaction_layer)])
            self.bert_att_resnet = BertAttention(config)
            self.bert_att_vit = BertAttention(config)

        self.classifier = nn.Linear(config.hidden_size, len(label2idx))

        if args.use_crf:
            self.crf = CRF(num_tags=len(label2idx), batch_first=True)

        device = torch.device('cuda' if args.device=='cuda' and torch.cuda.is_available() else 'cpu')
        self.contrastive_loss = ContrastiveLoss(device=device) if args.use_image else None

    def resize_token_embeddings(self, len_tokenizer):
        """修改了tokens, 所以需要resize一下"""
        self.emb.resize_token_embeddings(len_tokenizer)

    def forward(self, batch_inputs, word_to_token_index, batch_label, img_fe=None, bert_att_mask=None, resnet_img_fe=None, mode='train'):
        """
        Arguments:
            bert_att_mask [bs, seq_len]
        """

        if self.args.use_image:
            # 处理图片
            # print(img_fe['pixel_values'].shape) # torch.Size([32, 3, 224, 224])
            vit_out = self.vit(**img_fe)
            # print(vit_out.last_hidden_state.shape) # torch.Size([32, 197, 768])
            logits_img = vit_out.last_hidden_state

            with torch.no_grad():
                imgs_f, img_mean, img_att = self.my_resnet(resnet_img_fe)

        # 处理文本
        # print('batch_inputs shape:', {k: v.shape for k, v in batch_inputs.items()})
        input_ids = batch_inputs.input_ids
        attention_mask = batch_inputs['attention_mask'] # [bs, seq_len]
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # select the word index
        if self.args.use_image:
            token_rep = outputs.hidden_states[self.args.text_encoder_layer]
        else:
            token_rep = outputs.last_hidden_state # (batch_size, max_token_len, hidden_size)
        _, _, hidden_size = token_rep.shape
        word_rep = torch.gather(token_rep[:, :, :], dim=1, index=word_to_token_index.unsqueeze(-1).expand(-1, -1, hidden_size))
        # logger.info('{}, {}, {}'.format(vit_out.last_hidden_state.shape, word_rep.shape, bert_att.shape))

        # contrastive_loss = self.cal_contrastive_loss_v2(word_rep, logits_img) if self.args.use_image else 0
        contrastive_loss = self.cal_contrastive_loss_v3(outputs.pooler_output, logits_img) if self.args.use_image else 0
        cl_resnet = self.cal_contrastive_loss_resnet(outputs.pooler_output, imgs_f) if self.args.use_image else 0

        if not self.args.mmi:
            logits_img = None

        if self.args.use_image:
            word_rep_res = word_rep.clone()
            extended_att_mask = get_extended_attention_mask(bert_att_mask)
            """更换一下多模态交互方式
            for idx in range(self.args.interaction_layer):
                interaction = self.interaction_module[idx](word_rep, logits_img, attention_mask=extended_att_mask)
                interaction_hidden_states = interaction[0]
                word_rep = interaction_hidden_states
            """
            ext_att_mask = get_extended_attention_mask_v2(bert_att_mask)
            fuse_v, _, _ = self.bert_att_vit(word_rep, logits_img, logits_img, None, attention_mask=ext_att_mask)
            fuse_vit = fuse_v[0]

            # imgs_f = imgs_f.unsqueeze(1).expand(-1, 64, -1)
            # imgs_f = self.dense_resnet(imgs_f) # [bs, 64, 768]
            # fuse_score = torch.matmul(word_rep, imgs_f.transpose(-1, -2)) # [bs, seq_len, 64]
            # fuse_score = F.softmax(fuse_score, dim=-1)
            # fuse_image = torch.matmul(fuse_score, imgs_f) # [bs, seq_len, hidden_size]

            imgs_f = self.dense_resnet(imgs_f).unsqueeze(1).expand_as(word_rep_res)
            fuse_res, _, _ = self.bert_att_resnet(word_rep_res, imgs_f, imgs_f, None, attention_mask=extended_att_mask)
            fuse_image = fuse_res[0]

        if self.args.use_image:
            # cat_rep = torch.cat((word_rep, fuse_image), -1)
            cat_rep = torch.cat((fuse_vit, fuse_image), -1)
            word_rep = self.dense_cat(cat_rep)

        word_rep = self.dropout(word_rep)  # (batch_size, max_seq_len, hidden_size)
        logits = self.classifier(word_rep)  # (batch_size, max_seq_len, label_size)

        if mode == 'train':
            if self.args.use_crf:
                crf_mask = (batch_label != -100)
                batch_label = torch.where(batch_label==-100, 0, batch_label) # crf not support -100, change to 0
                log_likelihood = self.crf(logits, batch_label, mask=crf_mask, reduction='mean')
                loss = - log_likelihood # negative log likelihood
                # total_loss = (0.1 * contrastive_loss + 0.9 * loss) * 2
                # total_loss = (0.1 * contrastive_loss + 0.1 * cl_resnet +  0.9 * loss) * 2
                cl_loss = contrastive_loss + cl_resnet
                alpha = self.args.alpha
                total_loss = (alpha * loss + (1-alpha) * cl_loss) * 2
                # total_loss = loss
                return logits, total_loss
            else:
                return logits, None
        else: # eval
            if self.args.use_crf:
                crf_mask = (batch_label != -100)
                pred_tags = self.crf.decode(logits, mask=crf_mask) # List[List[int]]
                return pred_tags
            else:
                return logits.argmax(dim=-1).cpu().numpy().tolist()

    def cal_contrastive_loss_v1(self, logits_txt, logits_img):
        # convert shape to [batch_size, 768]
        logits_txt = logits_txt.sum(dim=1, keepdim=False)
        logits_img = logits_img.sum(dim=1, keepdim=False)

        cos_simi = nn.functional.cosine_similarity(logits_txt, logits_img) # [batch_size]
        loss = -nn.functional.log_softmax(cos_simi, dim=0).mean(dim=0)

        return loss

    def cal_contrastive_loss_v2(self, logits_txt, logits_img):
        # convert shape to [batch_size, 768]
        logits_txt = logits_txt.sum(dim=1, keepdim=False)
        logits_img = logits_img.sum(dim=1, keepdim=False)
        # logits_txt = logits_txt.mean(dim=1, keepdim=False)
        # logits_img = logits_img.mean(dim=1, keepdim=False)

        return self.contrastive_loss(logits_txt, logits_img)
    
    def cal_contrastive_loss_v3(self, sequence_output_pooler, logits_img):
        bs = logits_img.shape[0]
        logits_img = logits_img.view(bs, -1)

        text_output_cl = self.text_output_cl(self.relu(self.text_dense_cl(sequence_output_pooler)))
        image_output_cl = self.image_output_cl(self.relu(self.image_dense_cl(logits_img)))
        return self.contrastive_loss(text_output_cl, image_output_cl)
    
    def cal_contrastive_loss_resnet(self, sequence_output_pooler, imgs_f):
        text_output_cl = self.text_output_cl(self.relu(self.text_dense_cl(sequence_output_pooler)))
        image_output_cl = self.image_output_cl_resnet(self.relu(self.image_dense_cl_resnet(imgs_f)))
        return self.contrastive_loss(text_output_cl, image_output_cl)


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config) # Multi-Head Attention   Add & Norm
        self.intermediate = BertIntermediate(config) # Feed Forward
        self.output = BertOutput(config) # Add & Norm

        self.seq_len_dim = 1
        # chunk_size_feed_forward 默认为0 the feed forward layer is not chunked (ref https://huggingface.co/docs/transformers/main_classes/configuration)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward

    def forward(self, hidden_states, img_hidden_state, attention_mask=None, output_attentions=False, output_qks=None):
        self_attention_outputs, fusion_output, qks = self.attention(
            hidden_states, 
            hidden_states, 
            hidden_states, 
            img_hidden_state, 
            attention_mask,
            output_attentions=output_attentions,
            output_qks=output_qks)
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:] # add attentions if we output them

        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim,
                                                 attention_output, fusion_output) # [bs, seq_len, hidden_size]

        outputs = (layer_output,) + outputs
        if output_qks:
            outputs += (qks,)

        return outputs

    def feed_forward_chunk(self, attention_output, fusion_output):
        intermediate_output = self.intermediate(attention_output, fusion_output) # [bs, seq_len, intermediate_size]
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = MultiHeadAttention(config)
        self.output = BertSelfOutput(config) # Add & Norm

    def forward(self, query_hs, key_hs, value_hs, img_hidden_state, attention_mask=None, output_attentions=False, output_qks=None):
        self_outputs, fusion_output, qks = self.self(query_hs, key_hs, value_hs, img_hidden_state, attention_mask, output_attentions, output_qks)
        attention_output = self.output(self_outputs[0], query_hs) # [bs, seq_len, hidden_size]
        outputs = (attention_output, ) + self_outputs[1:] # add attentions if we output them

        return outputs, fusion_output, qks


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size) # "intermediate_size": 3072
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.fusion_dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states, fusion_output=None):
        # hidden_states: [bs, seq_len, hidden_size]
        hidden_states = self.dense(hidden_states)
        if fusion_output is not None:
            fusion_states = self.fusion_dense(fusion_output)
            hidden_states = hidden_states + fusion_states # 融合
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        # hidden_states: [bs, seq_len, intermediate_size]
        # input_tensor: [bs, seq_len, hidden_size]
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads # 12
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size        # 768

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.fusion = ModalFusion(config)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # [bs, num_head, seq_len, att_size]

    def forward(self, query_hs, key_hs, value_hs, img_hidden_state, attention_mask=None, output_attentions=False, output_qks=None):

        key_layer = self.transpose_for_scores(self.key(key_hs))       # [bs, num_head, seq_kv, att_size]
        value_layer = self.transpose_for_scores(self.value(value_hs)) # [bs, num_head, seq_kv, att_size]
        query_layer = self.transpose_for_scores(self.key(query_hs))   # [bs, num_head, seq_q, att_size]

        qks = (key_layer, value_layer) if output_qks else None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [bs, num_head, seq_q, seq_kv]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)     # [bs, num_head, seq_q, att_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # [bs, seq_q, num_head, att_size]
        new_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape) # [bs, seq_q, hidden_states]

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        fusion_output = self.fusion(context_layer, img_hidden_state) if img_hidden_state is not None else None

        return outputs, fusion_output, qks


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        hidden_states [bs, seq_len, hidden_size]
        input_tensor  [bs, seq_len, hidden_size]
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # Add & Norm

        return hidden_states # [bs, seq_len, hidden_size]

class ModalFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fusion_function = 'softmax'

    def forward(self, hidden_states, img_hidden_state):
        fusion_scores = torch.matmul(hidden_states, img_hidden_state.transpose(-1, -2)) # [bs, seq_len, img_len]
        if self.fusion_function == 'softmax':
            fusion_probs = nn.Softmax(dim=-1)(fusion_scores)
            fusion_output = torch.matmul(fusion_probs, img_hidden_state) # [bs, seq_len, hidden_size]
        elif self.fusion_function == 'max':
            fusion_probs = fusion_scores.max(dim=-1)
            fusion_output = torch.matmul(fusion_probs, img_hidden_state) # [bs, seq_len, hidden_size]
        return fusion_output
