# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.
 
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from pytorch_transformers.modeling_utils import SQuADHead
from sqlalchemy import false
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_transformers.modeling_bert import (BertEmbeddings, BertAttention, BertEncoder, BertLayer, BertSelfAttention,
        BertSelfOutput, BertIntermediate, BertOutput,
        BertPooler, BertLayerNorm, BertPreTrainedModel,)
from .linear_multihead_attention import LinearMultiheadAttention
import math
from torch.cuda.amp import autocast
from torch.nn.modules.linear import _LinearWithBias
logger = logging.getLogger(__name__)


class CaptionBertSelfAttention(BertSelfAttention):
    """
    Modified from BertSelfAttention to add support for output_hidden_states.
    """
    def __init__(self, config):
        super(CaptionBertSelfAttention, self).__init__(config)

    def forward(self, hidden_states, attention_mask, head_mask=None,
            history_state=None):
        if history_state is not None:
            x_states = torch.cat([history_state, hidden_states], dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs

class CaptionBertAttention(BertAttention):
    """
    Modified from BertAttention to add support for output_hidden_states.
    """
    def __init__(self, config):
        super(CaptionBertAttention, self).__init__(config)
        self.self = CaptionBertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None,
            history_state=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask, history_state)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class CaptionBertEncoder(BertEncoder):
    """
    Modified from BertEncoder to add support for output_hidden_states.
    """
    def __init__(self, config):
        super(CaptionBertEncoder, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([CaptionBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None,
                encoder_history_states=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            history_state = None if encoder_history_states is None else encoder_history_states[i]
            layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i],
                    history_state)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)


class CaptionBertLayer(BertLayer):
    """
    Modified from BertLayer to add support for output_hidden_states.
    """
    def __init__(self, config):
        super(CaptionBertLayer, self).__init__(config)
        self.attention = CaptionBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None,
                history_state=None):
        attention_outputs = self.attention(hidden_states, attention_mask,
                head_mask, history_state)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertImgModel(BertPreTrainedModel):
    """ Expand from BertModel to handle image region features as input
    """
    def __init__(self, config):
        super(BertImgModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = CaptionBertEncoder(config)
        self.pooler = BertPooler(config)

        self.img_dim = config.img_feature_dim
        logger.info('BertImgModel Image Dimension: {}'.format(self.img_dim))
        self.img_feature_type = config.img_feature_type
        if hasattr(config, 'use_img_layernorm'):
            self.use_img_layernorm = config.use_img_layernorm
        else:
            self.use_img_layernorm = None

        if config.img_feature_type == 'dis_code':
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
        elif config.img_feature_type == 'dis_code_t': # transpose
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_size, self.config.hidden_size, bias=True)
        elif config.img_feature_type == 'dis_code_scale': # scaled
            self.input_embeddings = nn.Linear(config.code_dim, config.code_size, bias=True)
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
        else:
            self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            if self.use_img_layernorm:
                #self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)
                self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)
        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,#101,2045,2024
            position_ids=None, head_mask=None, img_feats=None,
            encoder_history_states=None):#5.1275
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            # switch to float if needed + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers
        embedding_output = None
        img_embedding_output = None
        if(input_ids is not None):
            embedding_output = self.embeddings(input_ids, position_ids=position_ids,
                token_type_ids=token_type_ids)#64 * 70 *768
        if encoder_history_states:
            assert img_feats is None, "Cannot take image features while using encoder history states"

        if img_feats is not None:
            if self.img_feature_type == 'dis_code':
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_t': # transpose
                code_emb = self.code_embeddings(img_feats)
                code_emb = code_emb.permute(0, 2, 1)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_scale': # left scaled
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            else:
                img_embedding_output = self.img_embedding(img_feats)#0.1560
                if self.use_img_layernorm:
                    img_embedding_output = self.LayerNorm(img_embedding_output)

                # add dropout on image embedding
                img_embedding_output = self.dropout(img_embedding_output)

            # concatenate two embeddings
        if(embedding_output is None):
            embedding_output = img_embedding_output#结合
            #extended_attention_mask =extended_attention_mask[:,:,:,input_ids.shape[1]:]
        elif(img_embedding_output is None):
            embedding_output = embedding_output#结合
            #extended_attention_mask =extended_attention_mask[:,:,:,:input_ids.shape[1]]
        else:
            embedding_output = torch.cat((embedding_output, img_embedding_output), 1)#结合

        encoder_outputs = self.encoder(embedding_output,
                extended_attention_mask, head_mask=head_mask,
                encoder_history_states=encoder_history_states)#0.2604
        sequence_output = encoder_outputs[0]#
        pooled_output = self.pooler(sequence_output)#0.1419

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs

class BertHeadImgModel(BertImgModel):
    def __init__(self, config):
        super(BertHeadImgModel, self).__init__(config)
        self.decompose_head = nn.ModuleList([CaptionBertLayer(config) for _ in range(config.n_dehead)])
        self.freezed = False

    def freeze_bert(self):
        self.embeddings.requires_grad_ = False
        self.encoder.requires_grad_ = False
        self.freezed = True

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None, img_feats=None, encoder_history_states=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            # switch to float if needed + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers
        embedding_output = None
        img_embedding_output = None
        if(input_ids is not None):
            embedding_output = self.embeddings(input_ids, position_ids=position_ids,
                token_type_ids=token_type_ids)#64 * 70 *768
        if encoder_history_states:
            assert img_feats is None, "Cannot take image features while using encoder history states"

        if img_feats is not None:
            if self.img_feature_type == 'dis_code':
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_t': # transpose
                code_emb = self.code_embeddings(img_feats)
                code_emb = code_emb.permute(0, 2, 1)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_scale': # left scaled
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            else:
                img_embedding_output = self.img_embedding(img_feats)#0.1560
                if self.use_img_layernorm:
                    img_embedding_output = self.LayerNorm(img_embedding_output)

                # add dropout on image embedding
                img_embedding_output = self.dropout(img_embedding_output)

            # concatenate two embeddings
        if(embedding_output is None):
            embedding_output = img_embedding_output#结合
            #extended_attention_mask =extended_attention_mask[:,:,:,input_ids.shape[1]:]
        elif(img_embedding_output is None):
            embedding_output = embedding_output#结合
            #extended_attention_mask =extended_attention_mask[:,:,:,:input_ids.shape[1]]
        else:
            embedding_output = torch.cat((embedding_output, img_embedding_output), 1)#结合

        if self.freezed:
            with torch.no_grad():
                encoder_outputs = self.encoder(embedding_output,
                        extended_attention_mask, head_mask=head_mask,
                        encoder_history_states=encoder_history_states)
                        
        else:
            encoder_outputs = self.encoder(embedding_output,
                        extended_attention_mask, head_mask=head_mask,
                        encoder_history_states=encoder_history_states)
                        
        sequence_output = encoder_outputs[0]#

        head_outputs = sequence_output
        for i, layer_module in enumerate(self.decompose_head):
            head_outputs = layer_module(head_outputs, extended_attention_mask)[0]

        pooled_output = self.pooler(head_outputs)#0.1419

        # add hidden_states and attentions if they are here
        outputs = (head_outputs, pooled_output,) + encoder_outputs[1:]
        return outputs


def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss

class ContrastiveTwins(nn.Module):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """
    def __init__(self,oscar_checkpoint_path,config):
        super(ContrastiveTwins, self).__init__()
        self.num_labels = config.num_labels
        self.config = config
        if(oscar_checkpoint_path!=None and oscar_checkpoint_path!=""):
            self.bert = BertImgModel.from_pretrained(oscar_checkpoint_path, from_tf=bool('.ckpt' in oscar_checkpoint_path), config=config)
        else:
            self.bert = BertImgModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.maxpool_t = nn.AdaptiveMaxPool1d(1)
        self.maxpool_i = nn.AdaptiveMaxPool1d(1)
        self.relu = nn.ReLU()

    def init_code_embedding(self, em):
        self.bert.code_embeddings.weight.data = em.clone()
        
    @autocast()
    def forward(self, input_ids,token_type_ids=None, attention_mask=None,position_ids=None, head_mask=None, img_feats=None, dual=True, sequence_output=False):
        if dual:
            outputs = self.forward_dual(input_ids,token_type_ids, attention_mask,position_ids, head_mask, img_feats, sequence_output)
        else:
            outputs = self.forward_single(input_ids,token_type_ids, attention_mask,position_ids, head_mask, img_feats, sequence_output)
        return outputs
    
    def forward_single(self, input_ids,token_type_ids=None, attention_mask=None,position_ids=None, head_mask=None, img_feats=None, sequence_output=False):
        
        outputs_text = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)
              
        pooled_output = torch.mean(outputs_text[0],1)

        t_feature = pooled_output

        outputs = torch.tanh(t_feature) 

        if sequence_output:
            outputs = (outputs, outputs_text[0])

        return outputs
    
    def forward_dual(self, input_ids,token_type_ids=None, attention_mask=None,position_ids=None, head_mask=None, img_feats=None, sequence_output=False):
        # input ids only includes caption and tags.
        text_len = input_ids.shape[1]//2
        
        outputs_text = self.bert(input_ids[:,:text_len], position_ids=position_ids, token_type_ids=token_type_ids[:,:text_len],
                            attention_mask=attention_mask[:,:text_len], head_mask=head_mask, img_feats=None)
        outputs_img = self.bert(input_ids[:,text_len:], position_ids=position_ids, token_type_ids=token_type_ids[:,text_len:],
                            attention_mask=attention_mask[:,text_len:], head_mask=head_mask, img_feats=img_feats)        

        pooled_output_text = torch.mean(outputs_text[0],1)
        pooled_output_img = torch.mean(outputs_img[0],1)

        pooled_output_text = self.dropout(pooled_output_text)#batch_size,768
        pooled_output_img = self.dropout(pooled_output_img)#batch_size,768

        t_feature = pooled_output_text
        i_feature = pooled_output_img

        t_feature=torch.tanh(t_feature) 
        i_feature=torch.tanh(i_feature) 

        outputs = (i_feature,t_feature) 
        if sequence_output:
            outputs += (outputs_img[0], outputs_text[0])
        return outputs

class ContrastiveHeadTwins(nn.Module):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """
    def __init__(self,oscar_checkpoint_path, config):
        super(ContrastiveHeadTwins, self).__init__()
        self.num_labels = config.num_labels
        self.config = config
        if(oscar_checkpoint_path!=None and oscar_checkpoint_path!=""):
            self.bert = BertHeadImgModel.from_pretrained(oscar_checkpoint_path, from_tf=bool('.ckpt' in oscar_checkpoint_path), config=config)
        else:
            self.bert = BertHeadImgModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.relu = nn.ReLU()

        self.freeze_vinvl()

    def init_code_embedding(self, em):
        self.bert.code_embeddings.weight.data = em.clone()
    
    def freeze_vinvl(self):
        self.bert.freeze_bert()
        
    @autocast()
    def forward(self, input_ids,token_type_ids=None, attention_mask=None,position_ids=None, head_mask=None, img_feats=None, sequence_output=False):
        outputs = self.forward_dual(input_ids,token_type_ids, attention_mask,position_ids, head_mask, img_feats, sequence_output)
        return outputs
       
    def forward_dual(self, input_ids,token_type_ids=None, attention_mask=None,position_ids=None, head_mask=None, img_feats=None, sequence_output=False):
        # input ids only includes caption and tags.
        text_len = input_ids.shape[1]//2
        
        outputs_text = self.bert(input_ids[:,:text_len], position_ids=position_ids, token_type_ids=token_type_ids[:,:text_len],
                            attention_mask=attention_mask[:,:text_len], head_mask=head_mask, img_feats=None)
        outputs_img = self.bert(input_ids[:,text_len:], position_ids=position_ids, token_type_ids=token_type_ids[:,text_len:],
                            attention_mask=attention_mask[:,text_len:], head_mask=head_mask, img_feats=img_feats)        

        pooled_output_text = torch.mean(outputs_text[0],1)
        pooled_output_img = torch.mean(outputs_img[0],1)

        pooled_output_text = self.dropout(pooled_output_text)#batch_size,768
        pooled_output_img = self.dropout(pooled_output_img)#batch_size,768

        t_feature = pooled_output_text
        i_feature = pooled_output_img

        t_feature=torch.tanh(t_feature) 
        i_feature=torch.tanh(i_feature) 

        outputs = (i_feature,t_feature) 
        if sequence_output:
            outputs += (outputs_img[0], outputs_text[0])
        return outputs


class ContrastiveTwinsMeanMax(nn.Module):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """
    def __init__(self,oscar_checkpoint_path,config):
        super(ContrastiveTwinsMeanMax, self).__init__()
        self.num_labels = config.num_labels
        self.config = config
        if(oscar_checkpoint_path!=None and oscar_checkpoint_path!=""):
            self.bert = BertImgModel.from_pretrained(oscar_checkpoint_path, from_tf=bool('.ckpt' in oscar_checkpoint_path), config=config)
        else:
            self.bert = BertImgModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.maxpool_t = nn.AdaptiveMaxPool1d(1)
        self.maxpool_i = nn.AdaptiveMaxPool1d(1)
        self.relu = nn.ReLU()

    def init_code_embedding(self, em):
        self.bert.code_embeddings.weight.data = em.clone()
        
    @autocast()
    def forward(self, input_ids,token_type_ids=None, attention_mask=None,position_ids=None, head_mask=None, img_feats=None, dual=True):#label 8
        if dual:
            outputs = self.forward_dual(input_ids,token_type_ids, attention_mask,position_ids, head_mask, img_feats)
        else:
            outputs = self.forward_single(input_ids,token_type_ids, attention_mask,position_ids, head_mask, img_feats)
        return outputs
    
    def forward_single(self, input_ids,token_type_ids=None, attention_mask=None,position_ids=None, head_mask=None, img_feats=None):
        outputs_text = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)
              
        
        pooled_output_mean = torch.mean(outputs_text[0],1)
        pooled_output_max = self.maxpool_t(outputs_text[0].permute(0,2,1)).squeeze(-1)
        
        pooled_output = torch.cat([pooled_output_mean, pooled_output_max],dim=-1)
        pooled_output = self.dropout(pooled_output)#batch_size,768

        t_feature = pooled_output

        t_feature=torch.tanh(t_feature) 

        return t_feature
    
    def forward_dual(self, input_ids,token_type_ids=None, attention_mask=None,position_ids=None, head_mask=None, img_feats=None):
        text_len = input_ids.shape[1]//2
        #
        outputs_text = self.bert(input_ids[:,:text_len], position_ids=position_ids, token_type_ids=token_type_ids[:,:text_len],
                            attention_mask=attention_mask[:,:text_len], head_mask=head_mask, img_feats=None)
        outputs_img = self.bert(input_ids[:,text_len:], position_ids=position_ids, token_type_ids=token_type_ids[:,text_len:],
                            attention_mask=attention_mask[:,text_len:], head_mask=head_mask, img_feats=img_feats)        

        pooled_output_text = torch.mean(outputs_text[0],1)
        pooled_output_txt_max = self.maxpool_t(outputs_text[0].permute(0,2,1)).squeeze(-1)
        pooled_output_text = torch.cat([pooled_output_text, pooled_output_txt_max],dim=-1)

        pooled_output_img = torch.mean(outputs_img[0],1)
        pooled_output_img_max = self.maxpool_t(outputs_img[0].permute(0,2,1)).squeeze(-1)
        pooled_output_img = torch.cat([pooled_output_img, pooled_output_img_max],dim=-1)

        pooled_output_text = self.dropout(pooled_output_text)#batch_size,768
        pooled_output_img = self.dropout(pooled_output_img)#batch_size,768

        t_feature = pooled_output_text
        i_feature = pooled_output_img

        t_feature=torch.tanh(t_feature) 
        i_feature=torch.tanh(i_feature) 

        outputs = (i_feature,t_feature) 
        return outputs

class ContrastiveTwoLevel(nn.Module):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """
    def __init__(self,oscar_checkpoint_path,config):
        super(ContrastiveTwoLevel, self).__init__()
        self.num_labels = config.num_labels
        self.config = config
        if(oscar_checkpoint_path!=None and oscar_checkpoint_path!=""):
            self.bert = BertImgModel.from_pretrained(oscar_checkpoint_path, from_tf=bool('.ckpt' in oscar_checkpoint_path), config=config)
        else:
            self.bert = BertImgModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.relu = nn.ReLU()

        # level 2
    def init_code_embedding(self, em):
        self.bert.code_embeddings.weight.data = em.clone()
        
    @autocast()
    def forward(self, input_ids,token_type_ids=None, attention_mask=None,position_ids=None, head_mask=None, img_feats=None, dual=True):#label 8
        if dual:
            outputs = self.forward_dual(input_ids,token_type_ids, attention_mask,position_ids, head_mask, img_feats)
        else:
            outputs = self.forward_single(input_ids,token_type_ids, attention_mask,position_ids, head_mask, img_feats)
        return outputs
    
    def forward_single(self, input_ids,token_type_ids=None, attention_mask=None,position_ids=None, head_mask=None, img_feats=None):
        outputs_text = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)
              
        if(not self.config.bert_cls):
            pooled_output_text = torch.mean(outputs_text[0],1)
        else:
            pooled_output_text = outputs_text[1]        
        pooled_output_text = self.dropout(pooled_output_text)#batch_size,768

        t_feature = pooled_output_text

        t_feature=torch.tanh(t_feature) 

        return t_feature
    
    def forward_dual(self, input_ids,token_type_ids=None, attention_mask=None,position_ids=None, head_mask=None, img_feats=None):
        text_len = input_ids.shape[1]//2
        #
        outputs_text = self.bert(input_ids[:,:text_len], position_ids=position_ids, token_type_ids=token_type_ids[:,:text_len],
                            attention_mask=attention_mask[:,:text_len], head_mask=head_mask, img_feats=None)
        outputs_img = self.bert(input_ids[:,text_len:], position_ids=position_ids, token_type_ids=token_type_ids[:,text_len:],
                            attention_mask=attention_mask[:,text_len:], head_mask=head_mask, img_feats=img_feats)        

        pooled_output_text = torch.mean(outputs_text[0],1)
        pooled_output_img = torch.mean(outputs_img[0],1)
       
        pooled_output_text = self.dropout(pooled_output_text)#batch_size,768
        pooled_output_img = self.dropout(pooled_output_img)#batch_size,768

        t_feature = pooled_output_text
        i_feature = pooled_output_img

        t_feature=torch.tanh(t_feature) 
        i_feature=torch.tanh(i_feature) 

        outputs = (i_feature,t_feature) 
        return outputs

class ImageBertForSequenceClassification(BertPreTrainedModel):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """
    def __init__(self, config):
        super(ImageBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.config = config
        if config.img_feature_dim > 0:
            self.bert = BertImgModel(config)
        else:
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if hasattr(config, 'classifier'):
            if not hasattr(config, 'cls_hidden_scale'): 
                config.cls_hidden_scale = 2

            if config.classifier == 'linear':
                self.classifier = nn.Linear(config.hidden_size,
                                            self.config.num_labels)
            elif config.classifier == 'mlp':
                self.classifier = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size * config.cls_hidden_scale),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * config.cls_hidden_scale, self.config.num_labels)
                )
        else:
            self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)  # original
        self.init_weights()

    def init_code_embedding(self, em):
        self.bert.code_embeddings.weight.data = em.clone()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, 
            position_ids=None, head_mask=None, img_feats=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            if self.num_labels == 1: #  doing regression
                loss_fct = MSELoss()
                labels = labels.to(torch.float)
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if self.loss_type == 'kl':
                    # KL Loss: https://github.com/uclanlp/visualbert/blob/master/pytorch_pretrained_bert/modeling.py
                    loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
                    log_softmax = torch.nn.LogSoftmax(dim=-1)
                    reshaped_logits = logits.contiguous().view(-1, 3129)
                    reshaped_logits = log_softmax(reshaped_logits)
                    loss = loss_fct(reshaped_logits, labels.contiguous())
                elif self.loss_type == 'bce': # [VQA]
                    loss = instance_bce_with_logits(logits, labels)
                else: # cross_entropy [GQA, Retrieval, Captioning]
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs

class BERTInteractionHead(nn.Module):
    ''' Interaction head for the VLDeformer, perform fully-connected interaction on the features.
    '''
    def __init__(self, config, n_layers=1):
        super(BERTInteractionHead, self).__init__()
        self.layer = nn.ModuleList([CaptionBertLayer(config) for _ in range(n_layers)])
        self.pooler  = BertPooler(config)

        self.matcher = nn.Linear(config.hidden_size, 1)
    
    def forward(self, features, attention_mask):
        fused_features = features
        if len(attention_mask.shape) != 4:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        for i, layer_module in enumerate(self.layer):
            fused_features = layer_module(fused_features, attention_mask)[0]

        sequence_output = self.pooler(fused_features)

        cls = self.matcher(sequence_output).squeeze()

        return cls

class BERTInteractionCOSHead(nn.Module):
    ''' Interaction head for the VLDeformer, perform fully-connected interaction on the features.
    '''
    def __init__(self, config, n_layers=1):
        super(BERTInteractionCOSHead, self).__init__()
        self.layer = nn.ModuleList([CaptionBertLayer(config) for _ in range(n_layers)])
    
    def forward(self, features, attention_mask):
        text_len = 35
        fused_features = features
        if len(attention_mask.shape) != 4:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        for i, layer_module in enumerate(self.layer):
            fused_features = layer_module(fused_features, attention_mask)[0]

        r_t = torch.tanh(torch.mean(fused_features[:,:text_len], dim=1))
        r_i = torch.tanh(torch.mean(fused_features[:,text_len:], dim=1))

        sim = F.cosine_similarity(r_t,r_i)
        
        return sim

class SelfAttentionHead(nn.Module):
    ''' Interaction head for the VLDeformer, perform fully-connected interaction on the features.
    '''
    def __init__(self, config, n_layers=2):
        super(SelfAttentionHead, self).__init__()
        layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.pooler  = BertPooler(config)

        self.matcher = nn.Linear(config.hidden_size, 1)
    
    def forward(self, features, attention_mask):
        fused_features = features
        # if len(attention_mask.shape) != 4:
        #     attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        fused_features = self.encoder(fused_features)

        sequence_output = self.pooler(fused_features)

        cls = self.matcher(sequence_output).squeeze()

        return cls

class SelfLinearAttentionCOSHead(nn.Module):
    ''' Interaction head for the VLDeformer, perform fully-connected interaction on the features.
    '''
    def __init__(self, config, n_layers=1):
        super(SelfLinearAttentionCOSHead, self).__init__()
        self.encoder = LinearMultiheadAttention(embed_dim=config.hidden_size, nhead=4)
            
    def forward(self, features, attention_mask):
        text_len = 35

        fused_features = features

        fused_features = self.encoder(fused_features)

        r_t = torch.tanh(torch.mean(fused_features[:,:text_len], dim=1))
        r_i = torch.tanh(torch.mean(fused_features[:,text_len:], dim=1))

        cls = F.cosine_similarity(r_t,r_i)
        return cls

class CosineHead(nn.Module):
    ''' Interaction head for the VLDeformer, perform fully-connected interaction on the features.
    '''
    def __init__(self, config=None, n_layers=1):
        super(CosineHead, self).__init__()
        
    
    def forward(self, features, attention_mask):
        text_len = 35
        
        r_t = torch.tanh(torch.mean(features[:,:text_len], dim=1))
        r_i = torch.tanh(torch.mean(features[:,text_len:], dim=1))

        sim = F.cosine_similarity(r_t,r_i)

        return sim

from torch.nn import Parameter
from torch.nn.modules.activation import _LinearWithBias
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiHeadCrossAttention, self).__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
            if self._qkv_same_embed_dim:
                xavier_uniform_(self.in_proj_weight)
            else:
                xavier_uniform_(self.q_proj_weight)
                xavier_uniform_(self.k_proj_weight)
                xavier_uniform_(self.v_proj_weight)

            if self.in_proj_bias is not None:
                constant_(self.in_proj_bias, 0.)
                constant_(self.out_proj.bias, 0.)
            if self.bias_k is not None:
                xavier_normal_(self.bias_k)
            if self.bias_v is not None:
                xavier_normal_(self.bias_v)

    def forward(self,  query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        return F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)

class ContrastiveTwinsSEP(nn.Module):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """
    def __init__(self,oscar_checkpoint_path,config):
        super(ContrastiveTwinsSEP, self).__init__()
        self.num_labels = config.num_labels
        self.config = config
        if(oscar_checkpoint_path!=None and oscar_checkpoint_path!=""):
            self.bert = BertImgModel.from_pretrained(oscar_checkpoint_path, from_tf=bool('.ckpt' in oscar_checkpoint_path), config=config)
        else:
            self.bert = BertImgModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.relu = nn.ReLU()
        """
         self.out_linear=config.out_linear
        if(config.out_linear):        
            self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)  # original
            self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)
            self.linear3 = nn.Linear(config.hidden_size, config.hidden_size)  # original
            self.linear4 = nn.Linear(config.hidden_size, config.hidden_size)       
        """


    def init_code_embedding(self, em):
        self.bert.code_embeddings.weight.data = em.clone()
    @autocast()
    def forward(self, input_ids,token_type_ids=None, attention_mask=None,position_ids=None, head_mask=None, img_feats=None, a_token_len=None, b_token_len=None):#label 8
        text_len = input_ids.shape[1]//2
        batch_size = len(input_ids)
        #
        outputs_text = self.bert(input_ids[:,:text_len], position_ids=position_ids, token_type_ids=token_type_ids[:,:text_len],
                            attention_mask=attention_mask[:,:text_len], head_mask=head_mask, img_feats=None)
        outputs_img = self.bert(input_ids[:,text_len:], position_ids=position_ids, token_type_ids=token_type_ids[:,text_len:],
                            attention_mask=attention_mask[:,text_len:], head_mask=head_mask, img_feats=img_feats)        

        pooled_output_img = outputs_img[0][torch.range(0, batch_size-1).long(),b_token_len]
        pooled_output_text = outputs_text[0][torch.range(0, batch_size-1).long(),a_token_len]

        pooled_output_text = self.dropout(pooled_output_text)#batch_size,768
        pooled_output_img = self.dropout(pooled_output_img)#batch_size,768
        """
         if(self.out_linear):
            pooled_output_text = self.linear1(pooled_output_text)
            pooled_output_text = self.linear2(pooled_output_text)
            pooled_output_text = self.linear3(pooled_output_text)
            pooled_output_text = self.linear4(pooled_output_text)

            pooled_output_img = self.linear1(pooled_output_img)
            pooled_output_img = self.linear2(pooled_output_img)
            pooled_output_img = self.linear3(pooled_output_img)
            pooled_output_img = self.linear4(pooled_output_img)       
        """

        t_feature = pooled_output_text
        i_feature = pooled_output_img
        t_feature=torch.tanh(t_feature) 
        i_feature=torch.tanh(i_feature) 

        outputs = (i_feature,t_feature) 
        return outputs
    def t_forwards(self, input_ids,token_type_ids=None, attention_mask=None,position_ids=None, head_mask=None, img_feats=None):
        text_len = input_ids.shape[1]//2
        outputs_text = self.bert(input_ids[:,:text_len], position_ids=position_ids, token_type_ids=token_type_ids[:,:text_len],
                            attention_mask=attention_mask[:,:text_len], head_mask=head_mask, img_feats=None)
        pooled_output_text = torch.mean(outputs_text[0],1)
        pooled_output_text = self.dropout(pooled_output_text)#batch_size,768
        t_feature=torch.tanh(pooled_output_text)
        return t_feature
    def i_forwards(self, input_ids,token_type_ids=None, attention_mask=None,position_ids=None, head_mask=None, img_feats=None):
        text_len = input_ids.shape[1]//2
        outputs_img = self.bert(input_ids[:,text_len:], position_ids=position_ids, token_type_ids=token_type_ids[:,text_len:],
                            attention_mask=attention_mask[:,text_len:], head_mask=head_mask, img_feats=img_feats) 
        pooled_output_img = torch.mean(outputs_img[0],1)
        pooled_output_img = self.dropout(pooled_output_img)#batch_size,768
        i_feature = pooled_output_img
        i_feature=torch.tanh(i_feature) 
        return i_feature                            
