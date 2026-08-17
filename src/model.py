import os
import copy
import math
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_bert import *
from huggingface_hub import hf_hub_download

from custom_config import LongBERTConfig
from module import CrossAttention, TransContextModel


def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LongBERTOutput:
    def __init__(self):
        self.last_hidden_state = None
        self.pooled_output = None
        self.hidden_states = None
        self.last_attn_weights = None


def dilated_attention(query, key, value, key_padding_mask=None, attn_mask=None, dropout=None, training=None):
    assert len(query.shape) == 5, "query must be (batch, n_head, n_seg, seg_len, dim)"
    assert query.shape == key.shape == value.shape, "q/k/v shape mismatch"

    batch_size, n_head, n_seg, seg_len, dim = query.shape

    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(dim)

    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(3)
        if key_padding_mask.dtype != torch.bool:
            key_padding_mask = key_padding_mask.bool()
        scores = scores.masked_fill(key_padding_mask, float("-inf"))

    if attn_mask is not None:
        if len(attn_mask.shape) == 3:
            attn_mask = attn_mask.view(1, 1, n_seg, seg_len, seg_len)
        elif len(attn_mask.shape) == 4:
            attn_mask = attn_mask.view(batch_size, 1, n_seg, seg_len, seg_len)
        if attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.bool()
        scores = scores.masked_fill(attn_mask, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights)

    if dropout is not None and training:
        attn_weights = F.dropout(attn_weights, p=dropout, training=training)

    attn_output = torch.matmul(attn_weights, value)
    return attn_output, attn_weights, None


class DilatedMultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, n_head, segment_size, dilated_rate, dropout=0.1):
        super(DilatedMultiheadAttention, self).__init__()
        assert embedding_dim % n_head == 0, "The embedding dimension should be divisible by the number of heads"
        assert len(segment_size) == len(dilated_rate), "segment_size and dilated_rate should have the same length"

        self.d_proj = embedding_dim // n_head
        self.n_head = n_head
        self.segment_size = segment_size
        self.dilated_rate = dilated_rate
        self.dropout = dropout

        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        batch_size, seq_len, embedding_dim = query.shape
        attn_output = torch.zeros_like(query)

        for seg_size, dil_rate in zip(self.segment_size, self.dilated_rate):
            pad_len = (seg_size - seq_len % seg_size) % seg_size
            _seq_len = seq_len + pad_len

            if pad_len > 0:
                pad = torch.zeros(batch_size, pad_len, embedding_dim, device=query.device, dtype=query.dtype)
                _query = torch.cat([query, pad], dim=1)
                _key = torch.cat([key, pad], dim=1)
                _value = torch.cat([value, pad], dim=1)

                if key_padding_mask is not None:
                    pad_mask = torch.ones(
                        batch_size,
                        pad_len,
                        device=key_padding_mask.device,
                        dtype=key_padding_mask.dtype,
                    )
                    _key_padding_mask = torch.cat([key_padding_mask, pad_mask], dim=1)
                else:
                    _key_padding_mask = None
            else:
                _query, _key, _value = query, key, value
                _key_padding_mask = key_padding_mask

            n_segment = _seq_len // seg_size

            _query = _query.view(batch_size, n_segment, seg_size, embedding_dim)
            _key = _key.view(batch_size, n_segment, seg_size, embedding_dim)
            _value = _value.view(batch_size, n_segment, seg_size, embedding_dim)

            _query = _query[:, :, ::dil_rate, :]
            _key = _key[:, :, ::dil_rate, :]
            _value = _value[:, :, ::dil_rate, :]
            dil_seg_len = _query.shape[2]

            _query = self.q_proj(_query)
            _key = self.k_proj(_key)
            _value = self.v_proj(_value)

            _query = _query.reshape(batch_size, n_segment * dil_seg_len, self.n_head, self.d_proj)
            _key = _key.reshape(batch_size, n_segment * dil_seg_len, self.n_head, self.d_proj)
            _value = _value.reshape(batch_size, n_segment * dil_seg_len, self.n_head, self.d_proj)

            _query_flat = _query.permute(0, 2, 1, 3)
            _key_flat = _key.permute(0, 2, 1, 3)
            _value_flat = _value.permute(0, 2, 1, 3)

            cls_q = _query_flat[:, :, 0:1, :]
            cls_scores = torch.matmul(cls_q, _key_flat.transpose(-2, -1)) / (self.d_proj ** 0.5)

            if _key_padding_mask is not None:
                cls_key_padding_mask = _key_padding_mask.view(batch_size, n_segment, seg_size)[:, :, ::dil_rate]
                cls_key_padding_mask = cls_key_padding_mask.reshape(batch_size, 1, 1, n_segment * dil_seg_len)
                cls_scores = cls_scores.masked_fill(cls_key_padding_mask.bool(), float("-inf"))

            cls_attn = torch.softmax(cls_scores, dim=-1)
            cls_attn = torch.dropout(cls_attn, p=self.dropout, train=self.training)
            cls_global_out = torch.matmul(cls_attn, _value_flat)

            _query = _query_flat.view(batch_size, self.n_head, n_segment, dil_seg_len, self.d_proj)
            _key = _key_flat.view(batch_size, self.n_head, n_segment, dil_seg_len, self.d_proj)
            _value = _value_flat.view(batch_size, self.n_head, n_segment, dil_seg_len, self.d_proj)

            if _key_padding_mask is not None:
                _key_padding_mask = _key_padding_mask.view(batch_size, n_segment, seg_size)[:, :, ::dil_rate]

            _attn_out, _, _ = dilated_attention(
                _query,
                _key,
                _value,
                key_padding_mask=_key_padding_mask,
                attn_mask=attn_mask,
                dropout=self.dropout,
                training=self.training,
            )

            attn_out_resized = torch.zeros(
                batch_size,
                n_segment,
                seg_size,
                self.n_head,
                self.d_proj,
                device=_attn_out.device,
                dtype=_attn_out.dtype,
            )

            attn_out_resized[:, :, ::dil_rate, :, :] = _attn_out.permute(0, 2, 3, 1, 4)
            attn_out_resized[:, 0, 0, :, :] = attn_out_resized[:, 0, 0, :, :] + cls_global_out.squeeze(2)

            attn_out_flat = attn_out_resized.reshape(batch_size, n_segment, seg_size, embedding_dim)
            attn_out_seq = attn_out_flat.reshape(batch_size, _seq_len, embedding_dim)

            if pad_len > 0:
                attn_out_seq = attn_out_seq[:, :seq_len, :]

            attn_output += attn_out_seq / len(self.segment_size)

        return attn_output


class LongBERTEmbeddings(nn.Module):
    def __init__(self, config):
        super(LongBERTEmbeddings, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=3)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12, elementwise_affine=True)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids, position_ids):
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = word_embeddings + token_type_embeddings + position_embeddings
        else:
            embeddings = word_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)


class LongBERTLayer(nn.Module):
    def __init__(self, config):
        super(LongBERTLayer, self).__init__()
        self.attention = DilatedMultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.segment_size,
            config.dilated_rate,
            dropout=config.attention_probs_dropout_prob,
        )
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        residual = query

        attn_output = self.attention(query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        attn_output = residual + self.dropout(attn_output)
        attn_output = self.LayerNorm(attn_output)

        ffn_output = self.linear1(attn_output)
        ffn_output = residual + self.dropout(ffn_output)
        ffn_output = self.LayerNorm(ffn_output)

        return ffn_output


class LongBERTPooler(nn.Module):
    def __init__(self, config):
        super(LongBERTPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.activation = nn.Tanh()

    def forward(self, hidden_state):
        return self.activation(self.dense(hidden_state[:, 0, :]))


class LongBERTEncoder(nn.Module):
    def __init__(self, config):
        super(LongBERTEncoder, self).__init__()
        config.attention_probs_dropout_prob = 0.1
        self.layer = clone(LongBERTLayer(config), config.num_hidden_layers)
        self.pooler = LongBERTPooler(config)
        self.longbert_output = LongBERTOutput()

    def forward(self, hidden_state, attention_mask=None, output_hidden_states=False):
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        hidden_states = tuple()

        for layer in self.layer:
            hidden_state = layer(hidden_state, hidden_state, hidden_state, key_padding_mask=key_padding_mask)
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

        self.longbert_output.pooled_output = self.pooler(hidden_state)
        self.longbert_output.last_hidden_state = hidden_state
        if output_hidden_states:
            self.longbert_output.hidden_states = hidden_states

        return self.longbert_output


class LongBERTModel(nn.Module):
    def __init__(self, config=None):
        super(LongBERTModel, self).__init__()
        self.config = config
        self.embeddings = LongBERTEmbeddings(config) if config is not None else None
        self.encoder = LongBERTEncoder(config) if config is not None else None

    @classmethod
    def from_config(cls, config):
        return cls(config=config)

    @classmethod
    def from_pretrained(cls, ckpt, version="v2"):
        model_ckpt = hf_hub_download(repo_id=ckpt, filename=f"pytorch_model_{version}.bin")
        model_config = LongBERTConfig.from_pretrained(ckpt)
        model = cls(config=model_config)
        model.load_state_dict(torch.load(model_ckpt, map_location="cpu"))
        return model

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, output_hidden_states=False):
        batch_size, seq_len = input_ids.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)

        hidden_state = self.embeddings(input_ids, token_type_ids, position_ids)
        return self.encoder(hidden_state, attention_mask=attention_mask, output_hidden_states=output_hidden_states)


class CisDNATrans(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.vocab_size = 150000
        config.max_position_embeddings = 71680
        config.intermediate_size = 3072
        config.num_hidden_layers = 6
        config.segment_size = [128, 512, 1024, 2048]
        config.dilated_rate = [16, 64, 256, 512]

        self.bert = LongBERTModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def init_weights(self):
        for module_ in self.named_modules():
            if isinstance(module_[1], (torch.nn.Linear, torch.nn.Embedding)):
                module_[1].weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module_[1], torch.nn.LayerNorm):
                module_[1].bias.data.zero_()
                module_[1].weight.data.fill_(1.0)
            if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
                module_[1].bias.data.zero_()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state

        if masked_lm_labels is not None:
            mask = masked_lm_labels != -100
            selected_prediction_scores = self.cls(sequence_output[mask])
            selected_labels = masked_lm_labels[mask]
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(selected_prediction_scores, selected_labels)
            outputs = (masked_lm_loss,)

        return outputs


class RegFM(BertPreTrainedModel):
    def __init__(self, epi_config):
        super().__init__(epi_config)
        config = deepcopy(epi_config)
        num_cross_attentions = 4
        config.vocab_size = 2108
        config.max_position_embeddings = 2112

        dna_config = deepcopy(epi_config)
        dna_config.vocab_size = 150000
        dna_config.max_position_embeddings = 71680
        dna_config.intermediate_size = 3072
        dna_config.num_hidden_layers = 6
        dna_config.segment_size = [128, 512, 1024, 2048]
        dna_config.dilated_rate = [16, 64, 256, 512]

        config.attention_mode = "sparse"
        epi_config.attention_mode = "sparse"

        self.tf_bert = TransContextModel(config, epi_config)
        self.dna_bert = LongBERTModel(dna_config)
        self.cross_attentions = nn.ModuleList(
            [CrossAttention(dna_config.hidden_size, 2) for _ in range(num_cross_attentions)]
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.predictor = nn.Linear(config.hidden_size, 1)
        self.relu = nn.LeakyReLU(negative_slope=0.01)

        self.init_weights()

    def init_weights(self):
        for module_ in self.named_modules():
            if isinstance(module_[1], (torch.nn.Linear, torch.nn.Embedding)):
                module_[1].weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module_[1], torch.nn.LayerNorm):
                module_[1].bias.data.zero_()
                module_[1].weight.data.fill_(1.0)
            if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
                module_[1].bias.data.zero_()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        trans_ids=None,
        dna_ids=None,
        attention_mask=None,
        dna_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs_tf = self.tf_bert(
            input_ids=input_ids,
            epi_ids=trans_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        outputs = self.dna_bert(
            dna_ids,
            attention_mask=dna_attention_mask,
            token_type_ids=token_type_ids,
        )

        dna_token_output = self.dropout(outputs.last_hidden_state)
        tf_token_output = self.dropout2(outputs_tf[0])

        out_attn = None
        for cross_attention in self.cross_attentions:
            query = dna_token_output.permute(1, 0, 2)
            key = tf_token_output.permute(1, 0, 2)
            value = tf_token_output.permute(1, 0, 2)

            mapped_output, attn_weights = cross_attention(query, key, value)
            if out_attn is None:
                out_attn = attn_weights
            else:
                out_attn += attn_weights

            dna_token_output = mapped_output.permute(1, 0, 2)

        cls_token_hidden_state = dna_token_output[:, 0, :]
        logits = self.relu(self.predictor(cls_token_hidden_state))
        outputs = (logits, out_attn, outputs.last_hidden_state[:, :30, :], cls_token_hidden_state)

        if labels is not None:
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
