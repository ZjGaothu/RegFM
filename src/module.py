# coding=utf-8
"""TF encoder components used by RegFM."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_bert import *
from transformers.modeling_bert import GenomicBertModelNew as TransContextModel


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, attn_weights = self.attention(query, key, value, key_padding_mask=key_padding_mask)

        query = query + self.dropout(attn_output)
        query = self.norm1(query)

        ff_output = self.linear2(self.dropout(self.activation(self.linear1(query))))

        output = query + self.dropout(ff_output)
        output = self.norm2(output)

        return output, attn_weights


class TransContextForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        tf_config = deepcopy(config)
        tf_config.vocab_size = 2108
        tf_config.max_position_embeddings = 2112
        self.bert = TransContextModel(tf_config, config)
        self.cls = BertOnlyMLMHead(config)
        self.vocab_size = config.vocab_size
        self.config = config
        self.tf_config = tf_config
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

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        trans_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
        l2_lambda=0.01,
    ):
        outputs = self.bert(
            input_ids,
            trans_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        outputs = (prediction_scores, sequence_output)

        if masked_lm_labels is not None:
            class_counts = torch.bincount(masked_lm_labels[masked_lm_labels != -100], minlength=self.vocab_size)
            class_weights = 1.0 / (class_counts.float() + 1e-6)
            loss_fct = CrossEntropyLoss(weight=class_weights)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
            _, predictions = torch.max(prediction_scores, dim=-1)
            masked_indices = masked_lm_labels != -100
            masked_predictions = predictions[masked_indices]
            masked_labels = masked_lm_labels[masked_indices]
            accuracy = (masked_predictions == masked_labels).float().mean().item()
            outputs = (masked_lm_loss, accuracy) + outputs

        if lm_labels is not None:
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        return outputs
