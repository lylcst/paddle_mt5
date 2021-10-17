
import copy
import math
import os
import warnings
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .. import PretrainedModel, register_base_model

__all__ = [
    "MT5Model",
    "MT5EncoderModel",
    "MT5ForConditionalGeneration",
    "T5Config",
    "T5LayerNorm",
    "T5Attention",
    "T5DenseReluDense",
    "T5LayerSelfAttention",
    "T5LayerCrossAttention",
    "T5Block",
    "T5Model",
    "MT5Model",
    "T5EncoderModel",
    "MT5EncoderModel"
]

def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(
            activation_string, list(ACT2FN.keys())))


def mish(x):
    return x * F.tanh(F.softplus(x))


def linear_act(x):
    return x


def swish(x):
    return x * F.sigmoid(x)


def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + paddle.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * paddle.pow(x, 3.0))))


ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "gelu_new": gelu_new,
    "tanh": F.tanh,
    "sigmoid": F.sigmoid,
    "mish": mish,
    "linear": linear_act,
    "swish": swish,
}


class T5Config():

    model_type = "t5"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    def __init__(
        self,
        vocab_size=32128,
        d_model=512,
        d_kv=64,
        d_ff=2048,
        num_layers=6,
        num_decoder_layers=None,
        num_heads=8,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache

class T5LayerNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = paddle.static.create_parameter([hidden_size], dtype=paddle.float32, default_initializer=paddle.nn.initializer.Constant(value=1.0))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.astype(paddle.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)

        # convert into float16 if necessary
        if self.weight.dtype == paddle.float16:
            hidden_states = hidden_states.astype(paddle.float16)
        return hidden_states*self.weight


class T5DenseReluDense(nn.Layer):
    def __init__(self, d_model=512,
                       d_ff=2048, 
                       dropout_rate=0.1):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias_attr=None)
        self.wo = nn.Linear(d_ff, d_model, bias_attr=None)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = nn.functional.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedGeluDense(nn.Layer):
    def __init__(self, d_model=512, 
                        d_ff=2048,
                        dropout_rate=0.1):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias_attr=None)
        self.wi_1 = nn.Linear(d_model, d_ff, bias_attr=None)
        self.wo = nn.Linear(d_ff, d_model, bias_attr=None)
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu_act = ACT2FN['gelu_new']

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Layer):
    def __init__(self, d_model=512,
                        d_ff=2048, 
                        dropout_rate=0.1,
                        feed_forward_proj="relu",
                        layer_norm_epsilon=1e-6):
        super().__init__()
        if feed_forward_proj == "relu":
            self.DenseReluDense = T5DenseReluDense(d_model, d_ff, dropout_rate)
        elif config.feed_forward_proj == "gated-gelu":
            self.DenseReluDense = T5DenseGatedGeluDense(d_model, d_ff, dropout_rate)
        else:
            raise ValueError(
                f"{selffeed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`"
            )

        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Layer):
    def __init__(self,  d_model=512, 
                        d_ff=2048, 
                        d_kv=64,
                        num_heads=8,
                        dropout_rate=0.1,
                        feed_forward_proj="relu", 
                        is_decoder=False,
                        relative_attention_num_buckets=32,
                        layer_norm_epsilon=1e-6, 
                        has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.d_model = d_model
        self.key_value_proj_dim = d_kv
        self.n_heads = num_heads
        self.dropout = dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias_attr=None)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias_attr=None)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias_attr=None)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias_attr=None)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.gradient_checkpointing = False


    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):

        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).astype(paddle.int64) * num_buckets
            relative_position = paddle.abs(relative_position)
        else:
            relative_position = -paddle.minimum(relative_position, paddle.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            paddle.log(relative_position.astype(paddle.float32) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(paddle.int64)
        relative_postion_if_large = paddle.minimum(
            relative_postion_if_large, paddle.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += paddle.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = paddle.arange(
            query_length, dtype=paddle.int64
        ).reshape([-1,1])
        memory_position = paddle.arange(
            key_length, dtype=paddle.int64
        ).unsqueeze(0)
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.transpose([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        output_attentions=False,
    ):
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.reshape((batch_size, -1, self.n_heads, self.key_value_proj_dim)).transpose([0, 2, 1, 3])

        def unshape(states):
            """reshape"""
            return states.transpose([0, 2, 1, 3]).reshape((batch_size, -1, self.inner_dim))

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = paddle.concat([past_key_value, hidden_states], axis=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        shape_ = paddle.arange(len(key_states.shape)).numpy().tolist()
        shape_[3] = 2
        shape_[2] = 3
        scores = paddle.matmul(
            query_states, key_states.transpose(shape_)

        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = paddle.zeros(
                    (1, self.n_heads, real_seq_length, key_length),dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.stop_gradient = False
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        attn_weights = nn.functional.softmax(scores.astype(paddle.float32), axis=-1).astype(
            scores.dtype
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(paddle.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        # present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        # outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)
        outputs = (attn_output,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5LayerSelfAttention(nn.Layer):
    def __init__(self,  d_model=512, 
                        d_ff=2048, 
                        d_kv=64,
                        num_heads=8,
                        dropout_rate=0.1,
                        feed_forward_proj="relu", 
                        is_decoder=False,
                        relative_attention_num_buckets=32,
                        layer_norm_epsilon=1e-6, 
                        has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(d_model,
                                        d_ff, d_kv,
                                        num_heads,
                                        dropout_rate, 
                                        feed_forward_proj, 
                                        is_decoder,
                                        relative_attention_num_buckets,
                                        layer_norm_epsilon,
                                        has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Layer):
    def __init__(self,  d_model=512, 
                        d_ff=2048, 
                        d_kv=64,
                        num_heads=8,
                        dropout_rate=0.1,
                        feed_forward_proj="relu", 
                        is_decoder=False,
                        relative_attention_num_buckets=32,
                        layer_norm_epsilon=1e-6, 
                        has_relative_attention_bias=False):
        super().__init__()
        self.EncDecAttention = T5Attention(d_model,
                                d_ff, d_kv,
                                num_heads,
                                dropout_rate, 
                                feed_forward_proj, 
                                is_decoder,
                                relative_attention_num_buckets,
                                layer_norm_epsilon,
                                has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Layer):
    def __init__(self,  d_model=512, 
                        d_ff=2048, 
                        d_kv=64,
                        num_heads=8,
                        dropout_rate=0.1,
                        feed_forward_proj="relu", 
                        is_decoder=False,
                        relative_attention_num_buckets=32,
                        layer_norm_epsilon=1e-6, 
                        has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = is_decoder
        self.layer = nn.LayerList()
        self.layer.append(T5LayerSelfAttention(d_model, 
                                               d_ff, 
                                               d_kv,
                                               num_heads,
                                               dropout_rate,
                                               feed_forward_proj, 
                                               is_decoder,
                                               relative_attention_num_buckets,
                                               layer_norm_epsilon, 
                                               has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(d_model, 
                                                    d_ff,
                                                    d_kv,
                                                    num_heads,
                                                    dropout_rate,
                                                    feed_forward_proj, 
                                                    is_decoder,
                                                    relative_attention_num_buckets,
                                                    layer_norm_epsilon, 
                                                    has_relative_attention_bias=has_relative_attention_bias))

        self.layer.append(T5LayerFF(d_model, d_ff, dropout_rate, feed_forward_proj ,layer_norm_epsilon))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        past_key_value = None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        output_attentions=None,
        layer_head_mask=None,
    ):

        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=None,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == paddle.float16 and paddle.isinf(hidden_states).any():
            clamp_value = np.finfo(hidden_states.numpy().dtype).max - 1000
            hidden_states = paddle.clip(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=None,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == paddle.float16 and paddle.isinf(hidden_states).any():
                clamp_value = np.finfo(hidden_states.numpy().dtype).max - 1000
                hidden_states = paddle.clip(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == paddle.float16 and paddle.isinf(hidden_states).any():
            clamp_value = np.finfo(hidden_states.numpy().dtype).max - 1000
            hidden_states = paddle.clip(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)


        outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class T5PreTrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    base_model_prefix = "transformer"
    model_config_file = "model_config.json"
    
    pretrained_init_configuration = {
        "t5-small": {
            "d_model": 512,
            "d_ff": 2048, 
            "d_kv": 64,
            "num_heads": 8,
            "num_layers": 6,
            "dropout_rate": 0.1,
            "feed_forward_proj": "relu", 
            "is_decoder": False,
            "relative_attention_num_buckets": 32,
            "layer_norm_epsilon": 1e-6, 
            "initializer_factor": 1.0,
            "has_relative_attention_bias": False
            "vocab_size": 32128
        },
        "t5-base": {
            "d_model": 512,
            "d_ff": 2048, 
            "d_kv": 64,
            "num_heads": 8,
            "num_layers": 6,
            "dropout_rate": 0.1,
            "feed_forward_proj": "relu", 
            "is_decoder": False,
            "relative_attention_num_buckets": 32,
            "layer_norm_epsilon": 1e-6, 
            "initializer_factor": 1.0,
            "has_relative_attention_bias": False
            "vocab_size": 32128
        },
        "t5-large": {
            "d_model": 512,
            "d_ff": 2048, 
            "d_kv": 64,
            "num_heads": 8,
            "num_layers": 6,
            "dropout_rate": 0.1,
            "feed_forward_proj": "relu", 
            "is_decoder": False,
            "relative_attention_num_buckets": 32,
            "layer_norm_epsilon": 1e-6, 
            "initializer_factor": 1.0,
            "has_relative_attention_bias": False
            "vocab_size": 32128
        },
        "t5-3b": {
            "d_model": 512,
            "d_ff": 2048, 
            "d_kv": 64,
            "num_heads": 8,
            "num_layers": 6,
            "dropout_rate": 0.1,
            "feed_forward_proj": "relu", 
            "is_decoder": False,
            "relative_attention_num_buckets": 32,
            "layer_norm_epsilon": 1e-6, 
            "initializer_factor": 1.0,
            "has_relative_attention_bias": False
            "vocab_size": 32128
        },
        "t5-11b": {
            "d_model": 512,
            "d_ff": 2048, 
            "d_kv": 64,
            "num_heads": 8,
            "num_layers": 6,
            "dropout_rate": 0.1,
            "feed_forward_proj": "relu", 
            "is_decoder": False,
            "relative_attention_num_buckets": 32,
            "layer_norm_epsilon": 1e-6, 
            "initializer_factor": 1.0,
            "has_relative_attention_bias": False
            "vocab_size": 32128
        },

    }

    resource_files_names = {"model_state": "model_state.pdparams"}
    
    pretrained_resource_files_map = {
    "model_state": {
        "mt5-small":
            "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-base-v1.pdparams",
        "mt5-base":
            "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-large-v1.pdparams",
        "mt5-large":
            "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-xlarge-v1.pdparams",
        "mt5-3b":
            "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-xxlarge-v1.pdparams",
        "mt5-11b":
            "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-base-v2.pdparams",
        }
    }
    base_model_prefix = "transformer"

    @property
    def dummy_inputs(self):
        input_ids = paddle.to_tensor(DUMMY_INPUTS)
        input_mask = paddle.to_tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, layer):
        """Initialize the weights"""
        factor = self.initializer_factor  # Used for testing weights initialization
        if isinstance(factor, tuple):
            factor = factor[0]
        if isinstance(layer, T5LayerNorm):
            # module.weight.data.fill_(factor * 1.0)
            layer.weight.set_value(
                paddle.ones_like(layer.weight)*(factor * 1.0)
            )
        elif isinstance(layer, (T5Model, T5ForConditionalGeneration, T5EncoderModel)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            # module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            layer.shared.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=factor * 1.0,
                    shape=layer.shared.weight.shape
                )
            )
        elif isinstance(layer, T5DenseReluDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            # module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            layer.wi.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=factor * ((self.d_model) ** -0.5),
                    shape=layer.wi.weight.shape
                )
            )

            if hasattr(layer.wi, "bias") and layer.wi.bias is not None:
                # module.wi.bias.data.zero_()
                layer.wi.bias.set_value(
                    paddle.zeros_like(layer.wi.bias)
                )
            # module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            layer.wo.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=factor * ((self.d_ff) ** -0.5),
                    shape=layer.wo.weight.shape
                )
            )
            if hasattr(layer.wo, "bias") and layer.wo.bias is not None:
                # layer.wo.bias.data.zero_()
                layer.wo.bias.set_value(
                    paddle.zeros_like(layer.wo.bias)
                )
        elif isinstance(layer, T5DenseGatedGeluDense):
            # module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            layer.wi_0.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=factor * ((self.d_model) ** -0.5),
                    shape=layer.wi_0.weight.shape
                )
            )
            if hasattr(layer.wi_0, "bias") and layer.wi_0.bias is not None:
                # layer.wi_0.bias.data.zero_()
                layer.wi_0.bias.set_value(
                    paddle.zeros_like(layer.wi_0.bias)
                )
            # module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            layer.wi_1.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=factor * ((self.d_model) ** -0.5),
                    shape=layer.wi_1.weight.shape
                )
            )
            if hasattr(layer.wi_1, "bias") and layer.wi_1.bias is not None:
                # module.wi_1.bias.data.zero_()
                layer.wi_1.bias.set_value(
                    paddle.zeros_like(layer.wi_1.bias)
                )
            # module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            layer.wi_o.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=factor * ((self.d_ff) ** -0.5),
                    shape=layer.wi_o.weight.shape
                )
            )
            if hasattr(layer.wo, "bias") and layer.wo.bias is not None:
                # module.wo.bias.data.zero_()
                layer.wo.bias.set_value(
                    paddle.zeros_like(layer.wo.bias)
                )
        elif isinstance(layer, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.d_model
            key_value_proj_dim = self.d_kv
            n_heads = self.num_heads
            # module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            # module.k.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            # module.v.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            # module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))

            layer.q.weight.set_value(paddle.tensor.normal(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5), shape=layer.q.weight.shape))
            layer.k.weight.set_value(paddle.tensor.normal(mean=0.0, std=factor * (d_model ** -0.5), shape=layer.k.weight.shape))
            layer.v.weight.set_value(paddle.tensor.normal(mean=0.0, std=factor * (d_model ** -0.5), shape=layer.v.weight.shape))
            # layer.o.weight.set_value(paddle.tensor.normal(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5), shape=layer.o.weight.shape))


            if layer.has_relative_attention_bias:
                # layer.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))
                layer.relative_attention_bias.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=factor * ((d_model) ** -0.5),
                        shape=layer.relative_attention_bias.weight.shape
                    )
                )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (T5Attention, T5Stack)):
            module.gradient_checkpointing = value

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.cdecoder_start_token_id
        pad_token_id = self.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = paddle.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = paddle.concat([shifted_input_ids, input_ids[..., :-1]], axis=-1)
        else:
            shifted_input_ids = paddle.zeros(input_ids.shape, dtype=input_ids.dtype, place=input_ids.place)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids = paddle.where(shifted_input_ids == -100, paddle.ones(shifted_input_ids.shape)*pad_token_id, shifted_input_ids)

        assert paddle.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


class T5Stack(T5PreTrainedModel):
    def __init__(self, d_model=512, 
                        d_ff=2048, 
                        d_kv=64,
                        num_heads=8,
                        num_layers=6,
                        dropout_rate=0.1,
                        feed_forward_proj="relu", 
                        is_decoder=False,
                        is_encoder_decoder=False,
                        relative_attention_num_buckets=32,
                        layer_norm_epsilon=1e-6, 
                        initializer_factor=1.0,
                        has_relative_attention_bias=False,
                        embed_tokens=None,
                        ):
        super().__init__()
        # super().__init__(d_model=d_model, 
        #                 d_ff=d_ff, 
        #                 d_kv=d_kv,
        #                 num_heads=num_heads,
        #                 num_layers=num_layers,
        #                 dropout_rate=dropout_rate,
        #                 feed_forward_proj=feed_forward_proj, 
        #                 is_decoder=is_decoder,
        #                 relative_attention_num_buckets=relative_attention_num_buckets,
        #                 layer_norm_epsilon=layer_norm_epsilon, 
        #                 has_relative_attention_bias=has_relative_attention_bias)
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_heads = num_heads,
        self.embed_tokens = embed_tokens
        self.is_decoder = is_decoder
        self.num_layers = num_layers
        self.initializer_factor = initializer_factor

        self.block = nn.LayerList(
            [T5Block(d_model=d_model,
                    d_ff=d_ff, 
                    d_kv=d_kv,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    feed_forward_proj=feed_forward_proj, 
                    is_decoder=is_decoder,
                    relative_attention_num_buckets=relative_attention_num_buckets,
                    layer_norm_epsilon=layer_norm_epsilon,  
                    has_relative_attention_bias=bool(i == 0)) for i in range(num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

        self.apply(self._init_weights)


    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings
    
    def invert_attention_mask(self, encoder_attention_mask):

        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask.unsqueeze(1)
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(2)

        encoder_extended_attention_mask = encoder_extended_attention_mask.astype(dtype=paddle.float32)  # fp16 compatibility


        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9

        return encoder_extended_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_hidden_states=None,
        output_attentions=None
    ):

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        # extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds)

        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = paddle.ones([batch_size, seq_length])

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = paddle.ones(
                [batch_size, encoder_seq_length], dtype=paddle.int64
            )

        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = paddle.ones(encoder_hidden_shape)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None



        all_hidden_states = () if output_hidden_states is not None else None
        hidden_states = self.dropout(inputs_embeds)

        for i, layer_module in enumerate(self.block):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=None,
                position_bias=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=None,
                layer_head_mask=None,
                # cross_attn_layer_head_mask=None,
                # past_key_value=past_key_value,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            # position_bias = layer_outputs[2]
            # if self.is_decoder and encoder_hidden_states is not None:
            #     encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

            # if output_attentions:
            #     all_attentions = all_attentions + (layer_outputs[3],)
            #     if self.is_decoder:
            #         all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            return all_hidden_states
        return (hidden_states,)
        

class T5Model(T5PreTrainedModel):

    def __init__(self, vocab_size=32128,
                        d_model=512,
                        d_ff=2048, 
                        d_kv=64,
                        num_heads=8,
                        num_layers=6,
                        dropout_rate=0.1,
                        feed_forward_proj="relu", 
                        is_decoder=False,
                        relative_attention_num_buckets=32,
                        layer_norm_epsilon=1e-6, 
                        initializer_factor=1.0,
                        has_relative_attention_bias=False):
        super().__init__()


        self.initializer_factor = initializer_factor
        self.shared = nn.Embedding(vocab_size, d_model)

        # encoder_config = copy.deepcopy(config)
        # encoder_config.is_decoder = False
        # encoder_config.is_encoder_decoder = False
        # self.encoder = T5Stack(encoder_config, self.shared)

        self.encoder = T5Stack(d_model=d_model, 
                                d_ff=d_ff, 
                                d_kv=d_kv,
                                num_heads=num_heads,
                                num_layers=num_layers,
                                dropout_rate=dropout_rate,
                                feed_forward_proj=feed_forward_proj, 
                                is_decoder=False,
                                is_encoder_decoder=False,
                                relative_attention_num_buckets=relative_attention_num_buckets,
                                layer_norm_epsilon=layer_norm_epsilon, 
                                initializer_factor=initializer_factor,
                                has_relative_attention_bias=has_relative_attention_bias,
                                embed_tokens=self.shared
                                )

        # decoder_config = copy.deepcopy(config)
        # decoder_config.is_decoder = True
        # decoder_config.is_encoder_decoder = False
        # decoder_config.num_layers = config.num_decoder_layers
        # self.decoder = T5Stack(decoder_config, self.shared)

        self.decoder = T5Stack(d_model=d_model, 
                                d_ff=d_ff, 
                                d_kv=d_kv,
                                num_heads=num_heads,
                                num_layers=num_layers,
                                dropout_rate=dropout_rate,
                                feed_forward_proj=feed_forward_proj, 
                                is_decoder=True,
                                is_encoder_decoder=False,
                                relative_attention_num_buckets=relative_attention_num_buckets,
                                layer_norm_epsilon=layer_norm_epsilon, 
                                initializer_factor=initializer_factor,
                                has_relative_attention_bias=has_relative_attention_bias,
                                embed_tokens=self.shared
                                )

        # self.apply(self._init_weights)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        # past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        # return_dict=None,
    ):

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                # return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            # past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            # cross_attn_head_mask=cross_attn_head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return decoder_outputs + encoder_outputs

class T5ForConditionalGeneration(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias_attr=None)

        self.init_weights()


    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # if self.model_parallel:
        #     paddle.device.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.reshape([-1, lm_logits.shape[-1]]), labels.reshape([1,-1]).squeeze(0))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

class T5EncoderModel(T5PreTrainedModel):

    def __init__(self, vocab_size=32128,
                        d_model=512, 
                        d_ff=2048, 
                        d_kv=64,
                        num_heads=8,
                        num_layers=6,
                        dropout_rate=0.1,
                        feed_forward_proj="relu", 
                        is_decoder=False,
                        relative_attention_num_buckets=32,
                        layer_norm_epsilon=1e-6, 
                        initializer_factor=1.0,
                        has_relative_attention_bias=False,
                        embed_tokens=None):
        super().__init__()
        self.initializer_factor = initializer_factor,
        self.d_model = d_model
        self.d_ff = d_ff 
        self.d_kv = d_kv
        self.num_heads = num_heads
        self.shared = nn.Embedding(vocab_size, d_model)
        self.encoder = T5Stack(d_model=d_model, 
                                d_ff=d_ff, 
                                d_kv=d_kv,
                                num_heads=num_heads,
                                num_layers=num_layers,
                                dropout_rate=dropout_rate,
                                feed_forward_proj="relu", 
                                is_decoder=is_decoder,
                                relative_attention_num_buckets=relative_attention_num_buckets,
                                layer_norm_epsilon=layer_norm_epsilon, 
                                has_relative_attention_bias=has_relative_attention_bias,
                                is_encoder_decoder=False,
                                embed_tokens=self.shared)

        self.apply(self._init_weights)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )

        return encoder_outputs


class MT5Model(T5Model):
    r"""
    This class overrides :class:`~transformers.T5Model`. Please check the superclass for the appropriate documentation
    alongside usage examples.
    Examples::
        >>> from transformers import MT5Model, T5Tokenizer
        >>> model = MT5Model.from_pretrained("google/mt5-small")
        >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
        >>> summary = "Weiter Verhandlung in Syrien."
        >>> inputs = tokenizer(article, return_tensors="pt")
        >>> with tokenizer.as_target_tokenizer():
        ...     labels = tokenizer(summary, return_tensors="pt")
        >>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
        >>> hidden_states = outputs.last_hidden_state
    """
    model_type = "mt5"

class MT5ForConditionalGeneration(T5ForConditionalGeneration):
    model_type = "mt5"


class MT5EncoderModel(T5EncoderModel):
    r"""
    This class overrides :class:`~transformers.T5EncoderModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    Examples::
        >>> from transformers import MT5EncoderModel, T5Tokenizer
        >>> model = MT5EncoderModel.from_pretrained("google/mt5-small")
        >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
        >>> input_ids = tokenizer(article, return_tensors="pt").input_ids
        >>> outputs = model(input_ids)
        >>> hidden_state = outputs.last_hidden_state
    """

    model_type = "mt5"
