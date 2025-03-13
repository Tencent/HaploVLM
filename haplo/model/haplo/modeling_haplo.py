# Copyright (c) Lin Song. All rights reserved.
import math
import numpy as np
from typing import Optional, Tuple, Callable, Union

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils import logging
from transformers.image_processing_utils import select_best_resolution
from transformers.modeling_utils import (
    ALL_ATTENTION_FUNCTIONS, PreTrainedModel)
from transformers.cache_utils import (
    Cache, DynamicCache, SlidingWindowCache, StaticCache)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)

from .configuration_haplo import HaploConfig


logger = logging.get_logger(__name__)


def unpad_image(tensor, original_size):
    if not isinstance(original_size, (list, tuple)):
        if not isinstance(original_size, (torch.Tensor, np.ndarray)):
            raise TypeError(
                f'image_size invalid type: {type(original_size)} not valid, '
                'should be either list, tuple, np.ndarray or tensor'
            )
        original_size = original_size.tolist()
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(round(original_height * scale_factor, 7))
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(round(original_width * scale_factor, 7))
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    if not isinstance(grid_pinpoints, list):
        raise TypeError('grid_pinpoints should be a list of tuples or lists')

    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(
                f'image_size invalid type: {type(image_size)} not valid, '
                'should be either list, tuple, np.ndarray or tensor'
            )
        image_size = image_size.tolist()

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


def image_size_to_num_patches(image_size, grid_pinpoints, patch_size: int):
    if not isinstance(grid_pinpoints, list):
        raise TypeError('grid_pinpoints should be a list of tuples or lists')

    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(f'image_size invalid type {type(image_size)} '
                            f'with value {image_size}')
        image_size = image_size.tolist()

    best_resolution = select_best_resolution(image_size, grid_pinpoints)
    height, width = best_resolution
    num_patches = 0
    # consider change to ceil(height/patch_size)*ceil(width/patch_size) + 1
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            num_patches += 1
    # add the base patch
    # num_patches += 1
    return num_patches


class HaploMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    '''Rotates half the hidden dims of the input.'''
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    '''Applies Rotary Position Embedding to the query and key tensors.'''
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(
        batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class HaploAttention(nn.Module):
    '''Multi-headed attention from 'Attention Is All You Need' paper'''

    def __init__(self, config: HaploConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config,
            'head_dim',
            config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim)
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=True)
        if config.attention_rope:
            self.rotary_emb = HaploRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(
            hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            hidden_shape).transpose(1, 2)

        if self.config.attention_rope:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models;
            # cache_position needed for the static cache
            cache_kwargs = {
                'sin': sin, 'cos': cos, 'cache_position': cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, 'sliding_window', None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != 'eager':
            if (self.config._attn_implementation == 'sdpa' and
                    kwargs.get('output_attentions', False)):
                logger.warning_once(
                    '`torch.nn.functional.scaled_dot_product_attention` does '
                    'not support `output_attentions=True`. Falling back to '
                    'eager attention. This warning can be removed using the '
                    'argument `attn_implementation="eager"` when loading '
                    'the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class HaploNorm(nn.Module):
    def __init__(self, hidden_size, mode='rmsnorm', eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        if mode == 'layernorm':
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            assert mode == 'rmsnorm', (
                'Only `layernorm` and `rmsnorm` are supported.')
        self.hidden_size = hidden_size
        self.mode = mode
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        # hidden_states = hidden_states.to(torch.float32)
        if self.mode == 'rmsnorm':
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)
        elif self.mode == 'layernorm':
            normalized_shape = (self.hidden_size,)
            return nn.functional.layer_norm(
                hidden_states, normalized_shape, self.weight, self.bias,
                eps=self.variance_epsilon).to(input_dtype)
        else:
            raise ValueError('Invalid mode for HaploNorm')

    def extra_repr(self):
        return f'{tuple(self.weight.shape)}, eps={self.variance_epsilon}'


class HaploDecoderLayer(nn.Module):
    def __init__(self, config: HaploConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = HaploAttention(config=config, layer_idx=layer_idx)
        self.mlp = HaploMLP(config)
        self.input_layernorm = HaploNorm(
            config.hidden_size, mode=config.norm_mode, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HaploNorm(
            config.hidden_size, mode=config.norm_mode, eps=config.rms_norm_eps)
        if (config.sliding_window and
                config._attn_implementation != 'flash_attention_2'):
            logger.warning_once(
                'Sliding Window Attention is enabled but not implemented '
                f'for `{config._attn_implementation}`; '
                'unexpected results may be encountered.'
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple] = None,
        **kwargs,
    ) -> Tuple:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class HaploRotaryEmbedding(nn.Module):
    def __init__(self, config: HaploConfig, device=None):
        super().__init__()
        # BC: 'rope_type' was originally 'type'
        if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                'rope_type', config.rope_scaling.get('type'))
        else:
            self.rope_type = 'default'
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len)
            self.register_buffer('inv_freq', inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if (seq_len < self.original_max_seq_len and
                self.max_seq_len_cached > self.original_max_seq_len):
            # This .to() is needed if the model has been moved to a device
            # after being initialized (because the buffer is automatically
            # moved, # but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer(
                'inv_freq', self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if 'dynamic' in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32
        device_type = x.device.type
        device_type = (
            device_type if isinstance(device_type, str) and
            device_type != 'mps' else 'cpu')
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @
                     position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling
        # factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class PatchEmbed(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.proj = nn.Conv2d(
            3,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class HaploPreTrainedModel(PreTrainedModel):
    config_class = HaploConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['HaploDecoderLayer']
    _skip_keys_device_placement = ['past_key_values']
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class HaploModel(HaploPreTrainedModel):
    '''
    Transformer decoder consisting of *config.num_hidden_layers* layers.
    Each layer is a [`HaploDecoderLayer`]

    Args:
        config: HaploConfig
    '''

    def __init__(self, config: HaploConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.num_pre_layers = config.pre_config.num_hidden_layers
        self.layers = nn.ModuleList(
            [HaploDecoderLayer(config.pre_config, layer_idx)
                for layer_idx in range(config.pre_config.num_hidden_layers)] +
            [HaploDecoderLayer(config, layer_idx + self.num_pre_layers)
                for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = HaploNorm(
            config.hidden_size,
            mode=config.norm_mode,
            eps=config.rms_norm_eps
        )
        self.rotary_emb = HaploRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # pre-decoder
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.pre_config.hidden_size, self.padding_idx)
        self.patch_embed = PatchEmbed(config.pre_config)
        num_patches = (config.pre_config.default_image_size //
                       config.pre_config.patch_size) ** 2
        self.pre_pos_embed = nn.Parameter(
            torch.randn(1, num_patches, config.pre_config.hidden_size) * 0.02)
        self.image_newline = nn.Parameter(
            torch.zeros(config.pre_config.hidden_size))
        self.pre_norm = nn.LayerNorm(config.pre_config.hidden_size)
        self.pre_connector = nn.Linear(
            config.pre_config.hidden_size, config.hidden_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_image_embed(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
    ) -> torch.Tensor:
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.default_image_size,
            )
            for imsize in image_sizes
        ]
        if pixel_values.dim() == 5:
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch
                                  in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            raise ValueError(f'pixel_values of shape {pixel_values.shape}, '
                             'expect to be of 4 or 5 dimensions')

        image_embeds = self.patch_embed(pixel_values)
        image_embeds = image_embeds + self.pre_pos_embed
        image_embeds = torch.split(image_embeds, image_num_patches, dim=0)
        return image_embeds

    def get_video_embed(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, frames, channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(
            batch_size * frames, channels, height, width)
        video_embeds = self.patch_embed(pixel_values)
        video_embeds = video_embeds + self.pre_pos_embed
        video_embeds = video_embeds.reshape(
            batch_size, frames * video_embeds.shape[1], -1)
        return video_embeds

    def pack_image_embeds(self, image_embeds, image_sizes):
        image_newline = self.image_newline
        vision_aspect_ratio = self.config.image_aspect_ratio
        new_image_embeds = []
        embed_lens = []
        for image_idx, image_embed in enumerate(image_embeds):
            
            if image_embed.shape[0] >= 1 and 'any' in self.config.image_aspect_ratio:
                height = width = (self.config.default_image_size //
                                  self.config.patch_size)
                if height * width != image_embed.shape[1]:
                    raise ValueError('The number of patches is not consistent '
                                     'with the image size.')
                num_patch_height, num_patch_width = (
                    get_anyres_image_grid_shape(
                        image_sizes[image_idx],
                        self.config.image_grid_pinpoints,
                        self.config.default_image_size
                    )
                )
                image_embed = image_embed.view(
                    num_patch_height, num_patch_width, height, width, -1)
                image_embed = image_embed.permute(4, 0, 2, 1, 3).contiguous()
                image_embed = image_embed.flatten(1, 2).flatten(2, 3)
                image_embed = unpad_image(image_embed, image_sizes[image_idx])
                max_num_patches = int(vision_aspect_ratio.strip('anyres_max_'))
                channels, curr_height, curr_width = image_embed.shape
                ratio = math.sqrt(
                    curr_height * curr_width / (max_num_patches * height**2))
                if ratio > 1.1:
                    image_embed = image_embed[None]
                    dtype = image_embed.dtype
                    image_embed = nn.functional.interpolate(
                        image_embed.to(torch.float32),
                        [int(curr_height // ratio), int(curr_width // ratio)],
                        mode='bilinear'
                    )[0].to(dtype)
                if image_newline is not None:
                    image_embed = torch.cat(
                        (
                            image_embed,
                            image_newline[:, None, None]
                            .expand(*image_embed.shape[:-1], 1)
                            .to(image_embed.device, image_embed.dtype),
                        ),
                        dim=-1,
                    )
                image_embed = image_embed.flatten(1, 2).transpose(0, 1)
            else:
                image_embed = image_embed[0]
                if image_newline is not None:
                    image_embed = torch.cat(
                        (image_embed, image_newline[None].to(image_embed)),
                        dim=0)
            new_image_embeds.append(image_embed)
            embed_lens.append(image_embed.size(0))
        image_embeds = torch.cat(new_image_embeds, dim=0)
        embed_lens = torch.tensor(
            embed_lens, dtype=torch.long, device=image_embeds.device)
        return image_embeds, embed_lens

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.Tensor = None,
        image_sizes: torch.Tensor = None,
        pixel_values_videos: torch.FloatTensor = None,
        image_sizes_videos: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_start_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (output_attentions if output_attentions is not None
                             else self.config.output_attentions)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = (use_cache if use_cache is not None
                     else self.config.use_cache)
        return_dict = (return_dict if return_dict is not None
                       else self.config.use_return_dict)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                'You must specify exactly one of input_ids or inputs_embeds')

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                '`use_cache=True` is incompatible with gradient '
                'checkpointing. Setting `use_cache=False`.'
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_embed(pixel_values, image_sizes)
            image_embeds, embed_lens = self.pack_image_embeds(
                image_embeds, image_sizes)
            special_image_mask = (
                (input_ids == self.config.image_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            assert (input_ids == self.config.image_token_index).sum() == \
                image_embeds.shape[0], f'image tokens: {(input_ids == self.config.image_token_index).sum()} but image embeddings: {image_embeds.shape[0]}'
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_embed(pixel_values_videos)
            image_newline = (
                self.image_newline[None, None, :].repeat(
                    video_embeds.shape[0], 1, 1).to(
                        video_embeds.device)
            )
            video_embeds = torch.cat((video_embeds, image_newline), dim=1)
            video_embeds = video_embeds.flatten(0, 1)
            special_video_mask = (
                (input_ids == self.config.video_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(
                inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(
                special_video_mask, video_embeds)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (past_key_values.get_seq_length()
                                if past_key_values is not None else 0)
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position,
            past_key_values, output_attentions,
            attention_start_ids
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # pre normalize
        hidden_states = self.pre_norm(hidden_states)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder
        for layer_idx, decoder_layer in enumerate(self.layers):
            if layer_idx == self.num_pre_layers:
                hidden_states = self.pre_connector(hidden_states)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
        attention_start_ids: Optional[torch.Tensor] = None,
    ):
        if self.config._attn_implementation == 'flash_attention_2':
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = (
                    attention_mask[:, -1].sum().item() !=
                    input_tensor.size()[0])
                if is_padding_right:
                    raise ValueError(
                        'You are attempting to perform batched generation '
                        'with padding_side="right" this may lead to '
                        'unexpected behaviour for Flash Attention version of '
                        'Haplo. Make sure to call `tokenizer.padding_side  = '
                        '"left"` before tokenizing the input. '
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument
        # instead of its `attn_mask` argument, inorder to dispatch on Flash
        # Attention 2. This feature is not compatible with static cache,
        # as SDPA will fail to infer the attention mask.
        past_seen_tokens = (past_key_values.get_seq_length()
                            if past_key_values is not None else 0)
        using_static_cache = isinstance(
            past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(
            past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward
        # method calls the eager implementation's forward
        if (
            self.config._attn_implementation == 'sdpa'
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D,
        # we generate a causal mask here (4D).
        causal_mask = \
            self._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=target_length,
                dtype=dtype,
                device=device,
                cache_position=cache_position,
                batch_size=input_tensor.shape[0],
                config=self.config,
                past_key_values=past_key_values,
                attention_start_ids=attention_start_ids)

        if (
            self.config._attn_implementation == 'sdpa'
            and attention_mask is not None
            and attention_mask.device.type in ['cuda', 'xpu']
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask,
            # for example the relevant first rows when using left padding.
            # This is required by F.scaled_dot_product_attention
            # memory-efficient attention path. Details:
            # https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: HaploConfig,
        past_key_values: Cache,
        attention_start_ids: torch.Tensor,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted
            # form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device
            )
            if attention_start_ids is not None:
                causal_mask = causal_mask[None, None, :, :].expand(
                    batch_size, 1, -1, -1)
                assert attention_start_ids.shape[-1] <= target_length
                if target_length > attention_start_ids.shape[-1]:
                    attention_start_ids = torch.cat([
                        attention_start_ids,
                        torch.arange(
                            attention_start_ids.shape[-1],
                            target_length,
                            device=attention_start_ids.device,
                            dtype=attention_start_ids.dtype
                        )[None, :].repeat(len(attention_start_ids), 1)
                    ], dim=1)
                diagonal_attend_mask = (
                    attention_start_ids[:, None, None, :] >
                    cache_position.reshape(1, 1, -1, 1))
                causal_mask = causal_mask * diagonal_attend_mask
            else:
                diagonal_attend_mask = (
                    torch.arange(target_length, device=device) >
                    cache_position.reshape(-1, 1))
                if config.sliding_window is not None:
                    # if we have sliding window, we should not attend to tokens
                    # beyond sliding window length, so we mask them out also
                    # the check is needed to verify is current checkpoint was
                    # trained with sliding window or not
                    if (not isinstance(past_key_values, SlidingWindowCache) or
                            sequence_length > target_length):
                        sliding_attend_mask = (
                            torch.arange(target_length, device=device) <= (
                                cache_position.reshape(-1, 1) -
                                config.sliding_window))
                        diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
                causal_mask = causal_mask * diagonal_attend_mask
                causal_mask = causal_mask[None, None, :, :].expand(
                    batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = (
                    causal_mask[:, :, :, :mask_length] +
                    attention_mask[:, None, None, :].to(causal_mask.device))
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = (
                    causal_mask[:, :, :, :mask_length].masked_fill(
                        padding_mask, min_dtype))
        return causal_mask


class HaploForConditionalGeneration(HaploPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ['lm_head.weight']
    _tp_plan = {'lm_head': 'colwise_rep'}
    _pp_plan = {'lm_head': (['hidden_states'], ['logits'])}

    def __init__(self, config):
        super().__init__(config)
        self.model = HaploModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        pixel_values_videos: torch.FloatTensor = None,
        image_sizes_videos: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_start_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (output_attentions if output_attentions is not None
                             else self.config.output_attentions)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (return_dict if return_dict is not None
                       else self.config.use_return_dict)

        # decoder outputs consists of
        # (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            pixel_values_videos=pixel_values_videos,
            image_sizes_videos=image_sizes_videos,
            attention_mask=attention_mask,
            attention_start_ids=attention_start_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float
        # if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int)
            else logits_to_keep)
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        pixel_values_videos=None,
        image_sizes_videos=None,
        attention_mask=None,
        attention_start_ids=None,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            attention_start_ids=attention_start_ids,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            model_inputs['pixel_values'] = pixel_values
            model_inputs['image_sizes'] = image_sizes
            model_inputs['pixel_values_videos'] = pixel_values_videos
            model_inputs['image_sizes_videos'] = image_sizes_videos

        return model_inputs
