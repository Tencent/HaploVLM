# Copyright (c) Lin Song. All rights reserved.
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class HaploPerModelConfig(PretrainedConfig):

    model_type = 'haplo'
    keys_to_ignore_at_inference = ['past_key_values']

    # Default tensor parallel plan for base model `Haplo`
    base_model_tp_plan = {
        'layers.*.self_attn.q_proj': 'colwise',
        'layers.*.self_attn.k_proj': 'colwise',
        'layers.*.self_attn.v_proj': 'colwise',
        'layers.*.self_attn.o_proj': 'rowwise',
        'layers.*.mlp.gate_proj': 'colwise',
        'layers.*.mlp.up_proj': 'colwise',
        'layers.*.mlp.down_proj': 'rowwise',
    }
    base_model_pp_plan = {
        'embed_tokens': (['input_ids'], ['inputs_embeds']),
        'layers': (['hidden_states', 'attention_mask'], ['hidden_states']),
        'norm': (['hidden_states'], ['hidden_states']),
    }

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act='silu',
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        patch_size=14,
        norm_mode='rmsnorm',
        attention_rope=False,
        image_token_index=151646,
        video_token_index=151647,
        image_aspect_ratio='anyres_max_6',
        image_grid_pinpoints=[
            [336, 336],
            [336, 672],
            [336, 1008],
            [336, 1344],
            [336, 1680],
            [336, 2016],
            [672, 336],
            [672, 672],
            [672, 1008],
            [1008, 336],
            [1008, 672],
            [1344, 336],
            [1680, 336],
            [2016, 336]],
        image_newline=True,
        default_image_size=336,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.patch_size = patch_size
        self.norm_mode = norm_mode
        self.attention_rope = attention_rope
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.image_aspect_ratio = image_aspect_ratio
        self.image_grid_pinpoints = image_grid_pinpoints
        self.image_newline = image_newline
        self.default_image_size = default_image_size

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and 'type' in self.rope_scaling:
            self.rope_scaling['rope_type'] = self.rope_scaling['type']
        rope_config_validation(self)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class HaploConfig(PretrainedConfig):

    model_type = 'haplo'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        base_config: dict = dict(),
        pre_config: dict = dict(),
        post_config: dict = dict(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_config = HaploPerModelConfig(**base_config)
        self.pre_config = HaploPerModelConfig(**pre_config)
        self.post_config = HaploPerModelConfig(**post_config)

        for key, value in self.base_config.__dict__.items():
            self.__dict__[key] = value
