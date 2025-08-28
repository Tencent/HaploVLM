# Copyright (c) Lin Song. All rights reserved.
import torch
from diffusers import CogVideoXPipeline as _CogVideoXPipeline
from mmpretrain.registry import MODELS
from mmpretrain.models.utils.huggingface import register_hf_tokenizer

import math
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.utils import replace_example_docstring
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.pipelines.cogvideo.pipeline_cogvideox import EXAMPLE_DOC_STRING, retrieve_timesteps
from transformers import AutoTokenizer
from transformers import T5Tokenizer
from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.video_processor import VideoProcessor
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.loaders import CogVideoXLoraLoaderMixin
import inspect
# from projects.DistillDIT.distilldit.datasets.image_edit_dataset import InferenceProcessor
from PIL import Image
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import EXAMPLE_DOC_STRING, retrieve_timesteps, retrieve_latents


# class InferenceProcessor:
#     """Processor for handling inference tasks."""

#     def __init__(self, tokenizer, model_config, conv_template="llama_3",
#                  video_fps=1, frames_upbound=6, force_sample=True):
#         """Initialize the processor."""
#         self.tokenizer = tokenizer
#         self.image_processor = build_image_processor(
#             image_size=model_config.meformer.get("img_size", 336)
#         )
#         self.video_args = SimpleNamespace(
#             video_fps=video_fps, frames_upbound=frames_upbound,
#             force_sample=force_sample
#         )
#         self.conv_template = conv_template
#         self.model_config = model_config

#     def process_path_to_image(self, messages):
#         for msg in messages:
#             files = msg['files']
#             images = []
#             for idx, path in enumerate(files):
#                 if os.path.splitext(path)[1] == '.mp4':
#                     video, video_time, frame_time, num_frames_to_sample = (
#                         process_video_with_decord(path, self.video_args)
#                     )
#                     visual = video
#                 else:
#                     visual = [Image.open(path).convert('RGB')]
#                 images.extend(visual)
#             msg['files'] = images
#         return messages

#     def process_image(self, images):
#         """Process image or video files."""
#         image_sizes = [img.size for img in images]
#         if len(images) > 1:
#             self.model_config.image_aspect_ratio = "pad"
#         images = process_images(images, self.image_processor, self.model_config)
#         return images, image_sizes

#     def process_text(self, text):
#         """Process text input."""
#         conv = conv_templates[self.conv_template].copy()
#         if isinstance(text[0], dict):
#             # this is the history
#             for message in text:
#                 conv.append_message(message['role'], message['text'])
#         else:
#             conv.append_message(conv.roles[0], text)
#             conv.append_message(conv.roles[1], None)
#         if conv.messages[-1][0] != conv.roles[1]:
#             conv.append_message(conv.roles[1], None)

#         prompt = conv.get_prompt()
#         input_ids = (
#             tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
#         )
#         attention_mask = torch.ones_like(input_ids)
#         return input_ids, attention_mask

#     def __call__(self, messages):
#         """Process input data."""
#         messages = self.process_path_to_image(messages)
#         for msg in messages:
#             files = msg['files']
#             msg['text'] = (
#                 f"{DEFAULT_IMAGE_TOKEN}\n" * len(files) +
#                 ' '.join(msg['text']))
#         input_ids, attention_mask = self.process_text(messages)
#         results = dict(input_ids=input_ids, attention_mask=attention_mask)

#         images = []
#         for msg in messages:
#             images.extend(msg['files'])
#         if len(images) > 0:
#             images, image_sizes = self.process_image(images)
#             results.update(dict(
#                 images=images, image_sizes=image_sizes
#             ))
#         return results


class CogVideoXPipeline_Onedecoder(DiffusionPipeline, CogVideoXLoraLoaderMixin):

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        vae: AutoencoderKLCogVideoX,
        transformer,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, vae=vae, transformer=transformer, scheduler=scheduler
        )

        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if hasattr(self, "vae") and self.vae is not None else 4
        )
        self.vae_scaling_factor_image = (
            self.vae.config.scaling_factor if hasattr(self, "vae") and self.vae is not None else 0.7
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 1, 3, 4)
        latents = 1 / self.vae_scaling_factor_image * latents

        if 'npu' in self.vae.device.type:
            latents = latents.to(torch.float16)  # convert to fp16 for 910B
        frames = self.vae.decode(latents).sample
        return frames

    def add_prefix_instruction(self, prompt):
        user_prompt = '<|user|>\n'
        generation_prompt = 'Generate an image according to the following instructions\n'
        assistant_prompt = '<|assistant|>\n<|diffusion|>'
        prompt_suffix = "<|end|>\n"
        prompt = f"{user_prompt}{generation_prompt}{prompt}{prompt_suffix}{assistant_prompt}"
        return prompt

    def add_prefix_instruction_qwen2vl(self, prompt):
        prefix_prompt_1 = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nGenerate a video according to the following instructions.\n'
        prefix_prompt_2 = '<|im_end|>\n<|im_start|>assistant\n'
        prompt = f"{prefix_prompt_1}{prompt}{prefix_prompt_2}"
        return prompt

    def add_prefix_instruction_vanilla(self, prompt, tokenizer):
        # system_prompt = 'Generate an image according to the following instructions.\n '
        # prompt = f"{system_prompt}{prompt}"

        system_context = "Describe the image or video by detailing the following aspects:\n1. The main content and theme of the video.\n2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.\n3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.\n4. background environment, light, style and atmosphere.\n5. camera angles, movements, and transitions used in the video:"
        # system_context = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
        # prompt = 'Generate a video according to the following instructions\n' + prompt

        user_context = prompt
        conversation = [dict(role='system', content=system_context),
                        dict(role='user', content=user_context)]

        prompt_ = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        return prompt_

    def get_llm_inputs_id(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        tokenizer_path: Optional[str] = None,
        use_noise_token: bool = False,
        use_omni_template: bool = False,
        use_qwen2vl_template: bool = False,
        use_vanilla_template: bool = False,
        time_token: Optional[str] = '<time>',
        noise_token: Optional[str] = '<noise>',
        noise_number: int = 4050,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        padding_mode: str = "max_length",
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        llm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if use_noise_token:
            llm_tokenizer.add_tokens([noise_token], special_tokens=True)
            llm_tokenizer.add_tokens([time_token], special_tokens=True)
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token

        if use_omni_template:
            llm_tokenizer.padding_side = 'left'

        llm_tokenizer.padding_side = 'left'

        if not do_classifier_free_guidance:
            if use_omni_template:
                prompt = [self.add_prefix_instruction(item) for item in prompt]
                inputs = llm_tokenizer(
                    prompt,
                    padding="longest",
                    max_length=2048,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
            elif use_qwen2vl_template:
                assert llm_tokenizer.padding_side == 'left'
                prompt = [self.add_prefix_instruction_qwen2vl(item) for item in prompt]
                inputs = llm_tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=226,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
            elif use_vanilla_template:
                assert llm_tokenizer.padding_side == 'left'
                prompt = [self.add_prefix_instruction_vanilla(item, llm_tokenizer) for item in prompt]
                inputs = llm_tokenizer(
                    prompt,
                    # padding="max_length",
                    # max_length=226,
                    padding="longest",
                    truncation=True,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
            else:
                inputs = llm_tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=226,
                    # padding="longest",
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            if use_omni_template:
                prompt = negative_prompt + prompt
                prompt = [self.add_prefix_instruction(item) for item in prompt]
                inputs = llm_tokenizer(
                    prompt,
                    padding="longest",
                    max_length=2048,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
            elif use_qwen2vl_template:
                assert llm_tokenizer.padding_side == 'left'
                prompt = negative_prompt + prompt
                prompt = [self.add_prefix_instruction_qwen2vl(item) for item in prompt]
                inputs = llm_tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=226,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
            elif use_vanilla_template:
                assert llm_tokenizer.padding_side == 'left'
                prompt = negative_prompt + prompt
                prompt = [self.add_prefix_instruction_vanilla(item, llm_tokenizer) for item in prompt]
                inputs = llm_tokenizer(
                    prompt,
                    # padding="max_length",
                    # max_length=226,
                    padding="longest",
                    truncation=True,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
            else:
                prompt = negative_prompt + prompt
                inputs = llm_tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=226,
                    # padding="longest",
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        return input_ids, attention_mask

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None, image_file=None, use_origin_image=False, timestep=None
    ):
        if not use_origin_image:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            shape = (
                batch_size,
                (num_frames - 1) // self.vae_scale_factor_temporal + 1,
                num_channels_latents,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            )

            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                latents = latents.to(device)

            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
            return latents
        else:
            video = Image.open(image_file).convert('RGB')
            video = self.video_processor.preprocess_video(video, height=height, width=width)
            video = video.to(device=device, dtype=self.dtype)
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            num_frames = (video.size(2) - 1) // self.vae_scale_factor_temporal + 1 if latents is None else latents.size(1)
            shape = (
                batch_size,
                num_frames,
                num_channels_latents,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            )

            if isinstance(generator, list):
                if len(generator) != batch_size:
                    raise ValueError(
                        f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                        f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                    )

                init_latents = [
                    retrieve_latents(self.vae.encode(video[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
                ]
            else:
                if 'npu' in self.vae.device.type:
                    video = video.to(torch.float16)
                init_latents = [retrieve_latents(self.vae.encode(vid.unsqueeze(0)), generator) for vid in video]

            init_latents = torch.cat(init_latents, dim=0).to(dtype).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
            init_latents = self.vae_scaling_factor_image * init_latents

            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self.scheduler.add_noise(init_latents, noise, timestep)

            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
            return latents


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        tokenizer_path: Optional[str] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        use_omni_template: bool = False,
        use_qwen2vl_template: bool = False,
        use_vanilla_template: bool = False,
        padding_mode: str = "max_length",
        image_file: str = None,
        model_config = None,
        use_origin_image = False,
        alpha: int = None,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.transformer.cogvideox_model.config.sample_height * self.vae_scale_factor_spatial):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to self.transformer.cogvideox_model.config.sample_height * self.vae_scale_factor_spatial):
                The width in pixels of the generated image. This is set to 720 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 4. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.cogvideox_model.config.max_text_seq_length` otherwise may lead to poor results.

        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        if num_frames > 49:
            raise ValueError(
                "The number of frames must be less than 49 for now due to static positional embeddings. This will be updated in the future to remove this limitation."
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 2.5 prepare llm tokenizer inputs
        noise_number = ((num_frames - 1) // 4 + 1) * (height // 16) * (width // 16)
        if image_file is not None:
            raise KeyError('under developing')
            # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            # processor = InferenceProcessor(tokenizer, model_config, conv_template='llama_3_edit')
            # source_image_file = os.path.join(image_file)
            # processor_source_img = transforms.Compose([
            #     transforms.Resize(max([height, width]), antialias=True),
            #     transforms.CenterCrop([height, width]),
            # ])
            # source_image = processor_source_img(Image.open(source_image_file).convert('RGB'))
            # messages_cond = [{'role': '<|start_header_id|>user<|end_header_id|>\n\n',
            #                 'files': [source_image], 'text': [prompt]}]
            # messages_uncond = [{'role': '<|start_header_id|>user<|end_header_id|>\n\n',
            #                 'files': [source_image], 'text': [""]}]
            # inputs_cond = processor(messages_cond)
            # inputs_uncond = processor(messages_uncond)
            # input_ids = [inputs_uncond["input_ids"].to(self.transformer.device), inputs_cond['input_ids'].to(self.transformer.device)]
            # attention_mask = [inputs_uncond["attention_mask"].to(self.transformer.device), inputs_cond["attention_mask"].to(self.transformer.device)]
            # images = inputs_cond["images"].to(torch.bfloat16).to(self.transformer.device)
            # image_sizes = inputs_cond["image_sizes"]
            # llm_inputs = dict(input_ids=input_ids, attention_mask=attention_mask)
        else:
            input_ids, attention_mask = self.get_llm_inputs_id(
                prompt,
                negative_prompt,
                do_classifier_free_guidance,
                tokenizer_path=tokenizer_path,
                use_noise_token=False,
                use_omni_template=use_omni_template,
                use_qwen2vl_template=use_qwen2vl_template,
                use_vanilla_template=use_vanilla_template,
                time_token='<time>',
                noise_token='<noise>',
                noise_number=noise_number,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
                padding_mode=padding_mode,
                device=device,
            )
            images = None
            image_sizes = None
            # import ipdb; ipdb.set_trace()
            # prompt_embeds = self.text_encoder(input_ids.to(device))[0]
            # prompt_embeds = prompt_embeds.to(dtype=self.transformer.dtype, device=device)
            llm_inputs = dict(input_ids=input_ids.to(device=self.transformer.device), attention_mask=attention_mask.to(self.transformer.device))

        # 3. Encode input prompt
        # prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        #     prompt,
        #     negative_prompt,
        #     do_classifier_free_guidance,
        #     num_videos_per_prompt=num_videos_per_prompt,
        #     prompt_embeds=prompt_embeds,
        #     negative_prompt_embeds=negative_prompt_embeds,
        #     max_sequence_length=max_sequence_length,
        #     device=device,
        # )
        # if do_classifier_free_guidance:
        #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_embeds = None

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents.
        latent_channels = 16
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            self.transformer.dtype,
            device,
            generator,
            latents,
            image_file,
            use_origin_image,
            timesteps[20:21].repeat(batch_size * num_videos_per_prompt),
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        # image_rotary_emb = (
        #     self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
        #     if self.transformer.cogvideox_model.config.use_rotary_positional_embeddings
        #     else None
        # )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                if images is not None:
                    raise KeyError('under developing')
                    # assert do_classifier_free_guidance is True
                    # noise_pred_list = []
                    # for i in range(2):
                    #     noise_pred = self.transformer.video_generation_forward_fun(
                    #         hidden_states=latent_model_input[i].unsqueeze(0),
                    #         # T5_prompt_embeds=prompt_embeds,
                    #         timestep=timestep[i].unsqueeze(0),
                    #         input_ids=llm_inputs['input_ids'][i],
                    #         attention_mask=llm_inputs['attention_mask'][i],
                    #         # image_rotary_emb=image_rotary_emb,
                    #         # attention_kwargs=attention_kwargs,
                    #         # images=images,
                    #         # image_sizes=image_sizes,
                    #         return_dict=True,
                    #         alpha=alpha,
                    #     ).sample
                    #     noise_pred_list.append(noise_pred.float())
                    # noise_pred = torch.cat(noise_pred_list, dim=0)
                else:
                    noise_pred = self.transformer.video_generation_forward_fun(
                        hidden_states=latent_model_input,
                        # T5_prompt_embeds=prompt_embeds,
                        timestep=timestep,
                        input_ids=llm_inputs['input_ids'],
                        attention_mask=llm_inputs['attention_mask'],
                        # image_rotary_emb=image_rotary_emb,
                        # attention_kwargs=attention_kwargs,
                        # images=images,
                        # image_sizes=image_sizes,
                        return_dict=True,
                    ).sample
                    noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(self.transformer.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)


register_hf_tokenizer(CogVideoXPipeline_Onedecoder, MODELS)
