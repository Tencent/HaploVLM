# Copyright (c) Yicheng Xiao. All rights reserved.
import math
import os
from typing import Iterable, List, Union, Tuple

import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import select_best_resolution
from transformers.image_utils import (
    ImageInput, VideoInput, get_image_size, to_numpy_array)
from transformers.processing_utils import (
    ProcessingKwargs, ProcessorMixin, Unpack)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging

from .video_processing_haploomni import HaploOmniVideoProcessor


logger = logging.get_logger(__name__)


class HaploOmniProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    _defaults = {
        'text_kwargs': {
            'padding': False,
        },
        'image_kwargs': {},
        'video_kwargs': {},
    }


class HaploOmniProcessor(ProcessorMixin):

    attributes = ['image_processor', 'tokenizer', 'video_processor']
    valid_kwargs = [
        'chat_template',
        'num_image_tokens',
        'image_token',
        'video_token',
    ]
    image_processor_class = 'AutoImageProcessor'
    tokenizer_class = 'AutoTokenizer'
    video_processor_class = 'AutoImageProcessor'

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        num_image_tokens=None,
        chat_template=None,
        image_token='<image>',
        video_token='<video>',
        **kwargs,
    ):
        if video_processor is None:
            self.video_processor = HaploOmniVideoProcessor()
        else:
            self.video_processor = video_processor
        self.num_image_tokens = num_image_tokens
        self.image_token = tokenizer.image_token if hasattr(
            tokenizer, 'image_token') else image_token
        self.video_token = tokenizer.video_token if hasattr(
            tokenizer, 'video_token') else video_token
        super().__init__(
            image_processor, tokenizer, video_processor,
            chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput] = None,
        audio=None,
        videos: VideoInput = None,
        **kwargs: Unpack[HaploOmniProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            HaploOmniProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError(
                'Invalid input text. Please provide a string, '
                'or a list of strings')

        image_inputs = video_inputs = {}
        num_image_tokens = []
        num_video_tokens = []

        if images is not None:
            image_inputs = self.image_processor(
                images, **output_kwargs['images_kwargs'])

            image_sizes = iter(image_inputs['image_sizes'])
            height, width = get_image_size(
                to_numpy_array(image_inputs['pixel_values'][0][0]),
                channel_dim=output_kwargs['images_kwargs'].get('data_format'),
            )  # get image tensor size
            text, num_image_tokens = self._expand_image_tokens(
                text, image_sizes, height, width, self.image_token)

        if videos is not None:
            video_inputs = self.video_processor(
                videos, **output_kwargs['videos_kwargs'])

            one_video = to_numpy_array(video_inputs['pixel_values_videos'][0])
            height, width = get_image_size(
                one_video[0],
                channel_dim=output_kwargs['images_kwargs'].get('data_format'))
            num_frames = one_video.shape[0]
            patches_height_width = int(math.sqrt(self.num_image_tokens))
            num_video_tokens = (
                num_frames * patches_height_width * patches_height_width) + 1
            text = [
                sample.replace(
                    self.video_token, self.video_token * num_video_tokens
                ) for sample in text
            ]
            num_video_tokens = [
                [patches_height_width * patches_height_width] * num_frames +
                [1]
            ] * len(videos)

        text_inputs = self.tokenizer(text, **output_kwargs['text_kwargs'])

        image_token_id = self.tokenizer(self.image_token)['input_ids'][0]
        video_token_id = self.tokenizer(self.video_token)['input_ids'][0]
        attention_start_ids = []
        for input_ids, attn_mask in zip(
                text_inputs['input_ids'], text_inputs['attention_mask']):
            start_ids = torch.arange(
                len(input_ids), device=input_ids.device)
            image_mask = input_ids == image_token_id
            while image_mask.any():
                img_inds = image_mask.cumsum(0)
                per_img_mask = (img_inds <= num_image_tokens[0]) & image_mask
                assert per_img_mask.sum() == num_image_tokens[0]
                start_ids[per_img_mask] = start_ids[per_img_mask].min()
                image_mask[per_img_mask] = False
                num_image_tokens.pop(0)

            video_mask = input_ids == video_token_id
            while video_mask.any():
                for num_frame_tokens in num_video_tokens[0]:
                    video_inds = video_mask.cumsum(0)
                    per_vid_mask = ((video_inds <= num_frame_tokens) &
                                    video_mask)
                    assert per_vid_mask.sum() == num_frame_tokens
                    start_ids[per_vid_mask] = start_ids[per_vid_mask].min()
                    video_mask[per_vid_mask] = False
                num_video_tokens.pop(0)
            attention_start_ids.append(start_ids)
        text_inputs['attention_start_ids'] = torch.stack(attention_start_ids)
        return BatchFeature(
            data={**text_inputs, **image_inputs, **video_inputs})

    def _expand_image_tokens(
        self,
        text: List[TextInput],
        image_sizes: Iterable[Union[List[int], int]],
        height: int,
        width: int,
        special_token: str,
        num_frames: int = 1,
    ):
        prompt_strings = []
        num_all_image_tokens = []
        for sample in text:
            while special_token in sample:
                image_size_list = next(image_sizes)
                original_size = (image_size_list[0] if num_frames != 1
                                 else image_size_list)
                if not isinstance(original_size, (list, tuple)):
                    original_size = original_size.tolist()
                orig_height, orig_width = original_size
                num_image_tokens = self._get_number_of_features(
                    orig_height, orig_width, height, width)
                sample = sample.replace(
                    special_token,
                    '<placeholder>' * num_image_tokens * num_frames,
                    1)
                num_all_image_tokens.append(num_image_tokens)
            prompt_strings.append(sample)
        text = [sample.replace('<placeholder>', special_token)
                for sample in prompt_strings]
        return text, num_all_image_tokens

    def _get_number_of_features(
        self,
        orig_height: int,
        orig_width: int,
        height: int,
        width: int
    ) -> int:
        image_grid_pinpoints = self.image_processor.image_grid_pinpoints
        height_best_resolution, width_best_resolution = select_best_resolution(
            [orig_height, orig_width], image_grid_pinpoints
        )
        scale_height, scale_width = (height_best_resolution // height,
                                     width_best_resolution // width)

        patches_height = patches_width = int(math.sqrt(self.num_image_tokens))
        unpadded_features, newline_features = self._get_unpadded_features(
            orig_height, orig_width, patches_height, patches_width,
            scale_height, scale_width
        )

        # The base patch covers the entire image (no CLS for SigLIP)
        num_image_tokens = unpadded_features + newline_features
        return num_image_tokens

    def _get_unpadded_features(
        self,
        height: int,
        width: int,
        patches_height: int,
        patches_width: int,
        scale_height: int,
        scale_width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        current_height = patches_height * scale_height
        current_width = patches_width * scale_width

        original_aspect_ratio = width / height
        current_aspect_ratio = current_width / current_height
        if original_aspect_ratio > current_aspect_ratio:
            new_height = int(height * (current_width / width))
            padding = (current_height - new_height) // 2
            current_height -= padding * 2
        else:
            new_width = int(width * (current_height / height))
            padding = (current_width - new_width) // 2
            current_width -= padding * 2

        unpadded_features = current_height * current_width
        newline_features = current_height

        ratio = math.sqrt(
            current_height * current_width / (9 * patches_height**2))
        if ratio > 1.1:
            unpadded_features = (int(current_height // ratio) *
                                 int(current_width // ratio))
            newline_features = int(current_height // ratio)
        return (unpadded_features, newline_features)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names +
                                  image_processor_input_names))

    # override to save video-config in a separate config file
    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise ValueError(f'Provided path ({save_directory}) '
                             'should be a directory, not a file')
        os.makedirs(save_directory, exist_ok=True)
        video_processor_path = os.path.join(save_directory, 'video_processor')
        self.video_processor.save_pretrained(video_processor_path)

        video_processor_present = 'video_processor' in self.attributes
        if video_processor_present:
            self.attributes.remove('video_processor')

        outputs = super().save_pretrained(save_directory, **kwargs)

        if video_processor_present:
            self.attributes += ['video_processor']
        return outputs

    # override to load video-config from a separate config file
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        processor = super().from_pretrained(
            pretrained_model_name_or_path, **kwargs)

        if isinstance(processor, tuple):
            processor = processor[0]

        video_processor = HaploOmniVideoProcessor.from_pretrained(
            pretrained_model_name_or_path, subfolder='video_processor'
        )
        processor.video_processor = video_processor
        return processor


__all__ = ['HaploOmniProcessor']
