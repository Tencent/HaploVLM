# Copyright (c) Lin Song. All rights reserved.
from .configuration_haplo import HaploConfig
from .modeling_haplo import HaploForConditionalGeneration
from .image_processing_haplo import HaploImageProcessor
from .processing_haplo import HaploProcessor
from .video_processing_haplo import HaploVideoProcessor

from transformers import AutoImageProcessor, AutoProcessor, AutoConfig

AutoImageProcessor.register('haplo', HaploImageProcessor)
AutoProcessor.register('haplo', HaploProcessor)
AutoConfig.register('haplo', HaploConfig)

__all__ = [
    'HaploConfig',
    'HaploForConditionalGeneration',
    'HaploImageProcessor',
    'HaploProcessor',
    'HaploVideoProcessor',
]
