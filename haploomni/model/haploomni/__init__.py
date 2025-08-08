# Copyright (c) Yicheng Xiao. All rights reserved.
from .configuration_haploomni import HaploOmniConfig
from .modeling_haploomni import HaploOmniForConditionalGeneration
from .image_processing_haploomni import HaploOmniImageProcessor
from .processing_haploomni import HaploOmniProcessor
from .video_processing_haploomni import HaploOmniVideoProcessor

from transformers import AutoImageProcessor, AutoProcessor, AutoConfig

AutoImageProcessor.register('HaploOmni', HaploOmniImageProcessor)
AutoProcessor.register('HaploOmni', HaploOmniProcessor)
AutoConfig.register('HaploOmni', HaploOmniConfig)

__all__ = [
    'HaploOmniConfig',
    'HaploOmniForConditionalGeneration',
    'HaploOmniImageProcessor',
    'HaploOmniProcessor',
    'HaploOmniVideoProcessor',
]
