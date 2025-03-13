# Copyright (c) Lin Song. All rights reserved.
import math
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from transformers.image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
    get_size_dict,
    select_best_resolution
)
from transformers.image_transforms import (
    PaddingMode,
    convert_to_rgb,
    pad,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from transformers.utils import TensorType, is_vision_available, logging


logger = logging.get_logger(__name__)


if is_vision_available():
    from PIL import Image


def make_batched_images(images) -> List[List[ImageInput]]:
    '''
    Accepts images in list or nested list format, and makes
    a list of images for preprocessing.

    Args:
        images (`Union[List[List[ImageInput]], List[ImageInput], ImageInput]`):
            The input image.

    Returns:
        list: A list of images.
    '''
    if (isinstance(images, (list, tuple)) and
            isinstance(images[0], (list, tuple)) and
            is_valid_image(images[0][0])):
        return [img for img_list in images for img in img_list]

    elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
        return images

    elif is_valid_image(images):
        return [images]

    raise ValueError(f'Could not make batched video from {images}')


def divide_to_patches(
    image: np.array,
    patch_size: int,
    input_data_format
) -> List[np.array]:
    '''
    Divides an image into patches of a specified size.

    Args:
        image (`np.array`):
            The input image.
        patch_size (`int`):
            The size of each patch.
        input_data_format (`ChannelDimension` or `str`):
            The channel dimension format of the input image.

    Returns:
        list: A list of np.array representing the patches.
    '''
    patches = []
    height, width = get_image_size(image, channel_dim=input_data_format)
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            if input_data_format == ChannelDimension.LAST:
                patch = image[i:i + patch_size, j:j + patch_size]
            else:
                patch = image[:, i:i + patch_size, j:j + patch_size]
            patches.append(patch)

    return patches


def expand_to_square(
    image: np.array, background_color, input_data_format
) -> np.array:
    '''
    Expands an image to a square by adding a background color.
    '''

    height, width = get_image_size(image, channel_dim=input_data_format)
    if width == height:
        return image
    elif width > height:
        result = np.ones(
            (width, width, image.shape[2]), dtype=image.dtype
        ) * background_color
        result[(width - height) // 2:(width - height) // 2 + height, :] = image
        return result
    else:
        result = np.ones(
            (height, height, image.shape[2]), dtype=image.dtype
        ) * background_color
        result[:, (height - width) // 2:(height - width) // 2 + width] = image
        return result


def _get_patch_output_size(image, target_resolution, input_data_format):
    original_height, original_width = get_image_size(
        image, channel_dim=input_data_format)
    target_height, target_width = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    return new_height, new_width


class HaploImageProcessor(BaseImageProcessor):

    model_input_names = ['pixel_values_videos']

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        image_grid_pinpoints: List = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = True,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {'height': 384, 'width': 384}
        size = get_size_dict(size, default_to_square=False)
        image_grid_pinpoints = (
            image_grid_pinpoints
            if image_grid_pinpoints is not None
            else [
                [384, 384],
                [384, 768],
                [384, 1152],
                [384, 1536],
                [384, 1920],
                [384, 2304],
                [768, 384],
                [768, 768],
                [768, 1152],
                [768, 1536],
                [768, 1920],
                [768, 2304],
                [1152, 384],
                [1152, 768],
                [1152, 1152],
                [1152, 1536],
                [1152, 1920],
                [1152, 2304],
                [1536, 384],
                [1536, 768],
                [1536, 1152],
                [1536, 1536],
                [1536, 1920],
                [1536, 2304],
                [1920, 384],
                [1920, 768],
                [1920, 1152],
                [1920, 1536],
                [1920, 1920],
                [1920, 2304],
                [2304, 384],
                [2304, 768],
                [2304, 1152],
                [2304, 1536],
                [2304, 1920],
                [2304, 2304],
            ]
        )

        self.do_resize = do_resize
        self.size = size
        self.image_grid_pinpoints = image_grid_pinpoints
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = (image_mean if image_mean is not None
                           else OPENAI_CLIP_MEAN)
        self.image_std = (image_std if image_std is not None
                          else OPENAI_CLIP_STD)
        self.do_pad = do_pad
        self.do_convert_rgb = do_convert_rgb

    def pad(
        self,
        image: np.ndarray,
        padding: Union[int, Tuple[int, int], Iterable[Tuple[int, int]]],
        mode: PaddingMode = PaddingMode.CONSTANT,
        constant_values: Union[float, Iterable[float]] = 0.0,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:

        if isinstance(padding, int) or len(padding) != 4:
            return pad(image, padding, mode, constant_values,
                       data_format, input_data_format)

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        if mode == PaddingMode.CONSTANT:
            image = np.pad(image, padding, mode='constant',
                           constant_values=constant_values)
        elif mode == PaddingMode.REFLECT:
            image = np.pad(image, padding, mode='reflect')
        elif mode == PaddingMode.REPLICATE:
            image = np.pad(image, padding, mode='edge')
        elif mode == PaddingMode.SYMMETRIC:
            image = np.pad(image, padding, mode='symmetric')
        else:
            raise ValueError(f'Invalid padding mode: {mode}')
        image = (
            to_channel_dimension_format(image, data_format, input_data_format)
            if data_format is not None else image
        )
        return image

    def _resize_for_patching(
        self, image: np.array, target_resolution: tuple, resample,
        input_data_format: ChannelDimension
    ) -> np.array:
        '''
        Resizes an image to a target resolution while maintaining aspect ratio.

        Args:
            image (np.array):
                The input image.
            target_resolution (tuple):
                The target resolution (height, width) of the image.
            resample (`PILImageResampling`):
                Resampling filter to use if resizing the image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            np.array: The resized and padded image.
        '''
        new_height, new_width = _get_patch_output_size(
            image, target_resolution, input_data_format)
        
        # Resize the image
        resized_image = resize(
            image,
            (new_height, new_width),
            resample=resample,
            input_data_format=input_data_format
        )

        return resized_image

    def _pad_for_patching(
        self, image: np.array, target_resolution: tuple,
        input_data_format: ChannelDimension
    ) -> np.array:
        '''
        Pad an image to a target resolution while maintaining aspect ratio.
        '''
        target_height, target_width = target_resolution
        new_height, new_width = _get_patch_output_size(
            image, target_resolution, input_data_format)

        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        padded_image = self.pad(
            image, padding=((paste_y, paste_y), (paste_x, paste_x)))

        return padded_image

    def get_image_patches(
        self,
        image: np.array,
        grid_pinpoints,
        size: tuple,
        patch_size: int,
        resample: PILImageResampling,
        data_format: ChannelDimension,
        input_data_format: ChannelDimension,
    ) -> List[np.array]:
        if not isinstance(grid_pinpoints, list):
            raise TypeError('grid_pinpoints must be a list of '
                            'possible resolutions.')

        possible_resolutions = grid_pinpoints

        image_size = get_image_size(image, channel_dim=input_data_format)
        
        best_resolution = select_best_resolution(
            image_size, possible_resolutions)
        resized_image = self._resize_for_patching(
            image, best_resolution, resample=resample,
            input_data_format=input_data_format
        )
        padded_image = self._pad_for_patching(
            resized_image, best_resolution,
            input_data_format=input_data_format)

        patches = divide_to_patches(
            padded_image, patch_size=patch_size,
            input_data_format=input_data_format)

        # make sure that all patches are in the input data format
        patches = [
            to_channel_dimension_format(
                patch, channel_dim=data_format,
                input_channel_dim=input_data_format)
            for patch in patches
        ]
        return patches

    def _pad_for_batching(
        self,
        pixel_values: List[np.ndarray],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        max_patch = max(len(x) for x in pixel_values)
        pixel_values = [
            self.pad(
                image,
                padding=((0, max_patch - image.shape[0]),
                         (0, 0), (0, 0), (0, 0)),
                data_format=data_format,
                input_data_format=input_data_format,
            )
            for image in pixel_values
        ]

        return pixel_values

    def _preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Image.Image:
        if do_resize:
            images = [
                resize(image=image, size=size, resample=resample,
                       input_data_format=input_data_format)
                for image in images
            ]

        if do_rescale:
            images = [
                self.rescale(image=image, scale=rescale_factor,
                             input_data_format=input_data_format)
                for image in images
            ]

        if do_normalize:
            images = [
                self.normalize(image=image, mean=image_mean, std=image_std,
                               input_data_format=input_data_format)
                for image in images
            ]

        images = [
            to_channel_dimension_format(
                image, data_format, input_channel_dim=input_data_format)
            for image in images
        ]

        return images

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        image_grid_pinpoints: List = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False)
        image_grid_pinpoints = (
            image_grid_pinpoints if image_grid_pinpoints is not None
            else self.image_grid_pinpoints)
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = (rescale_factor if rescale_factor is not None
                          else self.rescale_factor)
        do_normalize = (do_normalize if do_normalize is not None
                        else self.do_normalize)
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_convert_rgb = (do_convert_rgb if do_convert_rgb is not None
                          else self.do_convert_rgb)

        images = make_batched_images(images)

        if not valid_images(images):
            raise ValueError(
                'Invalid image type. Must be of type PIL.Image.Image, '
                'numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray.'
            )

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                'It looks like you are trying to rescale already '
                'rescaled images. If the input images have pixel values '
                'between 0 and 1, set `do_rescale=False` to avoid '
                'rescaling them again.'
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        new_images = []
        image_sizes = [get_image_size(image, channel_dim=input_data_format)
                       for image in images]
        for image in images:
            # convert image into a list of patches
            # we intentially use the same data format as the input data format
            size_tuple = (
                (size['height'], size['width'])
                if 'height' in size and 'width' in size
                else (size['shortest_edge'], size['shortest_edge'])
            )
            image_patches = self.get_image_patches(
                image,
                image_grid_pinpoints,
                size=size_tuple,
                patch_size=size['height'],
                resample=resample,
                data_format=input_data_format,
                input_data_format=input_data_format,
            )

            # preprocess patches
            pixel_values = self._preprocess(
                image_patches,
                do_resize=do_resize,
                size=size_tuple,
                resample=resample,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
                input_data_format=input_data_format,
            )
            pixel_values = np.array(pixel_values)
            new_images.append(pixel_values)

        if do_pad:
            processed_images = self._pad_for_batching(new_images)

        return BatchFeature(
            data={'pixel_values': processed_images,
                  'image_sizes': image_sizes},
            tensor_type=return_tensors
        )


__all__ = ['HaploImageProcessor']
