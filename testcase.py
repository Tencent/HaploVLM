# Copyright (c) Lin Song. All rights reserved.
import argparse

import torch

from haplo import (HaploProcessor, HaploForConditionalGeneration)


def testcase(processor, model, device):
    conversation = [
        {
          'role': 'system',
          'content': [
              {'type': 'text',
               'text': 'Your name is Haplo developed by Tencent ARC Lab. '
                       'You are a helpful assistant.'},
            ],
        },
        {
          'role': 'user',
          'content': [
              {'type': 'text', 'text': 'Who are you?'},
            ],
        },
        # {
        #   'role': 'user',
        #   'content': [
        #       {'type': 'video', 'path': 'assets/video-1.mp4'},
        #       {'type': 'text', 'text': 'Describe this video.'},
        #     ],
        # },
        # {
        #   'role': 'user',
        #   'content': [
        #     {'type': 'image', 'path': 'assets/example-image.png'},
        #     {'type': 'text', 'text': 'Describe this image.'},
        #     ],
        # },
    ]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        return_dict=True,
        return_tensors='pt',
        num_frames=6,
    )
    inputs = inputs.to(device, dtype=torch.bfloat16)

    generate_ids = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=256,
        num_beams=1,
        temperature=0.7,
        use_cache=True,
    )
    response = processor.decode(
        generate_ids[0, inputs.input_ids.shape[1]:],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True
    )
    print(response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Haplo Test Case')
    parser.add_argument('model_path', type=str,
                        help='Path to the pretrained model')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run the model on (e.g. npu, cuda)')
    args = parser.parse_args()

    # Load model and processor
    processor = HaploProcessor.from_pretrained(args.model_path)
    model = HaploForConditionalGeneration.from_pretrained(args.model_path)
    model = model.to(args.device, dtype=torch.bfloat16)

    testcase(processor, model, args.device)
