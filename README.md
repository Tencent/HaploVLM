![Image](assets/logo.jpeg)

<div align="center">

# HaploOmni: Unified Single Transformer for Multimodal Video Understanding and Generation

[![arXiv paper](https://img.shields.io/badge/arXiv_paper-red)](https://arxiv.org/abs/2506.02975)&nbsp;
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/collections/stevengrove/ust-67d2582ac79d96983fa99697)&nbsp;
![Tencent ARC Lab](https://img.shields.io/badge/Developed_by-Tencent_ARC_Lab-blue)&nbsp;

</div>

## Highlights
With the advancement of language models, unified multimodal understanding and generation have made significant strides, with model architectures evolving from separated components to unified single-model frameworks. This paper explores an efficient training paradigm to build a single transformer for unified multimodal understanding and generation. Specifically, we propose a multimodal warmup strategy utilizing prior knowledge to extend capabilities. To address cross-modal compatibility challenges, we introduce feature pre-scaling and multimodal AdaLN techniques. Integrating the proposed technologies, we present the HaploOmni, a new single multimodal transformer. With limited training costs, HaploOmni achieves competitive performance across multiple image and video understanding and generation benchmarks over advanced unified models.

## Getting Started

### Installation

```bash
git clone https://github.com/Tencent/USTVLM.git
cd HaploVLM
git checkout HaploOmni

pip install -e . -v (Optional)
```

### Quick Start
Basic usage example for multimodal understanding:
```python
from haploomni import HaploOmniProcessor, HaploOmniForConditionalGeneration
from transformers import AutoTokenizer, AutoConfig
import torch

model_path = 'HaploOmni-Qwen2.5-7B'

processor = HaploOmniProcessor.from_pretrained(model_path)
tokenizer = processor.tokenizer
model = HaploOmniForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16
).to('npu')

conversation = [
    {'role': 'user', 'content': [
        # {'type': 'text', 'text': 'Who are you.'},
        {'type': 'text', 'text': 'How many people are there.'},
        {'type': 'image', 'path': 'test.jpg'}
    ]}
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors='pt',
    return_dict=True
).to('npu').to(torch.bfloat16)

outputs = model.generate(**inputs)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

## 🎤🎤🎤 Todo

- [ &#10004; ] Release the paper.
- [  &#10004; ] Release the checkpoints.
- [  &#10004; ] Release the inference demo for understanding task.
- [  &nbsp; &nbsp; ] Release the inference demo for generation task.


## Acknowledgement

```bibtex
@article{xiao2025haploomni,
  title={Haploomni: Unified single transformer for multimodal video understanding and generation},
  author={Xiao, Yicheng and Song, Lin and Yang, Rui and Cheng, Cheng and Xu, Zunnan and Zhang, Zhaoyang and Ge, Yixiao and Li, Xiu and Shan, Ying},
  journal={arXiv preprint arXiv:2506.02975},
  year={2025}
}
```
