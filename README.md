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
# Option1:
pip install git+https://github.com/Tencent/HaploVLM.git

# Option2:
git clone https://github.com/Tencent/USTVLM.git
cd USTVLM
pip install -e . -v
```

### Quick Start
Basic usage example:
```python
from ust import USTProcessor, USTForConditionalGeneration

processor = USTProcessor.from_pretrained('stevengrove/UST-7B-Pro')
model = USTForConditionalGeneration.from_pretrained(
    'stevengrove/UST-7B-Pro',
    torch_dtype=torch.bfloat16
).to('cuda')

conversation = [
    {'role': 'user', 'content': [
        {'type': 'text', 'text': 'Describe this image.'},
        {'type': 'image', 'path': 'assets/example-image.png'}
    ]}
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    return_tensors='pt'
).to('cuda')

outputs = model.generate(inputs)
print(processor.decode(outputs[0]))
```


## Acknowledgement

```bibtex
@article{USTVL,
    title={USTVL: A Single-Transformer Baseline for Multi-Modal Understanding},
    author={Yang, Rui and Song, Lin and Xiao, Yicheng and Huang, Runhui and Ge, Yixiao and Shan, Ying and Zhao, Hengshuang},
    journal={arXiv preprint arXiv:2503.14694},
    year={2025}
}
```
