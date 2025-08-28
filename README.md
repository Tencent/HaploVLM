![Image](assets/logo.jpeg)

<div align="center">

# HaploVL - A Single-Transformer Baseline for Multi-Modal Understanding

[![arXiv paper](https://img.shields.io/badge/arXiv_paper-red)](http://arxiv.org/abs/2503.14694)&nbsp;
[![Project page](https://img.shields.io/badge/Project_page-green)](https://haplo-vl.github.io/)&nbsp;
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/collections/stevengrove/haplo-67d2582ac79d96983fa99697)&nbsp;
![Tencent ARC Lab](https://img.shields.io/badge/Developed_by-Tencent_ARC_Lab-blue)&nbsp;

</div>

HaploVL is a multimodal understanding foundation model that delivers comprehensive cross-modal understanding capabilities for text, images, and video inputs through a single transformer architecture.

## Project Updates

- 🔥 **News**: ```2025/08/08```: We release HaploOmni in the branch ([model](https://huggingface.co/EasonXiao-888/HaploOmni)).
- 🔥 **News**: ```2025/05/01```: HaploVL is accepted by ICML2025.

## Highlights
This repository contains the PyTorch implementation, model weights, and training code for **Haplo**.

![Image](assets/framework.png)

🌟 **Unified Architecture**: Single transformer model supporting early fusion of multi-modal inputs and auto-regressive response generation  
🌟 **Efficient Training**: Optimized training recipe leveraging pre-trained knowledge with reduced resource consumption  
🌟 **Scalable Design**: Flexible framework supporting both Ascend NPU and GPU environments  
🌟 **Extended Capabilities**: Native support for multiple image understanding and video processing

## Getting Started

### Installation

```bash
# Option1:
pip install git+https://github.com/Tencent/HaploVLM.git

# Option2:
git clone https://github.com/Tencent/HaploVLM.git
cd HaploVLM
pip install -e . -v
```

### Quick Start
Basic usage example:
```python
from haplo import HaploProcessor, HaploForConditionalGeneration

processor = HaploProcessor.from_pretrained('stevengrove/Haplo-7B-Pro')
model = HaploForConditionalGeneration.from_pretrained(
    'stevengrove/Haplo-7B-Pro',
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

### Gradio Demo
Launch an interactive demo:
```bash
python demo/demo.py \
    -m "stevengrove/Haplo-7B-Pro-Video" \
    --server-port 8080 \
    --device cuda \
    --dtype bfloat16
```

**Multi-Modal Capabilities**

| Category                      | Example                                  |
|-------------------------------|------------------------------------------|
| Single Image Understanding    | ![Demo1](assets/demo_1.png)              |
| Multi-Image Understanding         | ![Demo3](assets/demo_2.png)              |
| Video Understanding           | ![Demo2](assets/demo_3.png)              |


## Acknowledgement

```bibtex
@article{HaploVL,
    title={HaploVL: A Single-Transformer Baseline for Multi-Modal Understanding},
    author={Yang, Rui and Song, Lin and Xiao, Yicheng and Huang, Runhui and Ge, Yixiao and Shan, Ying and Zhao, Hengshuang},
    journal={arXiv preprint arXiv:2503.14694},
    year={2025}
}
@article{xiao2025haploomni,
  title={Haploomni: Unified single transformer for multimodal video understanding and generation},
  author={Xiao, Yicheng and Song, Lin and Yang, Rui and Cheng, Cheng and Xu, Zunnan and Zhang, Zhaoyang and Ge, Yixiao and Li, Xiu and Shan, Ying},
  journal={arXiv preprint arXiv:2506.02975},
  year={2025}
}
```
