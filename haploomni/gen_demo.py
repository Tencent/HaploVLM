import torch
from diffusers import (
    CogVideoXDDIMScheduler,
)

from diffusers.utils import export_to_video
from haploomni import CogVideoXPipeline_Onedecoder as CogVideoXPipeline
from haploomni import HaploOmniProcessor, HaploOmniForConditionalGeneration
from transformers import AutoTokenizer, AutoConfig
import torch
import torch_npu
from PIL import Image


def video_to_gif(frames_list, gif_path, num_frames=9, width=720, height=480):
    # 保存为 GIF
    frames_list[0].save(gif_path, save_all=True, append_images=frames_list[1:], loop=0, duration=100)


def generate_img(source_images):
    images = [source_images[i] for i in [0, 1, 2, 4, 6, 7]]
    width, height = images[0].size
    total_width = width * len(images)
    result = Image.new("RGB", (total_width, height))
    for index, image in enumerate(images):
        result.paste(image, (index * width, 0))
    output_path = "test_img.png"
    result.save(output_path)


DTYPE = torch.bfloat16
device = torch.device('npu')

model_path = '/group/40043/yichengxiao/HaploOmni-Qwen2.5-7B_v1'
processor = HaploOmniProcessor.from_pretrained(model_path)
tokenizer = processor.tokenizer
model = HaploOmniForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=DTYPE
)

pipe_source_path = '/group/40043/yichengxiao/huggingface/model/CogVideoX-2b'
pipe = CogVideoXPipeline.from_pretrained(pipe_source_path, transformer=None, torch_dtype=DTYPE)
pipe.transformer = model.to(DTYPE)
pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
pipe.to(device)
pipe.vae.to(torch.float16)

prompt = ("A beautiful beach.")

video_generate = pipe(
    height=480,
    width=720,
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=9,   # 9 25
    use_dynamic_cfg=False,
    tokenizer_path=model_path,
    use_omni_template=False,
    use_qwen2vl_template=False,
    use_vanilla_template=True,
    negative_prompt=None,
    guidance_scale=6.5,
    generator=torch.Generator().manual_seed(42),
).frames[0]

export_to_video(video_generate, 'output_video.mp4', fps=8)
