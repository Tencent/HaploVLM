from haploomni import HaploOmniProcessor, HaploOmniForConditionalGeneration
from transformers import AutoTokenizer, AutoConfig
import torch


# model_path = '/group/40043/yichengxiao/huggingface/model/HaploOmni-Qwen2.5-7B'
model_path = '/group/40043/yichengxiao/HaploOmni-Qwen2.5-7B'

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

# text = processor.tokenizer.apply_chat_template(
#     conversation, chat_template=None, tokenize=False
# )

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors='pt',
    return_dict=True
    # chat_template=tokenizer.chat_template
).to('npu').to(torch.bfloat16)
generate_dict = {
    'max_length': 8172,
    'do_sample': True,
    'temperature': 0.6,
    'num_beams': 1
}
# inputs.update(generate_dict)
# import ipdb; ipdb.set_trace()
outputs = model.generate(**inputs)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
print(1)