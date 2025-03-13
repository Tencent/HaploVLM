# Copyright (c) Lin Song. All rights reserved.
import argparse

import torch
import gradio as gr

from haplo import HaploProcessor
from haplo import HaploForConditionalGeneration

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Haplo Demo')
parser.add_argument('model_path', type=str,
                    help='Path to the pretrained model')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device to run the model on (e.g. cuda, npu)')
args = parser.parse_args()

# Load model and processor
processor = HaploProcessor.from_pretrained(args.model_path)
model = HaploForConditionalGeneration.from_pretrained(args.model_path)
model = model.to(args.device, dtype=torch.bfloat16)

def format_content(content):
    '''Convert content to HTML format for display'''
    elements = []
    for item in content:
        if item['type'] == 'text':
            elements.append(f'<div>{item["text"]}</div>')
        elif item['type'] == 'image':
            elements.append(
                f'<img src="file/{item["path"]}">')
        elif item['type'] == 'video':
            elements.append(f'''
            <video controls>
                <source src='file/{item['path']}'>
            </video>
            ''')
    return ''.join(elements)


# Initialize system message (keep fixed)
SYSTEM_MESSAGE = [{
    'role': 'system',
    'content': [{
        'type': 'text',
        'text': 'You are a helpful assistant.'
    }]
}]


def process_message(
        _, new_image, new_video, new_text, max_new_tokens, temperature, num_frames):
    '''Process single-turn conversation, ignoring history'''
    # Build user message content
    user_content = []
    if new_image:
        user_content.append({'type': 'image', 'path': new_image})
    if new_video:
        user_content.append({'type': 'video', 'path': new_video})
    if new_text.strip():
        user_content.append({'type': 'text', 'text': new_text})

    # Build temporary conversation history
    temp_conversation = SYSTEM_MESSAGE.copy()
    temp_conversation.append({'role': 'user', 'content': user_content})

    # Generate model response
    inputs = processor.apply_chat_template(
        temp_conversation,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        return_dict=True,
        return_tensors='pt',
        num_frames=num_frames,
    )
    inputs = inputs.to(args.device, dtype=torch.bfloat16)

    generate_ids = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        num_beams=2,
        temperature=temperature,
        repetition_penalty=1.5,
        top_p=0.95,
        use_cache=True,
    )

    response = processor.decode(
        generate_ids[0, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    # Build formatted history (show current conversation only)
    formatted_history = [
        (format_content(user_content), response)
    ]
    print(formatted_history)

    # Always return initial conversation state
    return SYSTEM_MESSAGE, formatted_history, None, None, ''


# Create Gradio interface
with gr.Blocks(
        theme=gr.themes.Soft(),
        fill_height=True) as demo:
    gr.Markdown(
        '# Haplo: A Single-Transformer Baseline for Multi-Modal Understanding')

    # Use fixed initial state
    conversation_state = gr.State(SYSTEM_MESSAGE)

    with gr.Row():
        chatbot = gr.Chatbot(
            label='Model Response',
            bubble_full_width=False,
            render_markdown=True,
            sanitize_html=False
        )

    with gr.Row():
        with gr.Column(scale=0.5):
            image_input = gr.Image(type='filepath', label='Upload Image')
            video_input = gr.Video(label='Upload Video')
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label='Input Text',
                placeholder='Enter your question...',
                lines=10,
                max_lines=10
            )
            max_tokens = gr.Slider(
                minimum=1,
                maximum=2048,
                value=256,
                step=1,
                label='Max New Tokens'
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label='Temperature'
            )
            num_frames = gr.Slider(
                minimum=1,
                maximum=16,
                value=12,
                step=1,
                label='Sampled Frames'
            )

    with gr.Row():
        clear_btn = gr.Button('Clear')
        submit_btn = gr.Button('Send', variant='primary')

    # Event handling
    submit_btn.click(
        fn=process_message,
        inputs=[conversation_state, image_input, video_input,
                text_input, max_tokens, temperature, num_frames],
        outputs=[conversation_state, chatbot, image_input,
                 video_input, text_input]
    )

    clear_btn.click(
        fn=lambda: [SYSTEM_MESSAGE, [], None, None, ''],
        outputs=[conversation_state, chatbot, image_input,
                 video_input, text_input]
    )


if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=8080)
