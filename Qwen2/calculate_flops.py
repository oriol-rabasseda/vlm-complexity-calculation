import pip
from qwen_vl_utils import process_vision_info
from calflops import calculate_flops
import torch
from utils import *

def manage_inports():
    import transformers
    if transformers.__version__ == '4.45.0.dev0':
        return True
    else:
        pip.main(['install', 'git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830'])
        return False

def count_flops_qwen2(model_name,
                    image,
                    prompt,
                    seq_len=128,
                    device = 'cuda',
                    max_new_tokens = 1):
    installed = manage_inports()
    if not installed:
        print('Transformers package version has been changed, please re-run the command')
        return

    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map=device
    )

    # default processer
    processor = AutoProcessor.from_pretrained(model_name)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs
    inputs['max_new_tokens'] = max_new_tokens

    if prompt == "":
        inputs = get_raw_input(processor.tokenizer, seq_len, inputs, device)

    _, _, _, result = calculate_flops(model=model,
                                      forward_mode='generate',
                                      kwargs=inputs,
                                      output_precision=4,
                                      output_unit='T')

    return result