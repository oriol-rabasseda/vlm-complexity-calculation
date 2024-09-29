from calflops import calculate_flops
import torch
from utils import *
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

def count_flops_qwen2(model_name,
                    image,
                    prompt,
                    seq_len=128,
                    device = 'cuda',
                    max_new_tokens = 1):

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
                    "type": "image"
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    inputs['max_new_tokens'] = max_new_tokens

    if prompt == "":
        inputs = get_raw_input(processor.tokenizer, seq_len, inputs, device)

    _, _, _, result = calculate_flops(model=model,
                                      forward_mode='generate',
                                      kwargs=inputs,
                                      output_precision=4,
                                      output_unit='T')

    result += '\nMemory usage:\t' + str(round(torch.cuda.max_memory_allocated(device=device)/2**30, 4)) + ' GBytes'
    torch.cuda.reset_peak_memory_stats(device=device)

    return result