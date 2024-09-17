from calflops import calculate_flops
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

def count_flops_phi(model_name,
                    image,
                    prompt,
                    device = 'cuda',
                    max_new_tokens = 1024,
                    num_slices = 4):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation='eager'
    )

    processor = AutoProcessor.from_pretrained(model_name,
                                              trust_remote_code=True,
                                              num_crops=num_slices
                                              )

    messages = [
        {"role": "user", "content": f"<|image_1|>\n" + prompt},
    ]

    prompt_aux = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(prompt_aux, [image]).to(device)
    inputs["max_new_tokens"] = max_new_tokens
    inputs["eos_token_id"] = processor.tokenizer.eos_token_id

    calculate_flops(model=model,
                    forward_mode = 'generate',
                    kwargs = inputs,
                    output_precision = 4,
                    output_unit = 'T')