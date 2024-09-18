from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
from calflops import calculate_flops


def count_flops_llavanext(model_name,
                          image,
                          prompt,
                          device = 'cuda',
                          max_new_tokens = 1024,
                          num_slices = 4):

    processor = LlavaNextProcessor.from_pretrained(model_name)

    model = LlavaNextForConditionalGeneration.from_pretrained(model_name,
                                                              torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to(device)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        },
    ]
    prompt_aux = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt_aux).to(device)
    inputs["max_new_tokens"] = max_new_tokens

    calculate_flops(model=model,
                    forward_mode = 'generate',
                    kwargs = inputs,
                    output_precision = 4,
                    output_unit = 'T')