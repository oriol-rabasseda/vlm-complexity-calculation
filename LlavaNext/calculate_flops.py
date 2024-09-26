import pip
import torch
from PIL import Image
from calflops import calculate_flops
from packaging.version import Version
from utils import *

def manage_imports():
    import transformers
    if Version(transformers.__version__) > Version('4.44'):
        return True
    else:
        pip.main(['install', 'git+https://github.com/huggingface/transformers@e40bb4845e0eefb52ec1e9cac9c2446ab36aef81'])
        return False


def count_flops_llavanext(model_name,
                          image,
                          prompt,
                          seq_len=128,
                          device = 'cuda',
                          max_new_tokens = 1):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        },
    ]

    if "onevision" in model_name:
        installed = manage_imports()
        if not installed:
            print('Transformers package version has been changed, please re-run the command')
            return

        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

    else:
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        processor = LlavaNextProcessor.from_pretrained(model_name)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_name,
                                                              torch_dtype=torch.float16, low_cpu_mem_usage=True)

    model.to(device)

    prompt_aux = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt_aux, return_tensors='pt').to(device, torch.float16)
    inputs["max_new_tokens"] = max_new_tokens

    if prompt == "":
        inputs = get_raw_input(processor.tokenizer, seq_len, inputs, device)

    _, _, _, result = calculate_flops(model=model,
                                      forward_mode = 'generate',
                                      kwargs = inputs,
                                      output_precision = 4,
                                      output_unit = 'T')
    
    return result