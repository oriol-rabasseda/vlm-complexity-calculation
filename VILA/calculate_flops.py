from calflops import calculate_flops
from .llava.eval.run_vila import *
import torch

def count_flops_vila(model_name,
                     image,
                     prompt,
                     seq_len = 128,
                     device = 'cuda',
                     max_new_tokens = 1):

    images = [image]

    model_name = get_model_name_from_path(model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_name, model_name, device_map=device)

    qs = "<image>\n" + prompt
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_aux = conv.get_prompt()

    images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt_aux, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    inputs = dict()
    inputs['input_ids'] = input_ids
    inputs['images'] = [images_tensor]
    inputs['max_new_tokens'] = max_new_tokens
    inputs['stopping_criteria'] = [stopping_criteria]

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