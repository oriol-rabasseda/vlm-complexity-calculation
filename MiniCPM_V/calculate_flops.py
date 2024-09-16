from calflops import calculate_flops
import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from PIL import Image

def get_inputs_minicpmv(image,
                        model,
                        tokenizer,
                        prompt,
                        max_new_tokens = 1024):
    images = [image]
    content = (
            tokenizer.im_start
            + tokenizer.unk_token * model.config.query_num
            + tokenizer.im_end
            + "\n"
            + prompt
    )

    final_input = "<用户>" + content + "<AI>"

    inputs = {"data_list": [final_input],
              "img_list": [images],
              "tokenizer": tokenizer,
              "max_new_tokens": max_new_tokens}

    return inputs

def get_inputs_minicpmv_2(image,
                          model,
                          tokenizer,
                          prompt,
                          max_new_tokens = 1024):

    print(model.config)

    if model.config.slice_mode:
        images, final_placeholder = model.get_slice_image_placeholder(
            image, tokenizer
        )
        content = final_placeholder + "\n" + content

        final_input = "<用户>" + content + "<AI>"

        inputs = {"data_list": [final_input],
                  "img_list": [images],
                  "tokenizer": tokenizer,
                  "max_new_tokens": max_new_tokens}

        return inputs

    else:
        return get_inputs_minicpmv(image, model, tokenizer, prompt, max_new_tokens)


def count_flops_minicpm(model_name,
                        image,
                        prompt,
                        device = 'cuda',
                        max_new_tokens = 1024):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model = model.to(device=device, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if model_name == "openbmb/MiniCPM-V":
        inputs = get_inputs_minicpmv(image, model, tokenizer, prompt, max_new_tokens)

    elif model_name == "openbmb/MiniCPM-V-2":
        inputs = get_inputs_minicpmv_2(image, model, tokenizer, prompt, max_new_tokens)

    calculate_flops(model=model,
                    forward_mode = 'generate',
                    kwargs = inputs,
                    output_precision = 4,
                    output_unit = 'T')

'''
elif model_name == "openbmb/MiniCPM-V-2_6":
    msgs = [{'role': 'user', 'content': [image, content]}]
    processor = AutoProcessor.from_pretrained(model.config._name_or_path, trust_remote_code=True)

    msg = msgs[0]
    content = msg["content"]
    cur_msgs = []
    for c in content:
        if isinstance(c, Image.Image):
            cur_msgs.append("(<image>./</image>)")
        elif isinstance(c, str):
            cur_msgs.append(c)
    msg["content"] = "\n".join(cur_msgs)

    prompts_lists = [processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)]
    input_images_lists = [[image]]

    inputs = processor(
        prompts_lists,
        input_images_lists
    ).to(model.device)

    inputs["tokenizer"] = tokenizer
    inputs["max_new_tokens"] = 1024

    if 'image_sizes' in inputs:
        del inputs['image_sizes']

    flops, macs, params = calculate_flops(model=model,
                                          forward_mode='generate',
                                          kwargs=inputs)
    print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))

elif model_name == "openbmb/MiniCPM-Llama3-V-2_5":
    msgs = [{'role': 'user', 'content': [image, content]}]
    processor = AutoProcessor.from_pretrained(model.config._name_or_path, trust_remote_code=True)

    msg = msgs[0]
    content = msg["content"]
    cur_msgs = []
    for c in content:
        if isinstance(c, Image.Image):
            cur_msgs.append("(<image>./</image>)")
        elif isinstance(c, str):
            cur_msgs.append(c)
    msg["content"] = "\n".join(cur_msgs)

    prompt = processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    images = [image]

    inputs = processor(prompt, images)

    params = dict()
    params["model_inputs"] = inputs
    params["tokenizer"] = tokenizer
    params["max_new_tokens"] = 1024

    flops, macs, params = calculate_flops(model=model,
                                          forward_mode='generate',
                                          kwargs=params)
    print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))

'''