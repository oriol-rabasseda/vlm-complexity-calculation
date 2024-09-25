from MiniCPM_V.calculate_flops import *
from Phi_Vision.calculate_flops import *
from InternVL2.calculate_flops import *
from Qwen2.calculate_flops import *
from LlavaNext.calculate_flops import *
#from VILA.calculate_flops import *
import argparse
import numpy as np

def select_model(model_name="openbmb/MiniCPM-V",
                 image_file="",
                 query="",
                 seq_len=128,
                 max_new_tokens=1024,
                 device="cuda",
                 num_slices=None,
                 print_file=""):
    if image_file != "":
        image = Image.open(image_file).convert('RGB')

    else:
        image = Image.new('RGB', (1920, 1080))

    result = ""

    if "openbmb" in model_name:
        result = count_flops_minicpm(model_name=model_name,
                                     image=image,
                                     prompt=query,
                                     device=device,
                                     max_new_tokens=max_new_tokens,
                                     num_slices=num_slices
                                     )

    elif "microsoft" in model_name:
        result = count_flops_phi(model_name=model_name,
                                 image=image,
                                 prompt=query,
                                 seq_len=seq_len,
                                 device=device,
                                 max_new_tokens=max_new_tokens,
                                 num_slices=num_slices
                                 )

    elif "OpenGVLab" in model_name:
        result = count_flops_internvl2(model_name=model_name,
                                       image=image,
                                       prompt=query,
                                       seq_len=seq_len,
                                       device=device,
                                       max_new_tokens=max_new_tokens,
                                       num_slices=num_slices
                                       )

    elif "Qwen" in model_name:
       result = count_flops_qwen2(model_name=model_name,
                                  image_name=image_file,
                                  prompt=query,
                                  device=device,
                                  max_new_tokens=max_new_tokens
                                  )

    elif "llava-hf" in model_name:
        result = count_flops_llavanext(model_name=model_name,
                                       image=image,
                                       prompt=query,
                                       device=device,
                                       max_new_tokens=max_new_tokens
                                       )

    '''
    elif "VILA" in model_name:
        count_flops_vila(model_name=model_name,
                         image=image,
                         prompt=query,
                         device=device,
                         max_new_tokens=max_new_tokens
                         )
    '''

    if result != "":
        if print_file != "":
            f = open(print_file, "w")
            f.write(result)
            f.close()

        else:
            print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str, default="openbmb/MiniCPM-V")
    parser.add_argument("--image-file", type=str, default="")
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_slices", type=int, default=None)
    args = parser.parse_args()

    select_model(args.model_name, args.image_file, args.query, args.seq_len, args.max_new_tokens, args.device, args.num_slices)