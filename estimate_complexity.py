from MiniCPM_V.calculate_flops import *
from Phi_Vision.calculate_flops import *
from InternVL2.calculate_flops import *
from Qwen2.calculate_flops import *
from LlavaNext.calculate_flops import *
#from VILA.calculate_flops import *
import argparse
import numpy as np

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

    if args.image_file != "":
        image = Image.open(args.image_file).convert('RGB')

    else:
        image = Image.new('RGB', (1920, 1080))


    if "openbmb" in args.model_name:
        count_flops_minicpm(model_name = args.model_name,
                            image = image,
                            prompt = args.query,
                            device = args.device,
                            max_new_tokens = args.max_new_tokens,
                            num_slices = args.num_slices
                            )
    
    elif "microsoft" in args.model_name:
        count_flops_phi(model_name=args.model_name,
                        image=image,
                        prompt=args.query,
                        seq_len=args.seq_len,
                        device=args.device,
                        max_new_tokens=args.max_new_tokens,
                        num_slices=args.num_slices
                        )

    elif "OpenGVLab" in args.model_name:
        count_flops_internvl2(model_name=args.model_name,
                              image=image,
                              prompt=args.query,
                              device=args.device,
                              max_new_tokens=args.max_new_tokens,
                              num_slices=args.num_slices
                              )
    
    elif "Qwen" in args.model_name:
        count_flops_qwen2(model_name=args.model_name,
                          image_name=args.image_file,
                          prompt=args.query,
                          device=args.device,
                          max_new_tokens=args.max_new_tokens
                          )
    
    elif "llava-hf" in args.model_name:
        count_flops_llavanext(model_name=args.model_name,
                              image=image,
                              prompt=args.query,
                              device=args.device,
                              max_new_tokens=args.max_new_tokens
                              )

    '''
    elif "VILA" in args.model_name:
        count_flops_vila(model_name=args.model_name,
                         image=image,
                         prompt=args.query,
                         device=args.device,
                         max_new_tokens=args.max_new_tokens
                         )
    '''