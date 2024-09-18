from MiniCPM_V.calculate_flops import *
from Phi_Vision.calculate_flops import *
from InternVL2.calculate_flops import *
from Qwen2.calculate_flops import *
#from VILA.calculate_flops import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str, default="openbmb/MiniCPM-V")
    parser.add_argument("--image-file", type=str, default="./assets/airplane.jpeg")
    parser.add_argument("--query", type=str, default="Tell me the model of this aircraft.")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_slices", type=int, default=4)
    args = parser.parse_args()

    image = Image.open(args.image_file).convert('RGB')

    if "MiniCPM" in args.model_name:
        count_flops_minicpm(model_name = args.model_name,
                            image = image,
                            prompt = args.query,
                            device = args.device,
                            max_new_tokens = args.max_new_tokens,
                            num_slices = args.num_slices
                            )
    
    elif "Phi" in args.model_name:
        count_flops_phi(model_name=args.model_name,
                        image=image,
                        prompt=args.query,
                        device=args.device,
                        max_new_tokens=args.max_new_tokens,
                        num_slices=args.num_slices
                        )

    elif "InternVL2" in args.model_name:
        count_flops_internvl2(model_name=args.model_name,
                              image=image,
                              prompt=args.query,
                              device=args.device,
                              max_new_tokens=args.max_new_tokens,
                              num_slices=args.num_slices
                              )
    
    elif "Qwen2" in args.model_name:
        count_flops_qwen2(model_name=args.model_name,
                          image_name=args.image_file,
                          prompt=args.query,
                          device=args.device,
                          max_new_tokens=args.max_new_tokens
                          )
    
    elif 

    '''
    elif "VILA" in args.model_name:
        count_flops_vila(model_name=args.model_name,
                         image=image,
                         prompt=args.query,
                         device=args.device,
                         max_new_tokens=args.max_new_tokens
                         )
    '''