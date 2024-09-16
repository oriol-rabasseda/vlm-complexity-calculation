from MiniCPM_V.calculate_flops import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str, default="openbmb/MiniCPM-V")
    parser.add_argument("--image-file", type=str, default="./assets/airplane.jpeg")
    parser.add_argument("--query", type=str, default="Tell me the model of this aircraft.")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    image = Image.open(args.image_file).convert('RGB')

    count_flops_minicpm(model_name = args.model_name,
                        image = image,
                        prompt = args.query,
                        device = args.device,
                        max_new_tokens = args.max_new_tokens)
