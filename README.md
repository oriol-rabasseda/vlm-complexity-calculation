# Complexity calculator for VLMs

## Introduction
This tool is designed to compute the theoretical amount of FLOPs(floating-point operations)、MACs(multiply-add operations) and Parameters of Vision Language Models (VLMs). The tool computes all preprocess steps required to run the generation step of the model provided a real input (image + prompt). The models supported must be implemented in Pytorch and available in 🤗Huggingface Platform. The outcome of the tool is the printing of FLOPS, Parameter calculation value and proportion of each submodule of the model. This tool has been built in top of calflops (https://github.com/MrYxJ/calculate-flops.pytorch).

## Prerequisites
Create a conda environment with the required packages:
```
conda create --name <env> --file requirements.txt
```

Activate the environment:
```
conda activate <env>
```

## Supported models
The aim of this tool is to evaluate the complexity of low-weight VLMs. For this reason and due to the resources available, it has only been tested with low-weight models. However, it should also work with high-weight models if they are from the same collection.
The set of models supported and tested is the following, future methods will also be available (f.i. VILA):
| Collection | Model                | Name                                 | Link                                                              |
| ---------- | -------------------- | ------------------------------------ | ----------------------------------------------------------------- |
| MiniCPM-V  | MiniCPM-V 1.0        | openbmb/MiniCPM-V                    | [🤗](https://huggingface.co/openbmb/MiniCPM-V)                    |
| MiniCPM-V  | MiniCPM-V 2.0        | openbmb/MiniCPM-V-2                  | [🤗](https://huggingface.co/openbmb/MiniCPM-V-2)                  |
| MiniCPM-V  | MiniCPM-V 2.6        | openbmb/MiniCPM-V-2_6                | [🤗](https://huggingface.co/openbmb/MiniCPM-V-2_6)                |
| MiniCPM-V  | MiniCPM-Llama3-V 2.5 | openbmb/MiniCPM-Llama3-V-2_5         | [🤗](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5)         |
| InternVL2  | InternVL2-1B         | OpenGVLab/InternVL2-1B               | [🤗](https://huggingface.co/OpenGVLab/InternVL2-1B)               |
| InternVL2  | InternVL2-2B         | OpenGVLab/InternVL2-1B               | [🤗](https://huggingface.co/OpenGVLab/InternVL2-2B)               |
| InternVL2  | InternVL2-4B         | OpenGVLab/InternVL2-1B               | [🤗](https://huggingface.co/OpenGVLab/InternVL2-4B)               |
| InternVL2  | InternVL2-8B         | OpenGVLab/InternVL2-1B               | [🤗](https://huggingface.co/OpenGVLab/InternVL2-8B)               |
| Phi-Vision | Phi 3 Vision         | microsoft/Phi-3.5-vision-instruct    | [🤗](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)    |
| Phi-Vision | Phi 3.5 Vision       | microsoft/Phi-3-vision-128k-instruct | [🤗](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) |

## Execution
To run the tool, please run the following command:
```
python estimate_complexity.py \
--model-name openbmb/MiniCPM-V \
--image-file ./assets/airplane.jpeg \
--query "Tell me the model of this aircraft." \
--max_new_tokens" 1024 \
--device cuda \
--num_slices 4
```

Parameters:
* `--model-name`: name of the model. See Supported models for available names. Default openbmb/MiniCPM-V.
* `--image-file`: path to the image to be used in the complexity calculation.
* `--query`: text query to be used in the complexity calculation.
* `--max_new_tokens`: maximum number of new tokens. Default to 1024.
* `--device`: device where the calculation will be performed. Default to cuda.
* `--num_slices`: in the models where the input image is sliced into parts for better resolution and aspect ratio, the number of slices or maximum number of slices can be set. Default to 4.
