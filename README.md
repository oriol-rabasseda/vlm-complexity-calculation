# Complexity calculator for VLMs

## Introduction
This tool is designed to estimate the inference-time amount of FLOPs (Floating-Point OPerations), MACs (Multiply-ACcumulate operations), Trainable Parameters and Memory Requirements of Multimodal Large Language Models (MLLMs). The tool performs the preprocessing steps before the generation step of the model provided a real input (image + prompt). Models supported must be implemented in Pytorch and available in 🤗Huggingface Platform. The outcome of the tool is the printing of FLOPS, Trainable Parameters and Memory Requirments broken down by model sub-model. This tool has been built on top of calflops (https://github.com/MrYxJ/calculate-flops.pytorch).

## Prerequisites
Create a conda environment with the required packages:
```
conda create --name <env> --file requirements.txt
```

Activate the environment:
```
conda activate <env>
```

Install the transformer version according to the specific model:
```
pip install transformers==<version>
```

## Supported models

The aim of this tool is to evaluate the complexity of low-weight VLMs. For this reason, and due to the resources available, it has only been tested with low-weight models. The set of models tested is:

| Collection       | Model name  |
| ---------------- | ----------- |
| MiniCPM-V        | openbmb/MiniCPM-V <br> openbmb/MiniCPM-V-2 <br> openbmb/MiniCPM-Llama3-V-2_5 <br> openbmb/MiniCPM-V-2_6 <br> openbmb/MiniCPM-o-2_6 |
| InternVL2        | OpenGVLab/InternVL2-1B <br> OpenGVLab/InternVL2-2B <br> OpenGVLab/InternVL2-4B <br> OpenGVLab/InternVL2-8B |
| InternVL2.5      | OpenGVLab/InternVL2_5-1B <br> OpenGVLab/InternVL2_5-2B <br> OpenGVLab/InternVL2_5-4B <br> OpenGVLab/InternVL2_5-8B |
| InternVL2.5 MPO  | OpenGVLab/InternVL2_5-1B-MPO <br> OpenGVLab/InternVL2_5-2B-MPO <br> OpenGVLab/InternVL2_5-4B-MPO <br> OpenGVLab/InternVL2_5-8B-MPO |
| Phi-Vision       | microsoft/Phi-3-vision-128k-instruct <br> microsoft/Phi-3.5-vision-instruct <br> microsoft/Phi-4-multimodal-instruct |
| Qwen2-VL         | Qwen/Qwen2-VL-2B-Instruct <br> Qwen/Qwen2-VL-7B-Instruct |
| Qwen2.5-VL       | Qwen/Qwen2.5-VL-3B-Instruct <br> Qwen/Qwen2.5-VL-7B-Instruct |
| LLaVa-Next       | llava-hf/llava-v1.6-mistral-7b-hf <br> llava-hf/llava-v1.6-vicuna-7b-hf <br> llava-hf/llama3-llava-next-8b-hf |
| LLaVa-OneVision  | llava-hf/llava-onevision-qwen2-0.5b-ov-hf <br> llava-hf/llava-onevision-qwen2-0.5b-si-hf <br> llava-hf/llava-onevision-qwen2-7b-ov-hf <br> llava-hf/llava-onevision-qwen2-7b-si-hf |
| Ovis             | AIDC-AI/Ovis2-1B <br> AIDC-AI/Ovis2-2B <br> AIDC-AI/Ovis2-4B <br> AIDC-AI/Ovis2-8B <br> AIDC-AI/Ovis1.6-Llama3.2-3B |
| Deepseek-VL2     | deepseek-ai/deepseek-vl2-tiny <br> deepseek-ai/deepseek-vl2-small |
| Aya Vision       | CohereForAI/aya-vision-8b |
| Gemma-3          | google/gemma-3-4b-it |
| Granite-Vision   | ibm-granite/granite-vision-3.2-2b |

Models supported list might be bigger since heavy-weight models belonging to the same collections should also be supported.

## Execution
To run the tool, please run the following command:
```
python estimate_complexity.py \
--model-name openbmb/MiniCPM-V \
--image-file "" \
--query "" \
--seq_len 128 \
--max_new_tokens" 1 \
--device cuda \
--output-file ""
```

Parameters:
* `--model-name`: name of the model. See Supported models for available names. Default openbmb/MiniCPM-V.
* `--image-file`: path to the image to be used in the complexity calculation. If no image file is provided, then a dummy black image of resolution 1920x1080 is used. Default "".
* `--query`: text query to be used in the complexity calculation. If no query is provided, the dummy empty query is used and`seq_len` is used. Default "".
* `--seq_len`: number of padding tokens that will be added to the text input. Default 128.
* `--max_new_tokens`: maximum number of output tokens. Default to 1.
* `--device`: device where the calculation will be performed. Default to cuda.
* `--num_slices`: in the models where the input image is sliced into parts for better resolution and aspect ratio, the number of slices or maximum number of slices can be set. If it is not set, the default value of each model is used. Default to None.
* `--output-file`: filename where the text output will be writen. In the case that the output file is not set, the output will be printed. Default "".
