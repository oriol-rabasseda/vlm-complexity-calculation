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

Install the transformer version according to the model that you want to run:
```
pip install transformers==<version>
```
Versions corresponding to each model can be seen in the table of supported models.

## Supported models

The aim of this tool is to evaluate the complexity of low-weight VLMs. For this reason, and due to the resources available, it has only been tested with low-weight models. However, it should also work with high-weight models if they are from the same collection.

The set of models supported and tested is as follows, with future methods also being available (e.g., VILA). If a model works with Transformers 4.45.1 (the latest version currently), it will probably work with future releases.

| Collection       | Model name                                                                                                    | transformers version |
| ---------------- | ------------------------------------------------------------------------------------------------------------- | -------------- |
| MiniCPM-V 1&2    | openbmb/MiniCPM-V <br> openbmb/MiniCPM-V-2                                                                    | 4.40.2, 4.41.2 |
| MiniCPM-V 2.5&2.6| openbmb/MiniCPM-Llama3-V-2_5 <br> openbmb/MiniCPM-V-2_6                                                       | 4.40.2, 4.45.1 |
| InternVL2        | OpenGVLab/InternVL2-1B <br> OpenGVLab/InternVL2-2B <br> OpenGVLab/InternVL2-4B <br> OpenGVLab/InternVL2-8B    | 4.40.2, 4.45.1 |
| Phi-Vision       | microsoft/Phi-3-vision-128k-instruct <br> microsoft/Phi-3.5-vision-instruct                                   | 4.40.2, 4.45.1 |
| Qwen2-VL         | Qwen/Qwen2-VL-2B-Instruct <br> Qwen/Qwen2-VL-7B-Instruct                                                      | 4.45.1         |
| LLaVa-Next       | llava-hf/llava-v1.6-mistral-7b-hf <br> llava-hf/llava-v1.6-vicuna-7b-hf <br> llava-hf/llama3-llava-next-8b-hf | 4.40.2         |
| LLaVa-OneVision  | llava-hf/llava-onevision-qwen2-0.5b-ov-hf <br> llava-hf/llava-onevision-qwen2-0.5b-si-hf<br> llava-hf/llava-onevision-qwen2-7b-ov-hf <br> llava-hf/llava-onevision-qwen2-7b-si-hf  | 4.45.1 |
| VILA             | Efficient-Large-Model/VILA1.5-3b <br> Efficient-Large-Model/Llama-3-VILA1.5-8B                                | 4.40.2         |


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
