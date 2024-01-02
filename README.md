# PureMM


# Release
* [2013-12-28]：Model and inference code released.  


# Preparation
* Because PureMM uses lora tuning the LLM. Before evaluation, you should download the base LLM model:<br>
[vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5) <br>
* PureMM uses visual encoder. Please download from<br>
[openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)


# Evaluation
Before evaluation, please download the MME benchmark files [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) into the mme_bench directory. <br>
Then run the cmd: <br>

**`sh eval/eval_mme.sh`**

# MME Benchmark
PureMM achieved 1647.8 points, which was top4 on MME benchmark at 2023-12-31. <br>

![mme](https://github.com/Q-MM/PureMM/tree/main/assets/mme-20231231.png)


# Acknowledgments
* [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon. Thanks for their wonderful work.
* [Vicuna](https://github.com/lm-sys/FastChat): the amazing open-sourced large language model!
