# QQMM


# Release
* [2013-12-28]：Model and inference released.  



# Preparation
* Because QQMM use lora tuning the LLM. Before evaluation, you should download the base LLM model:<br>
[vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5) <br>
* QQMM uses visual encoder. Please download from [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)


# Evaluation
Before evaluation, please download the MME benchmark files [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) into mme_bench directory. <br>
Then run the cmd: <br>

**`sh eval/eval_mme.sh`**

# MME Benchmark
QQMM achieved xxx points, which was topx on MME benchmark at 2023-12-28.


# Acknowledgments
* [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon. Thanks for their wonderful work.
* [Vicuna](https://github.com/lm-sys/FastChat): the amazing open-sourced large language model!