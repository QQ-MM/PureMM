import argparse
import torch
import os
import json
from tqdm import tqdm

from model.conversation import conv_templates, SeparatorStyle
from model.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
import logging
import warnings
from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
from ..model import *


IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model_zoo creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'lora' in model_name.lower() and model_base is None:
        warnings.warn(
            'There is `lora` in model_zoo name but no `model_base` is provided. ')
    if 'lora' in model_name.lower() and model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)

        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        print('Loading PureMM from base model_zoo...')
        model = PureMMLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained,
                                                      **kwargs)
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            print(f'model_zoo.lm_head.weight.shape[0]: {model.lm_head.weight.shape[0]}; token_num: {token_num}')
            model.lm_head.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional PureMM weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                               non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        incompatible_keys = model.load_state_dict(non_lora_trainables, strict=False)
        # print("non_lora_trainables incompatible_keys: ", incompatible_keys)

        # vision_tower 在lora载入之前load，验证visual encoder lora训练效果
        vision_tower = model.get_vision_tower()
        print(f'vision_tower.is_loaded: {vision_tower.is_loaded}')
        if not vision_tower.is_loaded:
            vision_tower.load_model()
            print(f'vision_tower loaded!!!!')

        # print(f'model_zoo: {model_zoo}')
        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        # print(f'model_zoo after get lora: {model_zoo}')
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        # print(f'model_zoo after merge with lora: {model_zoo}')
        print('Model is loaded...')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = PureMMLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    
    vision_tower = model.get_vision_tower()
    print(f'vision_tower.is_loaded: {vision_tower.is_loaded}')
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    print(f'vision_tower loaded!!!!')

    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor
    print(f'image_processor: {image_processor}')

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    return tokenizer, model, image_processor, context_len


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f'model_name: {model_name}')
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    print('load model_zoo done!!!')
    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]

    questions_json = json.load(open(os.path.expanduser(args.question_file), "r"))
    image_dir = questions_json.get('root_dir', None)

    if 'mini_benchmark_IT_SFT_v1.2' in args.question_file:
        questions = questions_json.get('annotations')
    else:
        questions = questions_json.get('questions')
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    if not os.path.exists(args.answers_dir):
        logging.error(f'answers_dir not exist: {args.answers_dir}')
        os.mkdir(args.answers_dir)
    print('answers_dir: ', args.answers_dir)
    answers_file = os.path.join(args.answers_dir, os.path.basename(args.question_file))
    answers_file = answers_file.replace('.json', '_result.json')
    print('answers_file: ', answers_file)

    # answers_file = os.path.expanduser(args.answers_file)
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        # qs = line["text"]
        qs = line["question"]
        gt = line['answer']

        if 'mini_benchmark_IT_SFT_v1.2' in args.question_file:
            # qs = qs.replace('Please answer yes or no.', '')
            qs = qs.replace(' Please answer yes or no.', '\nAnswer the question using a single word or phrase.')

        cur_prompt = qs
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # image = Image.open(os.path.join(args.image_folder, image_file))
        if image_dir:
            image_path = os.path.join(image_dir, image_file)
        else:
            image_path = os.path.join(args.image_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensor = process_images([image], image_processor, model.config)[0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=128,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        raw_outputs = outputs
        if 'mini_benchmark_IT_SFT_v1.2' in args.question_file:
            if 'Yes' in outputs or 'yes' in outputs:
                outputs = 'Yes'
            else:
                outputs = 'No'

        # ans_id = shortuuid.uuid()
        cur_prompt = cur_prompt.replace('\nAnswer the question using a single word or phrase.',
                                        ' Please answer yes or no.')
        ans_file.write(json.dumps({"question_id": idx,
                                   "question": cur_prompt,
                                   "answer": outputs,
                                   "raw_answer": raw_outputs,
                                   "gt": gt}, ensure_ascii=False) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_zoo-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_zoo-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="conv_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--answers-dir", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)
