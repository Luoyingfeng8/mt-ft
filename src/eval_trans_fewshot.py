from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BertModel,
    AutoModelForSeq2SeqLM,
    FSMTTokenizer,
    FSMTForConditionalGeneration,
    # LlamaCrossAttentionEncDec
)
import datasets
import transformers
import numpy as np
import re
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser
import torch
import tqdm
from functools import partial
import logging
import json
import random
import regex

import time
from functools import reduce


logger = logging.getLogger(__name__)
log_level = "ERROR"
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()
from accelerate.utils import set_seed, gather_object
import evaluate
from accelerate import Accelerator

LANG_TABLE = {
    "af": "Afrikaans",
    "am": "Amharic",
    "an": "Aragonese",
    "ar": "Arabic",
    "as": "Assamese",
    "av": "Avaric",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "dz": "Dzongkha",
    "el": "Modern Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gd": "Gaelic",
    "gl": "Galician",
    "gu": "Gujarati",
    "ha": "Hausa",
    "he": "Modern Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "ig": "Igbo",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Central Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "ku": "Kurdish",
    "ky": "Kirghiz",
    "li": "Limburgish",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "nb": "Norwegian Bokmål",
    "ne": "Nepali",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "oc": "Occitan",
    "or": "Oriya",
    "pa": "Panjabi",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "rw": "Kinyarwanda",
    "se": "Northern Sami",
    "sh": "Serbo-Croatian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovene",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tk": "Turkmen",
    "tr": "Turkish",
    "tt": "Tatar",
    "ug": "Uighur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "wa": "Walloon",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zh": "Chinese",
    "zu": "Zulu",
}

def is_whitespace(string):
    # 使用正则表达式匹配空白字符或不可见字符
    pattern = r'^[\s\p{C}[\x00-\xFF]]+$'
    match = regex.match(pattern, string)
    return match is not None

def extract_pred(pred_text, split_str, remove_special_tokens=[]):
    ## extract pred
    pred = pred_text.split(split_str)[0].strip()
    pred = pred.split("\n")[0].strip()
    ## remove special tokens
    for s in remove_special_tokens:
        pred = pred.replace(s, "")
    ## last step: check
    pred = "#" if is_whitespace(pred) else pred
    return pred

def get_special_tokens(tokenizer):
    remove_special_tokens = ["<unk>", "</s>", "<pad>"]
    if getattr(tokenizer, "pad_token", None):
        remove_special_tokens.append(tokenizer.pad_token)
    if getattr(tokenizer, "eos_token", None):
        remove_special_tokens.append(tokenizer.eos_token)
    if getattr(tokenizer, "bos_token", None):
        remove_special_tokens.append(tokenizer.bos_token)
    if getattr(tokenizer, "unk_token", None):
        remove_special_tokens.append(tokenizer.unk_token)
    return remove_special_tokens

def load_model(args):
    # torch_dtype='auto'
    global accelerator
    torch_dtype=torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=f"cuda:{accelerator.process_index}",
    )
    # model = LlamaCrossAttentionEncDec.from_pretrained(args.model_name_or_path, torch_dtype=torch_dtype, device_map={"": accelerator.process_index})
    # model = AutoModelForSeq2SeqLM.from_pretrained(
    #     args.model_name_or_path, 
    #     torch_dtype=torch_dtype,
    #     trust_remote_code=True,
    #     device_map={"": accelerator.process_index},
    # )
    return model

def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left", trust_remote_code=True)
    
    if "Llama-2" in args.model_name_or_path or "Tower" in args.model_name_or_path or "LLaMA" in args.model_name_or_path:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    if "Llama-3" in args.model_name_or_path:
        tokenizer.pad_token_id = 128002

    return tokenizer

def load_data(file):
    try:
        data = json.load(open(file)) # json file
    except:
        data = [json.loads(line) for line in open(file)] # jsonline
    return data


def eval_few_shot():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, )
    parser.add_argument("--eval_mode", type=str, default='fewshot')
    parser.add_argument("--test_file", type=str,)
    parser.add_argument("--lang_pair", type=str, default='de-en')
    parser.add_argument("--few_shot_file", type=str,)
    parser.add_argument("--shot", type=int, default=3)
    parser.add_argument("--res_file", type=str, )
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--num_batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()

    set_seed(args.seed)
    
    global accelerator
    accelerator = Accelerator()
    
    model = load_model(args)
    tokenizer = load_tokenizer(args)
    remove_special_tokens = get_special_tokens(tokenizer)
    
    test_dataset = load_data(args.test_file)
    
    src_lang, tgt_lang = args.lang_pair.split("-")
    src_fullname = LANG_TABLE[src_lang]
    tgt_fullname = LANG_TABLE[tgt_lang]

    prefix = f"Translate this from {src_fullname} to {tgt_fullname}:\n"

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()
    start = time.time()

    ## for few-shot evaluate
    def make_shots(example):
        src = example["translation"][src_lang]
        demonstrations = random.sample(fewshot_dataset, args.shot)
        prompt = prefix
        for shot in demonstrations:
            s, t = shot["translation"][src_lang], shot["translation"][tgt_lang]
            prompt += f"{src_fullname}: {s}\n" + f"{tgt_fullname}: {t}\n"
        prompt += f"{src_fullname}: {src}\n" + f"{tgt_fullname}: "
        example["prompt"] = prompt
        return example
    
    def zero_shot(example):
        src = example["translation"][src_lang]
        prompt = prefix
        prompt += f"{src_fullname}: {src}\n" + f"{tgt_fullname}: "
        example["prompt"] = prompt
        return example

    if args.eval_mode == "fewshot":
        fewshot_dataset = load_data(args.few_shot_file)
        test_dataset = list(map(make_shots, test_dataset))
    elif args.eval_mode == "zeroshot":
        test_dataset = list(map(zero_shot, test_dataset))

    # batch, left pad (for inference), and tokenize
    def make_batch(prompts, batch_size=4):
        batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
        batches_tok = []
        for prompt_batch in batches:
            input_ids = tokenizer(
                    prompt_batch, 
                    return_tensors="pt", 
                    padding='longest', 
                    truncation=False, 
                    pad_to_multiple_of=8,
                    add_special_tokens=False).to("cuda") 
            batches_tok.append(input_ids)
                
        return batches_tok

    # divide the prompt list onto the available GPUs 
    test_dataset_input = [x["prompt"] for x in test_dataset]
    with accelerator.split_between_processes(test_dataset_input) as prompts:
        results = dict(outputs=[], num_tokens=0)

        # have each GPU do inference in batches
        prompt_batches = make_batch(prompts, batch_size=args.num_batch)
        prompt_batches = tqdm.tqdm(prompt_batches, total=len(prompt_batches), disable=not accelerator.is_local_main_process)
        for prompts_tokenized in prompt_batches:
            outputs_tokenized = model.generate(**prompts_tokenized, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)

            # remove prompt from gen. tokens
            outputs_tokenized = [ tok_out[len(tok_in):] 
                for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 

            # count and decode gen. tokens 
            num_tokens = sum([ len(t) for t in outputs_tokenized ])
            outputs = tokenizer.batch_decode(outputs_tokenized)
            # print("\n\n".join(outputs))
            outputs = list(map(lambda x: extract_pred(x, split_str=src_fullname, remove_special_tokens=remove_special_tokens),  outputs))
            # store in results{} to be gathered by accelerate
            results["outputs"].extend(outputs)
            results["num_tokens"] += num_tokens

    results = [ results ] # transform to list, otherwise gather_object() will not collect correctly

    # collect results from all the GPUs
    results_gathered = gather_object(results)
    
    if accelerator.is_main_process:
        timediff = time.time() - start
        num_tokens = sum([r["num_tokens"] for r in results_gathered ])
        preds = list(reduce(lambda x,y: x+y["outputs"], results_gathered, []))
        print(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")
        refs = [ [item["translation"][tgt_lang]] for item in test_dataset]
        # result = metric.compute(predictions=preds, references=refs)
        # print(result)
        with open(args.res_file, mode='w') as fout:
            fout.write("\n".join(preds) + '\n')
                
if __name__ == "__main__":
    eval_few_shot()
