import argparse
import datetime
import torch
import os
import json
from tqdm import tqdm
import ray
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import nltk
from nltk.metrics import precision, recall, f_measure
from nltk.translate import meteor_score
import jieba
import re

from PIL import Image
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_image(image_path):
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    return image

def contain_chinese_string(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(chinese_pattern.search(text))


def cal_per_metrics(pred, gt):

    metrics = {}

    if contain_chinese_string(gt) or contain_chinese_string(pred):
        reference = jieba.lcut(gt)
        hypothesis = jieba.lcut(pred)
    else:
        reference = gt.split()
        hypothesis = pred.split()

    metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)
    metrics["meteor"] = meteor_score.meteor_score([reference], hypothesis)

    reference = set(reference)
    hypothesis = set(hypothesis)
    metrics["f_measure"] = f_measure(reference, hypothesis)

    metrics["precision"] = precision(reference, hypothesis)
    metrics["recall"] = recall(reference, hypothesis)
    metrics["edit_dist"] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
    return metrics

@ray.remote(num_gpus=0)
def compute_metrics(data):
    """Calculate evaluation metrics"""
    edit_dist = []
    bleu = []
    meteor = []
    f_measure = []
    precision = []
    recall = []
    for example in tqdm(data):
        metrics = cal_per_metrics(example['output'], example['conversations'][1]['value'])
        edit_dist.append(metrics['edit_dist'])
        bleu.append(metrics['bleu'])
        meteor.append(metrics['meteor'])
        f_measure.append(metrics['f_measure'])
        precision.append(metrics['precision'])
        recall.append(metrics['recall'])
    return {
        "edit_dist": edit_dist,
        "bleu": bleu,
        "meteor": meteor,
        "f_measure": f_measure,
        "precision": precision,
        "recall": recall,
    }

@ray.remote(num_gpus=1)
def single_gpu_eval_model(args, data, language='en'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize model
    generator = Qwen2VLForConditionalGeneration
    model = LLM(
        model = args.model_path,
        gpu_memory_utilization = 0.7
    )
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=1.0,
        max_tokens=2048,
        skip_special_tokens=True,
    )
    
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Evaluation metrics
    messages = []
    for x in data:
        img_path = os.path.join(args.image_root, 'ocr', language+"_pdf_png", x['image'])
        messages.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{img_path}"},
                {"type": "text", "text": f"<image>\nOCR this image: "}
            ]
        }])
    results = []
    for i in tqdm(range(0, len(messages), args.batch_size)):
        batch = messages[i:i+args.batch_size]
        text = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        
        # Process visual input
        image_inputs, video_inputs = process_vision_info(batch)
        inputs = [{"prompt": prompt, "multi_modal_data": {"image": image}} for prompt, image in zip(text, image_inputs)]
        outputs = model.generate(
            inputs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        outputs_decoded = [o.outputs[0].text for o in outputs]
        results.extend(outputs_decoded)
    return results

def main(args):
    output_path = os.path.join(
            args.model_path,
            f"ocr_results_{datetime.datetime.now().strftime('%m%d_%H%M%S')}.json"
        )
    ray.init()
    num_gpus = int(ray.cluster_resources().get("GPU", 0))
    # prepare questions
    ALL_DATASETS = [
        'en_page_ocr', 'cn_page_ocr'
    ]
    target_datasets = [ALL_DATASETS[args.task_id]]
    for dataset in target_datasets:
        print(f"Processing {dataset}...")
        
        ds_path = os.path.join(args.data_root, 'ocr', f"{dataset}.json")
        
        if not os.path.exists(os.path.join(args.output_path, args.output_file)):
            data = json.load(open(ds_path, "r"))
            # questions = json.load(open(os.path.expanduser(args.question_file), "r"))
            mini_batches = [get_chunk(data, num_gpus, chunk_idx) for chunk_idx in range(num_gpus)]

            # distribute evaluation
            anss = [single_gpu_eval_model.remote(args, mini_batch, language=dataset.split('_')[0]) for mini_batch in mini_batches]
            anss = ray.get(anss)
            anss = [item for sublist in anss for item in sublist]
            data = [{**data, "output": ans} for data, ans in zip(data, anss)]
            with open(os.path.join(args.output_path, args.output_file), "w") as f:
                json.dump(data, f, indent=4)
        else:
            data = json.load(open(os.path.join(args.output_path, args.output_file), "r"))
            anss = [x['output'] for x in data]
        # save answers
        mini_eval_batches = [get_chunk(data, args.num_processes, chunk_idx) for chunk_idx in range(args.num_processes)]
        scores = [compute_metrics.remote(mini_eval_batch) for mini_eval_batch in mini_eval_batches]
        scores = ray.get(scores)
        metrics = {}
        for score in scores:
            for k in score:
                if k not in metrics:
                    metrics[k] = []
                metrics[k].extend(score[k])
        for k in metrics:
            assert len(metrics[k]) == len(data)
            metrics[k] = sum(metrics[k]) / len(metrics[k])
        with open(os.path.join(args.model_path, args.metrics_file), "w") as f:
            json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="xxx")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--image_root", type=str, default="xxx")
    parser.add_argument("--data_root", type=str, default="xxx")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--output_path", type=str, default="./logs")
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--metrics_file", type=str, default="FOX_cn_evaluations_metrics.json")
    parser.add_argument("--task_id", type=int, default=0, help="ID of the dataset to evaluate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_processes", type=int, default=16, help="Number of processes")
    args = parser.parse_args()
    main(args)
