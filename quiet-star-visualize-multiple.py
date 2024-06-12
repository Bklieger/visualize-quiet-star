import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
import os
default_problem = "Question: The local firefighters are doing a fill the boot fundraiser. Their goal is to raise $6300. After the first 3 hours, they have raised $2100.  For how many hours do they have to fundraise in total to reach their goal, assuming an equal amount raised in every hour?\n\n Let's think step by step."

parser = argparse.ArgumentParser()
parser.add_argument("--input_text", type=str, default=default_problem, help="Custom input text for generation")
parser.add_argument("--checkpoint", type=str, default="ezelikman/quietstar-8-ahead")
parser.add_argument("--output_length", type=int, default=450, help="Length of the generated output")
parser.add_argument("--temp", type=float, default=0.3, help="Temperature for sampling")
parser.add_argument("--root_prefix", type=str, default="YOUR_ROOT_HERE")
parser.add_argument("--output_file", type=str, default="generated_data.json", help="File name for the output data")
args = parser.parse_args()

def model_init():
    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto',
        cache_dir=args.root_prefix + "cache"
    )
    print("Loaded model")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    special_tokens_to_add = ["<|startthought|>", "<|endthought|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
    model.resize_token_embeddings(len(tokenizer))
    
    model.tokenizer = tokenizer
    model.eval()
    return model

def generate_text_with_probabilities(input_text, model, tokenizer, output_length=50, temp=0.2):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    probabilities = []
    distances_from_top = []
    generated_tokens = []
    token_texts = []
    
    with torch.no_grad():
        for _ in range(output_length):
            outputs = model(input_ids)
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            start_thought_prob = next_token_probs[:, tokenizer.convert_tokens_to_ids("<|startthought|>")].item()
            num_more_likely_tokens = (next_token_probs > start_thought_prob).sum().item()
            
            next_token = torch.multinomial(next_token_probs, num_samples=1)
            
            next_token_text = tokenizer.decode(next_token.item())
            
            # Mask out start and end thought tokens during generation
            if next_token_text=="<|startthought|>" or next_token_text=="<|endthought|>":
                pass
            else:
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Unindent to show the start and end thought tokens
                # == [start] ==
                generated_tokens.append(next_token.item())
                token_texts.append(next_token_text)
                probabilities.append(start_thought_prob)
                distances_from_top.append(num_more_likely_tokens)
                # == [end] ==
            
            
            if next_token == tokenizer.eos_token_id:
                break
    
    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return output_text, generated_text, probabilities, distances_from_top, token_texts

model = model_init()
tokenizer = model.tokenizer
output_text, generated_text, probabilities, distances_from_top, token_texts = generate_text_with_probabilities(args.input_text, model, tokenizer, args.output_length, args.temp)

data = {
    "input_text": args.input_text,
    "output_text": output_text,
    "generated_text": generated_text,
    "probabilities": probabilities,
    "distances_from_top": distances_from_top,
    "token_texts": token_texts
}

print(data)

for i in range(10):
    output_text, generated_text, probabilities, distances_from_top, token_texts = generate_text_with_probabilities(args.input_text, model, tokenizer, args.output_length, args.temp)

    data["output_text"] += output_text
    data["generated_text"] += generated_text
    data["probabilities"] += probabilities
    data["distances_from_top"] += distances_from_top
    data["token_texts"] += token_texts


with open(args.output_file, "w") as f:
    json.dump(data, f)

print(f"Data saved to {args.output_file}")
