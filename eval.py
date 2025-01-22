import argparse
import torch
import os
import json
import yaml
import numpy as np
from transformers import MistralForCausalLM
from mission import Mission, MissionTokenizer
from utils import get_missions_vocab, gen_mission_str

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ckpt', type=str, help='path to the model checkpoint')
    parser.add_argument('--data_dir', type=str, help='data directory')
    parser.add_argument('--data_file', type=str, help='data file')
    parser.add_argument('--training_config', type=str, default='configs/training_config.yaml')
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    with open(args.training_config, 'r') as f:
        config = yaml.safe_load(f)

    model = MistralForCausalLM.from_pretrained(args.ckpt)
    model.eval()
    model.cuda()

    num_node_tokens = config['num_node_tokens']
    context_length = config['context_length']
    tokenizer = MissionTokenizer(get_missions_vocab(num_node_tokens))
    tokenizer.model_max_length = context_length
    tokenizer.padding_side = 'left'      

    data_file = os.path.join(args.data_dir, args.data_file)

    with open(data_file, "r") as json_file:
        data = json.load(json_file)

    for example in data:
        mission = Mission(example, from_dict=True)
        # prompt = 'decision: '
        mission_str = gen_mission_str(mission, num_node_tokens=num_node_tokens)
        inputs = tokenizer([mission_str], return_tensors='pt', padding=True, truncation=True, max_length=context_length).to(model.device)
        print('Q:', tokenizer.batch_decode(inputs['input_ids']))
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], pad_token_id=tokenizer.pad_token_id, max_length=context_length, do_sample=False)
            print(tokenizer.batch_decode(outputs))            
        input('Press Enter to continue...')