import os
import numpy as np
from datasets import load_dataset
from transformers import MistralForCausalLM
from mission import Mission, MissionTokenizer
from utils import get_missions_vocab, gen_mission_str
import glob  
import torch


def load_model(output_dir, sweep_id, model_name):
    model_dir = os.path.join(output_dir, f"sweeps/{sweep_id}", model_name)
    if os.path.exists(os.path.join(model_dir, 'best_model')):
        print(f"Found best model")
        model_dir = os.path.join(model_dir, 'best_model')
    else:
        if os.path.exists(model_dir):
            checkpoints = glob.glob(os.path.join(model_dir, 'checkpoint-*'))
            print(f"Found checkpoints: {checkpoints}")            
            if checkpoints:
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
                print(f"Loading {checkpoints[0]}")
                model_dir = checkpoints[0]
    print(f"Loading model from {model_dir}")
    if os.path.exists(model_dir):
        model = MistralForCausalLM.from_pretrained(model_dir)
        model.eval()
        if torch.cuda.is_available():
            print("Moving model to GPU")
            model.cuda()
        return model
    else:
        print(f"Model directory {model_dir} does not exist.")
        return None  
    

def load_data(config, data='val_file'):        
    num_node_tokens = config['num_node_tokens']
    # directed = config['directed']
    context_length = config['context_length']
    tokenizer = MissionTokenizer(get_missions_vocab(num_node_tokens))
    tokenizer.model_max_length = context_length
    tokenizer.padding_side = 'left'     

    data_file = os.path.join(config['data_dir'], config[data])
    dataset = load_dataset("json", data_files={'val':data_file})

    def tokenize(example):
        mission = Mission(example, from_dict=True)
        node_ids = np.random.permutation(num_node_tokens)[:mission.number_of_nodes()]
        mission_str = gen_mission_str(mission, node_ids=node_ids)
        token_ids = tokenizer.encode(mission_str)
        return {'input_ids': token_ids, 'mission_strs': mission_str, 'node_ids': node_ids} 

    dataset = dataset.map(tokenize, batched=False)['val'] 
    return tokenizer, dataset