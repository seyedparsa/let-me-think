import json
import os
import re
import numpy as np
import networkx as nx
from ast import literal_eval
from utils import generate_graph, dfs
from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling


class Mission(object):
    def __init__(self, args, from_dict=False):
        if from_dict:
            self.graph_type = args['start']
            self.task_type = args['task_type']
            self.graph = nx.from_edgelist(literal_eval(args['graph']))
            self.start = args['start']
            self.target = args['target']
            if args['phrag']:
                self.phrag = nx.from_edgelist(literal_eval(args['phrag']))
            return

        self.graph_type = args.graph_type
        self.task_type = args.task_type
        self.graph = generate_graph(args)
        if self.task_type == 'decision':
            self.phrag = generate_graph(args) 
        self.start, self.target = np.random.choice(args.num_nodes, 2, replace=False)

    def search(self, search_type):
        if search_type == 'dfs':
            walk = dfs(self.graph, self.start)
            walk = walk[:walk.index(self.target) + 1]
            return walk


class MissionTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab):        
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.unk_token = 'UNK'        
        self.pad_token = 'PAD'
        self.bos_token = 'BOS'
        self.sep_token = '/'
        self.eos_token = 'EOS'
        self.unk_token_id = vocab[self.unk_token]
        self.pad_token_id = vocab[self.pad_token]
        self.bos_token_id = vocab[self.bos_token]
        self.sep_token_id = vocab[self.sep_token]
        self.eos_token_id = vocab[self.eos_token]
        super().__init__()
        

    def encode(self, mission_str, max_length):
        tokens = re.findall(r"([A-Za-z]+|\d+|\S)", mission_str)
        token_ids = [self.vocab[token] for token in tokens]
        # token_ids = token_ids[:max_length]
        # print(mission_str)
        # print(len(token_ids))
        assert(len(token_ids) <= max_length)
        token_ids += [self.vocab['PAD']] * (max_length - len(token_ids))
        return token_ids

    def decode(self, token_ids):
        return ' '.join([self.id_to_token[token_id] for token_id in token_ids])
    
    def get_vocab(self):
        return self.vocab
    
    def save_vocabulary(self, save_directory, filename_prefix=None):
        """Save the vocabulary to a file."""
        os.makedirs(save_directory, exist_ok=True)

        vocab_file = os.path.join(save_directory, (filename_prefix or "") + "vocab.json")
        with open(vocab_file, "w") as f:
            json.dump(self.vocab, f)

        return (vocab_file,)

    def __len__(self):
        return len(self.vocab)            
    

class MissionDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, examples):
        batch = super().__call__(examples)
        sep_token_id = self.tokenizer.sep_token_id
        for i, input_ids in enumerate(batch["input_ids"]):
            sep_index = input_ids.tolist().index(sep_token_id) + 1
            batch["labels"][i][:sep_index] = -100
        return batch