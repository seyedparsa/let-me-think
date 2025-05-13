import json
import os
import re
import numpy as np
import networkx as nx
from ast import literal_eval
from utils import gen_graph, dfs, bfs, gen_walk
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
                self.tegrat = args['tegrat']
            return

        self.graph_type = args.graph_type
        self.task_type = args.task_type
        self.graph = gen_graph(args)

        if args.st_pair == 'random':
            self.start, self.target = np.random.choice(self.graph.number_of_nodes(), 2, replace=False)
        elif args.st_pair == 'far':
            self.start, self.target = 0, self.graph.number_of_nodes() - 1

        if self.task_type == 'decision':
            self.phrag = gen_graph(args)     
            if args.st_pair == 'random':
                self.tegrat = np.random.choice(range(self.graph.number_of_nodes(), self.number_of_nodes()))
            elif args.st_pair == 'far':
                self.tegrat = self.number_of_nodes() - 1

    def search(self, search_type):
        if search_type == 'empty':
            return []
        elif search_type == 'optimal':
            return nx.shortest_path(self.graph, self.start, self.target)
        elif search_type.startswith('optimal-'):
            rep = int(search_type.split('-')[1])
            opt = nx.shortest_path(self.graph, self.start, self.target)
            repeated_opt = [opt[i] for i in range(len(opt)) for _ in range(rep)]
            return repeated_opt
        elif search_type == 'dfs':
            return dfs(self.graph, self.start, self.target)
        elif search_type == 'path':
            return dfs(self.graph, self.start, self.target, return_mistakes=False)
        elif search_type == 'dfs-pruned':
            return dfs(self.graph, self.start, self.target, return_mistakes=True, return_backtrack=False)
        elif search_type == 'dfs-short':
            sp = nx.shortest_path(self.graph, self.start, self.target)
            walk = dfs(self.graph, self.start)
            return walk[:len(sp)]
        elif search_type == 'bfs':
            return bfs(self.graph, self.start, self.target)
        elif search_type == 'random':
            walk = [self.start]
            while walk[-1] != self.target:
                walk.append(np.random.choice(list(self.graph.neighbors(walk[-1]))))
            return walk
        elif search_type.startswith('walk-'):
            walk_len = int(search_type.split('-')[1])
            walk_len = max(walk_len, len(nx.shortest_path(self.graph, self.start, self.target))-1)
            return gen_walk(self.graph, self.start, self.target, walk_len)
        elif search_type.startswith('walk_mix-'):
            max_len = int(search_type.split('-')[1])            
            min_len = len(nx.shortest_path(self.graph, self.start, self.target)) - 1            
            max_len = max(min_len, max_len)
            walk_len = np.random.randint(min_len, max_len + 1)
            return gen_walk(self.graph, self.start, self.target, walk_len)
        else:
            raise ValueError(f'Unknown search type: {search_type}')
        
    def number_of_nodes(self):
        res = self.graph.number_of_nodes()
        if self.task_type == 'decision':
            res += self.phrag.number_of_nodes()
        return res


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
        

    def encode(self, mission_str):
        tokens = self._tokenize(mission_str)
        token_ids = [self.vocab[token] for token in tokens]
        # if len(token_ids) > self.model_max_length:
        #     raise ValueError(f'Token length {len(token_ids)} exceeds max length {self.model_max_length}')
            # print(f'Warning: token length {len(token_ids)} exceeds max length {self.model_max_length}', flush=True)
            # token_ids = token_ids[:self.model_max_length]
        return token_ids

    def decode(self, token_ids, **kwargs):
        return ' '.join([self.id_to_token[int(token_id)] for token_id in token_ids])
    
    def get_vocab(self):
        return self.vocab
    
    def _tokenize(self, text):
        return ['BOS'] + re.findall(r"([A-Za-z]+|\d+|\S)", text) + ['EOS']

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        return self.id_to_token.get(index, self.unk_token)

    def convert_tokens_to_ids(self, tokens):
        return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self._convert_id_to_token(index) for index in ids]
    
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
            sep_index = input_ids.tolist().index(sep_token_id)
            batch["labels"][i][:sep_index + 1] = -100
        return batch