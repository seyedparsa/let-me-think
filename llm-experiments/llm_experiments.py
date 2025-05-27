import os
import argparse

# Set environment variables first
os.environ["VLLM_USE_V1"] = "0"

import torch
from transformers import AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from scipy import stats
import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# Add vLLM imports
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from cached_generation import cached_generate as gen # honestly really messy and should be redone
# from vllm.logits_process import LogitsProcessor
from typing import List
# Set precision
model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
# model="simplescaling/s1-32B"
# model="Qwen/Qwen3-32B"
n = 32
test_time_tokens = 1024 * 4
depth = 4
short = 3
long = 5
dead_end = 3
seed = 0
num_cot_lengths = 16
np.random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Experiments for Graph Tasks")
    parser.add_argument("--model", 
                        type=str, 
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 
                        choices=["deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "simplescaling/s1-32B", "Qwen/Qwen3-32B"],
                        help="Select model from deepseek-ai/DeepSeek-R1-Distill-Qwen-32B, simplescaling/s1-32B, or Qwen/Qwen3-32B")
    parser.add_argument("--n", type=int, default=64, help="Number of prompts to generate.")
    parser.add_argument("--test_time_tokens", type=int, default=1024*3, help="Test time tokens (e.g. 2048).")
    parser.add_argument("--depth", type=int, default=2, help="Depth for the BridgeGraph.")
    parser.add_argument("--short", type=int, default=3, help="Short parameter for BridgeGraph (must be <= long).")
    parser.add_argument("--long", type=int, default=9, help="Long parameter for BridgeGraph (must be >= short).")
    parser.add_argument("--dead_end", type=int, default=0, help="Dead end parameter for BridgeGraph.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generation.")
    parser.add_argument("--num_cot_lengths", type=int, default=16, help="Number of chain-of-thought lengths to evaluate.")
    # New arguments for graph type
    parser.add_argument("--graph", type=str, default="bridge", choices=["bridge", "path"],
                        help="Graph type to use: 'bridge' for BridgeGraph or 'path' for PathGraph.")
    parser.add_argument("--path_length", type=int, default=5,
                        help="For PathGraph: number of nodes in each disjoint path component.")
    return parser.parse_args()

# Parse command line arguments
args = parse_args()
model = args.model
n = args.n
test_time_tokens = args.test_time_tokens
depth = args.depth
short = args.short
long = args.long
dead_end = args.dead_end
seed = args.seed
num_cot_lengths = args.num_cot_lengths
graph_type = args.graph
path_length = args.path_length

# Set the random seed
np.random.seed(seed)

# Load tokenizer from Qwen/Qwen2.5-32B-Instruct
if model == "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
elif model == "Qwen/Qwen3-32B":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
elif model == "simplescaling/s1-32B":
    tokenizer = AutoTokenizer.from_pretrained("simplescaling/s1-32B")
else:
    raise ValueError(f"Model {model} not supported")

dtype = torch.bfloat16
cached_generate = lambda llm, prompts, sampling_params: gen(llm, prompts, sampling_params, model=model) # honestly really messy and not that helpful
def init_llm():
    exit()
    return LLM(
        model=model,
        tensor_parallel_size=2,  # Adjust based on available GPUs
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        max_logprobs=160000,
        max_model_len=test_time_tokens+1000,
        max_num_seqs = 64,#128,
        enable_prefix_caching=True,
    )
llm = None#init_llm()
class BridgeGraph:
    def __init__(self, depth, short, long, dead_end = 0):
        self.depth = depth
        self.short = short
        self.long = long
        self.dead_end = dead_end
        assert short <= long, "please just swap them"
        self.graph = self.generate_graph()
    
    def generate_graph(self):
        self.cycle_length = self.long + self.short
        self.nodes_per_layer = self.cycle_length + self.dead_end
        self.number_of_nodes_per_component = (self.depth * self.nodes_per_layer + 1)
        graph = nx.Graph()
        for j in [0, self.number_of_nodes_per_component]:
            shared_node = j
            for i in range(self.depth):
                nodes = list(range(j + i * self.nodes_per_layer+1, j + i * self.nodes_per_layer+self.cycle_length+1))+[shared_node]
                graph.add_edges_from(zip(nodes, nodes[1:]+[nodes[0]]))
                if self.dead_end:
                    dead_end = [shared_node] + list(range(j + i * self.nodes_per_layer+self.cycle_length+1, j + i * self.nodes_per_layer+self.cycle_length+self.dead_end+1))
                    graph.add_edges_from(zip(dead_end[:-1], dead_end[1:]))
                shared_node = nodes[self.short-1]
        return graph

    def gen_instance(self):
        graph = self.graph
        perm = np.random.permutation(self.number_of_nodes_per_component * 2)
        graph = nx.relabel_nodes(graph, dict(enumerate(perm)), copy=True)
        return graph, np.random.permutation(graph.edges()), perm

    def create_prompt(self):
        graph, edges, perm = self.gen_instance()

        randomize_order = np.random.permutation([self.number_of_nodes_per_component - self.long-1 - self.dead_end, 2 * self.number_of_nodes_per_component - self.long-1 - self.dead_end])
        if model == "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" or model == "Qwen/Qwen3-32B":
            prompt = f"<|user|>\nGiven the following list of undirected edges in a graph (with nodes labeled 0 through {self.number_of_nodes_per_component * 2 - 1}), is node {perm[0]} in the same component as {perm[randomize_order[0]]} or as {perm[randomize_order[1]]}? (it has a path to exactly one of the two) Please reason step by step.\n" #
        elif model == "simplescaling/s1-32B":
            prompt = f"<|im_start|>user\nGiven the following list of undirected edges in a graph (with nodes labeled 0 through {self.number_of_nodes_per_component * 2 - 1}), is node {perm[0]} in the same component as {perm[randomize_order[0]]} or as {perm[randomize_order[1]]}? (it has a path to exactly one of the two) Please reason step by step.\n" #
        else:
            raise ValueError(f"Model {model} not supported")
        for edge in edges[:-1]:
            prompt += f"({edge[0]}, {edge[1]}), "
        if model == "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" or model == "Qwen/Qwen3-32B":
            prompt += f"({edges[-1][0]}, {edges[-1][1]})<|end of sentence|>\n<|assistant|><think>\n"
        elif model == "simplescaling/s1-32B":
            prompt += f"({edges[-1][0]}, {edges[-1][1]})<|im_end|>\n<|im_start|>assistant\n"
        return prompt, perm[self.number_of_nodes_per_component - self.long-1 - self.dead_end], perm[2*self.number_of_nodes_per_component - self.long-1 - self.dead_end], perm
class PathGraph:
    def __init__(self, length):
        self.length = length
        self.graph = self.generate_graph()
        # self.graph_edges = list(self.graph.edges())
    

    def gen_instance(self):
        graph = self.graph
        perm = np.random.permutation(self.length * 2)
        graph = nx.relabel_nodes(graph, dict(enumerate(perm)), copy=True)
        return graph, np.random.permutation(graph.edges()), perm[self.shortest_path], perm

    def create_prompt(self):
        graph, edges, shortest_path, perm = self.gen_instance()
        randomize_order = np.random.permutation([self.length-1, self.length*2-1])
        if model == "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" or model == "Qwen/Qwen3-32B":
            prompt = f"<|user|>\nGiven the following list of undirected edges in a graph (with nodes labeled 0 through {self.number_of_nodes_per_component * 2 - 1}), is node {perm[0]} in the same component as {perm[randomize_order[0]]} or as {perm[randomize_order[1]]}? (it has a path to exactly one of the two) Please reason step by step.\n" #
        elif model == "simplescaling/s1-32B":
            prompt = f"<|im_start|>user\nGiven the following list of undirected edges in a graph (with nodes labeled 0 through {self.number_of_nodes_per_component * 2 - 1}), is node {perm[0]} in the same component as {perm[randomize_order[0]]} or as {perm[randomize_order[1]]}? (it has a path to exactly one of the two) Please reason step by step.\n" #
        else:
            raise ValueError(f"Model {model} not supported")
        for edge in edges[:-1]:
            prompt += f"({edge[0]}, {edge[1]}), "
        if model == "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" or model == "Qwen/Qwen3-32B":
            prompt += f"({edges[-1][0]}, {edges[-1][1]})<|end of sentence|>\n<|assistant|><think>\n"
        elif model == "simplescaling/s1-32B":
            prompt += f"({edges[-1][0]}, {edges[-1][1]})<|im_end|>\n<|im_start|>assistant\n"

        # prompt += f"({edges[-1][0]}, {edges[-1][1]})\n<think>"
        return prompt, perm[self.length-1], perm[self.length*2-1], perm
def hypergeometric_majority_prob_with_tiebreaking(n_pop, k_success, m_sample):
    """
    Calculate probability of getting a majority of successes in m_sample draws
    without replacement from a population of n_pop items, k_success of which are successes.
    Ties (if m_sample is even) contribute 0.5 probability.
    P(X > m_sample / 2) + 0.5 * P(X == m_sample / 2) if m_sample is even
    P(X > m_sample / 2) if m_sample is odd
    """
    prob_strict_majority = stats.hypergeom.sf(m_sample // 2, n_pop, k_success, m_sample)

    prob_tie = 0.0
    if m_sample % 2 == 0 and m_sample > 0: # Handle ties for even, non-zero subsample size
        # pmf(k, M, n, N) is P(X=k)
        prob_tie = stats.hypergeom.pmf(m_sample // 2, n_pop, k_success, m_sample)

    return prob_strict_majority + 0.5 * prob_tie

def compute_candidate_probs_for_prompts(prompts: List[str],
                                        candidate_lists: List[List[str]], llm):
    """
    Computes the exact probability of each candidate answer string following the prompt,
    based on the model's logits.
    """
    if len(prompts) != len(candidate_lists):
        raise ValueError("The number of prompts must equal the number of candidate lists.")

    n = len(prompts)
    if n == 0:
        return torch.empty((0, 0))
    k = max(len(candidates) for candidates in candidate_lists) if n > 0 else 0
    if k == 0:
         return torch.empty((n, 0)) # Handle case with prompts but no candidates

    # --- Batch‐tokenize all candidates at once ---
    flat_cands = [cand for clist in candidate_lists for cand in clist]
    if len(flat_cands) == 0:
        return torch.empty((n, 0))

    # This returns a dict with "input_ids": List[List[int]]
    batch_enc = tokenizer(flat_cands, add_special_tokens=False)
    flat_token_lists = batch_enc["input_ids"]

    # Compute max length and collect all token IDs
    max_candidate_token_len = max(len(toks) for toks in flat_token_lists)
    all_candidate_tokens_flat = {tok for toks in flat_token_lists for tok in toks}

    # Now reshape back into List[List[List[int]]]
    tokenized_candidate_lists = []
    idx = 0
    for clist in candidate_lists:
        count = len(clist)
        tokenized_candidate_lists.append(flat_token_lists[idx:idx+count])
        idx += count

    # --- rest of your old setup (digit_token_ids, allowed_token_ids, ...) ---
    # Note: you may still want to warn if some tokens aren't in digit_token_ids, etc.
    digit_token_ids = set()
    for digit in "0123456789":
        toks = tokenizer.encode(digit, add_special_tokens=False)
        if toks:
            digit_token_ids.add(toks[0])
    allowed_token_ids = list(digit_token_ids)
    if not digit_token_ids:
        raise ValueError("Could not find token IDs for any digits.")


    if max_candidate_token_len == 0:
        #  print("Warning: All candidates tokenized to empty sequences.")
        raise ValueError("All candidates tokenized to empty sequences.")
        #  return torch.zeros((n, 1))


    # --- Generate with logprobs and restricted tokens ---
    # Request logprobs for a sufficient number of top tokens.
    # Even with allowed_token_ids, logprobs helps verify distribution.
    # Set num_logprobs to at least the number of allowed tokens + a buffer
    num_logprobs = len(allowed_token_ids) + 1 # Ensure all allowed tokens can be captured
    sampling_params = SamplingParams(
        temperature=0.0, # Deterministic probabilities
        max_tokens=max_candidate_token_len,
        min_tokens=max_candidate_token_len,
        logprobs=num_logprobs, # Request top N log probabilities
        allowed_token_ids=allowed_token_ids, # Restrict generation to these tokens
        ignore_eos=True,
        stop_token_ids=[],
        # prompt_logprobs=num_logprobs # Optional
    )
    outputs = cached_generate(llm, prompts, sampling_params)
    if llm is None and not outputs:
        llm = init_llm()
        outputs = cached_generate(llm, prompts, sampling_params)
    print("outputs generated")
    # --- Calculate sequence probabilities ---
    result_probs = torch.zeros((n, k))
    logprob_fallback = -100.0 # Fallback for tokens not in top-k logprobs

    for i, output in enumerate(outputs):
        if not output.outputs or not output.outputs[0].logprobs:
            print(f"Warning: No logprobs found for prompt {i}. Assigning zero probability.")
            continue

        logprobs_list = output.outputs[0].logprobs
        actual_generated_len = len(logprobs_list)
        if actual_generated_len < max_candidate_token_len:
             # This is more likely now if EOS is generated early within the allowed set
             print(f"Warning: Generated sequence length {actual_generated_len} is less than max candidate token length {max_candidate_token_len} for prompt {i}.")
             pass # Don't warn excessively, just proceed

        current_candidates_tokenized = tokenized_candidate_lists[i]

        for j, cand_tokens in enumerate(current_candidates_tokenized):
            cand_len = len(cand_tokens)
            assert cand_len <= max_candidate_token_len, f"Candidate length {cand_len} is greater than max candidate token length {max_candidate_token_len} for prompt {i}."
            if cand_len == 0 or cand_len > actual_generated_len:
                result_probs[i, j] = 0.0
                print(f"Warning: Candidate length {cand_len} is out of range for prompt {i}.")
                continue

            total_logprob = 0.0
            for step in range(cand_len):
                step_logprobs_dict = logprobs_list[step]
                target_token_id = cand_tokens[step]

                # Since we restricted allowed_token_ids, the target *should* be in logprobs
                # if its probability wasn't zero.
                # print(step_logprobs_dict)   
                logprob = step_logprobs_dict.get(str(target_token_id), step_logprobs_dict.get(target_token_id, logprob_fallback))

                assert logprob != logprob_fallback, (f"Warning: Allowed token {target_token_id} for candidate '{candidate_lists[i][j]}' at step {step} not in top {num_logprobs} logprobs for prompt {i}.")
                # if logprob == logprob_fallback:
                #      # This might happen if the model assigns truly zero probability even within the allowed set,
                #      # or if num_logprobs was somehow smaller than len(allowed_token_ids).
                #      assert False,(f"Warning: Allowed token {target_token_id} ('{tokenizer.decode([target_token_id])}') for candidate '{candidate_lists[i][j]}' at step {step} not in top {num_logprobs} logprobs for prompt {i}.")
                #      pass
                # if logprob != logprob_fallback:
                if type(logprob) == dict:
                    logprob = logprob.get("logprob", logprob_fallback)
                elif type(logprob) == float:
                    logprob = logprob
                else:
                    logprob = logprob.logprob
                # try:
                #     logprob = logprob.get("logprob", logprob_fallback)
                assert logprob != logprob_fallback, (f"Warning: Allowed token {target_token_id} for candidate '{candidate_lists[i][j]}' at step {step} not in top {num_logprobs} logprobs for prompt {i}.")

                # logprob = logprob.logprob
                total_logprob += logprob

            clamped_logprob = max(total_logprob, -700.0)
            result_probs[i, j] = torch.exp(torch.tensor(clamped_logprob))

    return result_probs

def sequential_scaling_with_sampling(n, depth, short, long, cot_lengths, repeats=1, llm=None, subsample_sizes=None, dead_end=0, graph_type="bridge", path_length=5):
    """
    Version of sequential_scaling that uses sampling instead of exact probability calculation.
    Optimized to perform model inference in a single batch across all CoT lengths.
    Includes budget forcing for CoT generation using a functional logits processor.

    Args:
        n: Number of prompts to generate
        depth, short, long: BridgeGraph parameters
        cot_lengths: List of chain of thought lengths to evaluate
        repeats: Number of chains of thought to generate for each prompt
        subsample_sizes: List of subsample sizes to evaluate (default: all sizes from 1 to repeats)
        dead_end: Dead end parameter for BridgeGraph
        graph_type: Must be either "bridge" or "path" (affects dataset generation)
        path_length: If graph_type=="path", number of nodes in each disjoint component.

    Returns:
        Dictionary containing empirical accuracy curves for each subsample size
        and theoretical probability curves
    """
    if subsample_sizes is None:
        subsample_sizes = list(range(1, repeats + 1, 2))
    else:
        subsample_sizes = [s for s in subsample_sizes if s <= repeats]

    # Fix: Check array size instead of truthiness
    test_time_tokens = max(1,max(cot_lengths)) if cot_lengths.size > 0 else 1 # This is the max_tokens for CoT

    # Create the dataset based on the selected graph type
    if graph_type == "bridge":
        dataset = BridgeGraph(depth, short, long, dead_end)
    elif graph_type == "path":
        dataset = PathGraph(path_length)
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")

    prompts = []
    answers = []
    incorrects = []
    perms = []

    for i in range(n):
        prompt, answer, incorrect, perm = dataset.create_prompt()
        prompts.append(prompt)
        answers.append(answer)
        incorrects.append(incorrect)
        perms.append(perm)

    # --- Prepare for CoT Generation with Budget Forcing ---
    logits_processors = []
    if test_time_tokens > 0:
        # Get EOS token ID and the ID for the token to force (e.g., space)
        eos_token_id = tokenizer.eos_token_id#151645 #
        print(eos_token_id)
        if eos_token_id is None:
            raise ValueError("Tokenizer does not have an EOS token defined.")
        

        force_token_candidates = tokenizer.encode("wait", add_special_tokens=False)
        if not force_token_candidates:
             print("Warning: Could not encode space 'wait' reliably. Using token ID 0 as force token.")
             force_token_id = 0
        else:
             force_token_id = force_token_candidates[0]

        banned_token_id = 151668 if model =="Qwen/Qwen3-32B" else 151645
        # Define the logits processor function using closure
        def budget_forcing_processor_func(token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
            """
            Logits processor function to force generation until max_tokens.
            Uses eos_token_id, force_token_id, and test_time_tokens from the outer scope.
            """
            # Note: We don't need max_tokens check here because ignore_eos=True
            # and the SamplingParams max_tokens handle the stopping condition.
            # We just need to prevent EOS selection before that.
            # logits[force_token_id] = 0

            logits[force_token_id] = torch.log(torch.exp(logits[tokenizer.eos_token_id])+torch.exp(logits[banned_token_id]))
            logits[tokenizer.eos_token_id] = -9999.
            logits[banned_token_id] = -9999.

            return logits

        # Add the function itself to the list
        logits_processors.append(budget_forcing_processor_func)

        # Generate multiple thinking steps for each prompt with vLLM using the 'n' parameter
        sampling_params = SamplingParams(
            max_tokens=test_time_tokens,
            temperature=0.6, #qwen 3 thinking temperature
            n=repeats,
            ignore_eos=True, # Crucial: Prevents stopping even if EOS is somehow sampled
            logits_processors=logits_processors # Pass the list containing our function,
        )

    # Prepare prompt batches - Use the original prompts list directly
    all_prompts_for_cot_gen = prompts

    # Generate thinking steps (full CoTs)
    if test_time_tokens > 0 and len(all_prompts_for_cot_gen) > 0:

        outputs = cached_generate(llm, all_prompts_for_cot_gen, sampling_params)
        if llm is None and not outputs:
            llm = init_llm()
            outputs = cached_generate(llm, all_prompts_for_cot_gen, sampling_params)
    else:
        outputs = []

    # === NEW batched encoding ===

    # 1) Collect all generated_texts (in prompt‐major order)
    all_generated_texts = []
    flat_cot_token_ids = []
    for i in range(n):
        if i < len(outputs):
            assert len(outputs[i].outputs) == repeats, \
                f"Expected {repeats} outputs for prompt {i}, got {len(outputs[i].outputs)}"
            for seq_out in outputs[i].outputs:
                all_generated_texts.append(seq_out.text)
                flat_cot_token_ids.append(seq_out.token_ids)
        else:
            # no output → treat as empty string → encodes to [] by tokenizer
            all_generated_texts.extend([""] * repeats)

    # 2) Batch‐tokenize once
    # batch_cot_enc      = tokenizer.batch_encode_plus(all_generated_texts, add_special_tokens=False)
    # flat_cot_token_ids = batch_cot_enc["input_ids"]  # List[List[int]], length = n*repeats

    # 3) Reshape back into processed_prompts[i][j]
    processed_prompts = []
    idx = 0
    for _ in range(n):
        chains = []
        for _ in range(repeats):
            chains.append(flat_cot_token_ids[idx])
            idx += 1
        processed_prompts.append(chains)
    print(all_generated_texts[-1])
    # print(processed_prompts[-1][-1])
    # print(151667 in processed_prompts[-1][-1])
    print(np.mean([len(p) for p in flat_cot_token_ids]))
    print(len(processed_prompts[-1][-1]))
    print("wait" in all_generated_texts[-1])
    # Initialize result dictionaries
    empirical_accuracy = {size: [0.0] * len(cot_lengths) for size in subsample_sizes}
    theoretical_probs = {size: [0.0] * len(cot_lengths) for size in subsample_sizes}
    decision_accuracy = [0.0] * len(cot_lengths) # Base accuracy for individual chains per cot_length

    # --- Step 2: Build token-level eval-prompts in batch ---
    # Batch-encode the original prompts once
    batch_prompt_enc   = tokenizer.batch_encode_plus(prompts, add_special_tokens=False)
    tokenized_prompts  = batch_prompt_enc["input_ids"]  # List[List[int]]

    # --- New: Batch‐encode all answer‐prefix strings at once ---
    print("using <|user|> and thinking temperature!")
    if model == "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" or model == "Qwen/Qwen3-32B":
        answer_prefix_strs = [
            #f"\n</think>
            f"\n</think>\n\nAnswer: Node {perm[0]} is in the same component as node "
            for perm in perms
        ]
    elif model == "simplescaling/s1-32B":
        answer_prefix_strs = [
            #f"\n</think>
            f"\n\nAnswer: Node {perm[0]} is in the same component as node "
            for perm in perms
        ]
    else:
        raise ValueError(f"Model {model} not supported")
    batch_ans_pref_enc        = tokenizer.batch_encode_plus(answer_prefix_strs, add_special_tokens=False)
    tokenized_ans_prefixes    = batch_ans_pref_enc["input_ids"]  # List[List[int]]

    all_full_prompt_tokens = []   # will hold the full tokens for eval
    all_candidate_lists    = []   # List[Tuple[str,str]]
    prompt_metadata        = []   # List[dict]
    # Now build every evaluation prompt by reusing the pre‐tokenized pieces
    for i in range(n):
        str_answer    = str(answers[i])
        str_incorrect = str(incorrects[i])
        # fetch the batched answer‐prefix tokens
        tokenized_answer_prefix = tokenized_ans_prefixes[i]

        for j in range(repeats):
            for k, thinking_token in enumerate(cot_lengths):
                base_tokens = tokenized_prompts[i]
                cot_tokens  = (
                    processed_prompts[i][j][:thinking_token]
                    if processed_prompts[i][j] else []
                )
                if type(cot_tokens) == tuple:
                    # print(cot_tokens)
                    cot_tokens = list(cot_tokens)
                full_tokens = base_tokens + cot_tokens + tokenized_answer_prefix
                all_full_prompt_tokens.append(full_tokens)
                all_candidate_lists.append((str_answer, str_incorrect))
                prompt_metadata.append({
                    'original_idx':   i,
                    'repeat_idx':     j,
                    'cot_length_idx': k
                })
    print("decoding evaluation prompts")
    # --- Step 3: Decode all evaluation prompts at once ---
    if not all_full_prompt_tokens:
        return {
            'empirical':   {size: [0.0]*len(cot_lengths) for size in subsample_sizes},
            'theoretical': {size: [0.0]*len(cot_lengths) for size in subsample_sizes},
            'base_accuracy': [0.0]*len(cot_lengths),
        }
    all_eval_prompts = tokenizer.batch_decode(
        all_full_prompt_tokens, skip_special_tokens=True
    )
    print("computing candidate probabilities")
    # Compute probabilities in one go
    all_candidate_probs = compute_candidate_probs_for_prompts(
        all_eval_prompts,
        all_candidate_lists,
        llm
    )
    print("processing results")
    # --- Step 4: Process Results ---
    # Store intermediate results per prompt and cot_length
    # results_by_prompt stores P(correct) for each chain
    results_by_prompt = [[[] for _ in cot_lengths] for _ in range(n)]
    # base_correctness_by_prompt stores the deterministic outcome (0/0.5/1) for base accuracy and theoretical calc
    base_correctness_by_prompt = [[[] for _ in cot_lengths] for _ in range(n)]

    for idx, meta in enumerate(prompt_metadata):
        original_idx = meta['original_idx']
        cot_length_idx = meta['cot_length_idx']
        probs = all_candidate_probs[idx] # Shape (2,) - contains probabilities

        prob_correct = 0.0
        base_correct = 0.0
        if probs.shape[0] == 2:
            p0 = probs[0].item() if torch.is_tensor(probs[0]) else float(probs[0])
            p1 = probs[1].item() if torch.is_tensor(probs[1]) else float(probs[1])
            prob_correct = p0 # Assuming p0 is P(correct)

            # # Determine deterministic base correctness
            if abs(p0 - p1) < 1e-8: # Handle floating point comparison for equality
                base_correct = 0.5
            elif p0 > p1:
                base_correct = 1.0
            # base_correct = p0-p1+1/2
            # base_correct = p0/(p0+p1)
            # else: base_correct remains 0.0 (p0 < p1)
        else:
            print(f"Warning: Unexpected probability shape {probs.shape} for index {idx}. Treating as prob_correct=0, base_correct=0.")

        results_by_prompt[original_idx][cot_length_idx].append(prob_correct)
        base_correctness_by_prompt[original_idx][cot_length_idx].append(base_correct)
    # base_correctness_by_prompt = np.array(base_correctness_by_prompt)
    #normalize base_correctness_by_prompt
    # base_correctness_by_prompt = base_correctness_by_prompt / np.sum(base_correctness_by_prompt, axis=2, keepdims=True)
    print("final metrics")

    # Now calculate final metrics by iterating through cot_lengths
    for k, thinking_token in enumerate(cot_lengths): # Use index k
        # Calculate base accuracy for this cot_length (using the deterministic 0/0.5/1 values)
        total_base_correct_chains = sum(sum(base_correctness_by_prompt[i][k]) for i in range(n))
        # total_base_correct_chains = np.sum(base_correctness_by_prompt, axis=(0,2))[k]
        if n * repeats > 0:
             decision_accuracy[k] = total_base_correct_chains / (n * repeats)
        else:
             decision_accuracy[k] = 0.0 # Avoid division by zero

        # Calculate empirical and theoretical accuracy for each subsample size
        for size_idx, size in enumerate(subsample_sizes):
            if size == 0: continue # Skip size 0

            total_prompt_empirical_acc = 0.0
            total_prompt_theoretical_prob = 0.0

            for i in range(n): # Loop over original prompts
                # chain_correct_probs is the list of probabilities P(correct) for each repeat
                chain_correct_probs = results_by_prompt[i][k] # List of probabilities for this prompt at this cot_length

                # --- Empirical Accuracy Calculation (Sum of Probabilities Voting) ---
                correct_subsample_majority_count = 0.0
                # Number of subsamples to draw for the simulation
                num_subsamples_to_draw = min(1, 10000 // size) if size < repeats and size > 0 else 1

                if repeats >= size and size > 0: # Ensure we can draw a subsample # disabled for efficiency
                    for _ in range(num_subsamples_to_draw):
                        # Select indices for the subsample
                        if size < repeats:
                            subsample_indices = np.random.choice(repeats, size=size, replace=False)
                        else:
                            subsample_indices = list(range(repeats)) # Use all chains if size == repeats

                        # Sum the probabilities P(correct) for the selected chains
                        vote_strength = sum(chain_correct_probs[sub_idx] for sub_idx in subsample_indices)

                        # Determine majority based on the sum of probabilities
                        # Use a small tolerance for float comparison
                        tolerance = 1e-8
                        if vote_strength > (size / 2.0) + tolerance:
                            correct_subsample_majority_count += 1.0
                        elif abs(vote_strength - (size / 2.0)) <= tolerance: # Check for tie
                            correct_subsample_majority_count += 0.5
                        # else: vote_strength < size / 2.0 -> incorrect majority (add 0.0)
                else: # Cannot form a subsample or size is 0
                    correct_subsample_majority_count = 0.0

                prompt_empirical_acc = correct_subsample_majority_count / num_subsamples_to_draw if num_subsamples_to_draw > 0 else 0.0
                total_prompt_empirical_acc += prompt_empirical_acc

                # --- Theoretical Probability Calculation (remains the same) ---
                num_base_correct_chains = sum(base_correctness_by_prompt[i][k]) # Sum of 0, 0.5, 1 values
                k_success_rounded = round(num_base_correct_chains)

                theo_prob = hypergeometric_majority_prob_with_tiebreaking(
                    n_pop=repeats,
                    k_success=k_success_rounded, # Use rounded count based on p0>p1 comparison
                    m_sample=size
                )
                total_prompt_theoretical_prob += theo_prob

            # Average over prompts
            if n > 0:
                empirical_accuracy[size][k] = total_prompt_empirical_acc / n
                theoretical_probs[size][k] = total_prompt_theoretical_prob / n
            else:
                 empirical_accuracy[size][k] = 0.0
                 theoretical_probs[size][k] = 0.0

    # Return the results
    return {
        'empirical': empirical_accuracy,
        'theoretical': theoretical_probs,
        'base_accuracy': decision_accuracy
    }

matplotlib.rcParams.update({'font.size': 16})

cot_lengths = np.linspace(0, test_time_tokens, num_cot_lengths, endpoint=True).astype(int)
subsample_sizes = list(range(1, 2**5 + 1))
repeats = max(subsample_sizes)  # Number of chains per prompt
details = f"d={depth}_s={short}_l={long}_deadend={dead_end}, n={n}, repeats={repeats}, {graph_type} graph"

all_results = sequential_scaling_with_sampling(n, depth, short, long,
                                            cot_lengths, repeats=repeats, 
                                            llm=llm, subsample_sizes=subsample_sizes, dead_end=dead_end,
                                            graph_type=graph_type, path_length=path_length)
# Old figures
if False:
    # Plot results
    plt.figure(figsize=(12, 8))
    test_dataset = BridgeGraph(depth, short, long)
    # Estimate prompt token length (might vary slightly per prompt, use an average or max)
    # Note: The exact prompt length calculation might need refinement if prompts vary significantly
    try:
        prompt_token_length = len(tokenizer.encode(test_dataset.create_prompt()[0], add_special_tokens=False)) + len(tokenizer.encode(f"\nAnswer: Node 0 is in the same component as node ", add_special_tokens=False))
    except IndexError: # Handle case where create_prompt might fail if dataset is empty/invalid
        prompt_token_length = 100 # Estimate if calculation fails

    # Plot theoretical probability curves
    # Ensure cot_lengths is a numpy array for vectorized operations
    cot_lengths_np = np.array(cot_lengths)
    for size in subsample_sizes:
        # Calculate x-axis tokens: (base_prompt + CoT_length) * num_chains_in_vote
        x_axis_tokens = (cot_lengths_np + prompt_token_length) * size
        plt.plot(x_axis_tokens, all_results['theoretical'][size],
                label=f'Majority (n={size})', marker='', linestyle='-')
    print("theoretical results")
    print(all_results['theoretical'])
    # # Plot base accuracy
    # plt.plot(cot_lengths, all_results['base_accuracy'], 
    #          label='Base Accuracy (single chain)', marker='s', linestyle='-')
    plt.xlabel("Total Tokens Processed for Majority Vote") # Updated X-axis label
    plt.ylabel("Accuracy")
    plt.title(f"Sequential Scaling of {model.split('/')[1]} on st-Connectivity with Majority Voting\n({details})")
    plt.legend()
    # Plot empirical accuracy curves
    # for size in subsample_sizes:
    #     plt.plot(cot_lengths, all_results['empirical'][size], 
    #              label=f'Empirical (n={size})', marker='o', linestyle='--')
    plt.grid(True, alpha=0.3)
    try:
        plt.savefig(f"results/token_budget/tokens_{details}.png")
    except Exception as e:
        print(f"Error: Could not save figure: {e}")


    plt.figure(figsize=(12, 8))
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(subsample_sizes)))
    # Plot theoretical probability curves
    for i, size in enumerate(subsample_sizes):
        plt.plot(cot_lengths, all_results['theoretical'][size], 
                label=f'Majority of {size}', marker='', linestyle='-', color=colors[i])

    # # Plot base accuracy
    # plt.plot(cot_lengths, all_results['base_accuracy'], 
    #          label='Base Accuracy (single chain)', marker='s', linestyle='-')
    plt.xlabel("CoT Length")
    plt.ylabel("Accuracy")
    plt.title(f"Scaling of {model.split('/')[1]}\non n={n} labelings of Bridge(depth={depth}, short={short}, long={long}, deadend={dead_end})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    try:
        plt.savefig(f"results/{details}{model.split('/')[1]}_thinking.png")
    except Exception as e:
        print(f"Error: Could not save figure: {e}")


img = [all_results['theoretical'][size] for size in subsample_sizes]
# Plot heatmap first
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(img,
    aspect='auto',
    cmap='viridis',
    origin='lower',
    vmin=0.5, vmax=1.0  # or use np.min(accuracies), np.max(accuracies)
)

# Y-axis ticks labels to subsample_sizes
powers = np.array([2**i for i in [0,2,3,4,5,6]])
ax.set_yticks(ticks=(powers-1), labels=powers)

x_ticks = np.linspace(0, len(cot_lengths)-1, 6, endpoint=True).astype(int)
ax.set_xticks(ticks=x_ticks, labels=cot_lengths[x_ticks])
fig.colorbar(im, label="Accuracy")
# ax.savefig(f"test.png", dpi=300)

# # Define levels and colors
levels = [0.6,0.7,0.8,0.9]
colors = ['white'] * (len(levels) - 1) + ['red']  # Make 0.9 red

# # Plot contours
contour = ax.contour(list(range(len(cot_lengths))), list(range(len(subsample_sizes))), img, levels=levels, colors="white", linewidths=1.5)
plt.clabel(contour, inline=True, fmt="%.1f")


# # Labels and save    
ax.set_xlabel("Sequential Scale")

ax.set_ylabel("Parallel Scale")
ax.set_title(f"Parallel and Sequential Scaling of\n{model.split("/")[1]} {graph_type} graph")
fig.tight_layout()
# plt.savefig(f"test.png", dpi=300)

plt.savefig(f"{model.split('/')[1]}_{details}.pdf", format="pdf")


print("figures made")

print("After offload and cache clear, GPU usage:", torch.cuda.memory_allocated() / 1e9, "GB")