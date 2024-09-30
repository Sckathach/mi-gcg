import functools
import gc
import io
from typing import List

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from datasets import load_dataset
from jaxtyping import Float, Int
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer


def get_harmful_instructions():
    url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'
    response = requests.get(url)

    dataset = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    instructions = dataset['goal'].tolist()

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmless_instructions():
    hf_path = 'tatsu-lab/alpaca'
    dataset = load_dataset(hf_path)

    # filter for instructions that do not have inputs
    instructions = []
    for i in range(len(dataset['train'])):
        if dataset['train'][i]['input'].strip() == '':
            instructions.append(dataset['train'][i]['instruction'])

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def direction_ablation_hook(
        activation: Float[Tensor, "... d_act"],  # noqa: F722
        direction: Float[Tensor, "d_act"]  # noqa: F821
):
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj


def get_orthogonalized_matrix(
        matrix: Float[Tensor, '... d_model'],  # noqa: F722
        vec: Float[Tensor, 'd_model']  # noqa: F821
) -> Float[Tensor, '... d_model']:  # noqa: F722
    proj = einops.einsum(matrix, vec.view(-1, 1), '... d_model, d_model single -> ... single') * vec
    return matrix - proj


class RefusalLab:
    def __init__(self, model, layer=14, pos=-1, n_inst_train=32):
        self.model = model
        self.qwen_chat_template = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant"

        self.harmful_inst_train, self.harmful_inst_test = get_harmful_instructions()
        self.harmless_inst_train, self.harmless_inst_test = get_harmless_instructions()

        self.tokenize_instructions_fn = functools.partial(self.tokenize_instructions_qwen_chat,
                                                          tokenizer=self.model.tokenizer)

        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.n_inst_train = n_inst_train
        self.pos = pos
        self.layer = layer
        self.refusal_dir = self.get_refusal_dir(layer).to("cuda")

    def get_setup_for_layer(self, layer):
        self.layer = layer
        self.refusal_dir = self.get_refusal_dir(layer).to("cuda")

    def tokenize_instructions_qwen_chat(
            self,
            tokenizer: AutoTokenizer,
            instructions: List[str]
    ) -> Int[Tensor, 'batch_size seq_len']:  # noqa: F722
        prompts = [self.qwen_chat_template.format(instruction=instruction) for instruction in instructions]
        return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").input_ids

    def _generate_with_hooks(
            self,
            toks: Int[Tensor, 'batch_size seq_len'],  # noqa: F722
            max_tokens_generated: int = 64,
            fwd_hooks=None,
    ) -> List[str]:
        if fwd_hooks is None:
            fwd_hooks = []

        all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long,
                               device=toks.device)
        all_toks[:, :toks.shape[1]] = toks

        for i in range(max_tokens_generated):
            with self.model.hooks(fwd_hooks=fwd_hooks):
                logits = self.model(all_toks[:, :-max_tokens_generated + i])
                next_tokens = logits[:, -1, :].argmax(dim=-1)  # greedy sampling (temperature=0)
                all_toks[:, -max_tokens_generated + i] = next_tokens

        return self.model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=True)

    def get_generations(
            self,
            instructions: List[str],
            fwd_hooks=None,
            max_tokens_generated: int = 64,
            batch_size: int = 4,
    ) -> List[str]:
        if fwd_hooks is None:
            fwd_hooks = []

        generations = []

        for i in tqdm(range(0, len(instructions), batch_size)):
            toks = self.tokenize_instructions_fn(instructions=instructions[i:i + batch_size])
            generation = self._generate_with_hooks(
                toks,
                max_tokens_generated=max_tokens_generated,
                fwd_hooks=fwd_hooks,
            )
            generations.extend(generation)

        return generations

    def scan_to_cpu(self, tokens, layer=None, pos=None):
        if layer is None:
            layer = self.layer
        if pos is None:
            pos = self.pos

        logits, cache = self.model.run_with_cache(tokens, names_filter=lambda hook_name: "resid" in hook_name)
        cpu_scan = cache["resid_pre", layer][:, pos, :].cpu()
        del cache, logits
        gc.collect()
        torch.cuda.empty_cache()
        return cpu_scan

    def scan_inst(self, instructions, n_inst_train=None, layer=None, pos=None):
        if n_inst_train is None:
            n_inst_train = self.n_inst_train
        if layer is None:
            layer = self.layer
        if pos is None:
            pos = self.pos

        tokens = self.tokenize_instructions_fn(instructions=instructions[:n_inst_train])
        scan = self.scan_to_cpu(tokens, layer=layer, pos=pos)
        return scan

    def get_refusal_dir(self, layer=None, n_inst_train=None, pos=None):
        if n_inst_train is None:
            n_inst_train = self.n_inst_train
        if layer is None:
            layer = self.layer
        if pos is None:
            pos = self.pos

        harmful_mean_act = self.scan_inst(
            self.harmful_inst_train,
            n_inst_train=n_inst_train,
            layer=layer,
            pos=pos
        ).mean(dim=0)
        harmless_mean_act = self.scan_inst(
            self.harmless_inst_train,
            n_inst_train=n_inst_train,
            layer=layer,
            pos=pos
        ).mean(dim=0)

        refusal_dir = harmful_mean_act - harmless_mean_act
        refusal_dir = refusal_dir / refusal_dir.norm()
        return refusal_dir

    def chat(self, sentence):
        return self.get_generations([sentence])[0]

    def get_refusal_value(self, sentence):
        if isinstance(sentence, str):
            tokens = self.model.to_tokens(sentence, prepend_bos=False)
        else:
            tokens = sentence
        embed = torch.embedding(self.model.W_E, tokens)
        output = self.model.forward(embed, start_at_layer=0, stop_at_layer=15).squeeze(0)[-1]

        return self.cos(output, self.refusal_dir).item()

    def plot_scan(self, layer=None, n_inst_train=None, pos=None):
        if n_inst_train is None:
            n_inst_train = self.n_inst_train
        if layer is None:
            layer = self.layer
        if pos is None:
            pos = self.pos

        harmless_scan = self.scan_inst(self.harmless_inst_train, n_inst_train=n_inst_train, layer=layer, pos=pos)
        harmful_scan = self.scan_inst(self.harmful_inst_train, n_inst_train=n_inst_train, layer=layer, pos=pos)
        combined_scans = torch.vstack((harmless_scan, harmful_scan))

        # Convert combined_scans to a numpy array (PCA from scikit-learn requires this)
        combined_scans_array = combined_scans.detach().numpy()

        # Apply PCA to reduce dimensionality (e.g., to 2D)
        pca = PCA(n_components=2)
        reduced_scans_array = pca.fit_transform(combined_scans_array)

        # Split the reduced activations back into separate arrays for A and B
        a_reduced = reduced_scans_array[:len(harmless_scan), :]
        b_reduced = reduced_scans_array[len(harmless_scan):, :]

        # Convert the numpy arrays back to PyTorch tensors
        a_reduced_torch = torch.tensor(a_reduced)
        b_reduced_torch = torch.tensor(b_reduced)

        fig, ax = plt.subplots()
        # Create a scatter plot with different colors for each set
        ax.scatter(a_reduced_torch[:, 0], a_reduced_torch[:, 1], color='darkorange', label='Harmless queries')
        ax.scatter(b_reduced_torch[:, 0], b_reduced_torch[:, 1], color='darkorchid', label='Harmful queries')

        # Add labels and title to the plot
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(f"Layer {layer} Activations (PCA with 2 components)")
        plt.legend()

        fig.patch.set_alpha(0)  # Transparent figure background
        ax.set_facecolor('none')  # Transparent plot background
        # Display the plot
        plt.show()

    def plot_refusal_similarity(self):
        harmless_similarities = []
        harmful_similarities = []

        for layer in range(3, self.model.cfg.n_layers):
            harmful_scan = self.scan_inst(self.harmful_inst_train, layer=layer)
            harmless_scan = self.scan_inst(self.harmless_inst_train, layer=layer)
            refusal_dir = harmful_scan.mean(dim=0) - harmless_scan.mean(dim=0)
            refusal_dir = refusal_dir / refusal_dir.norm()

            harmful_similarities += [self.cos(refusal_dir, harmful_scan).mean()]
            harmless_similarities += [self.cos(refusal_dir, harmless_scan).mean()]

        x = np.arange(3, self.model.cfg.n_layers)

        fig, ax = plt.subplots()
        ax.plot(x, harmful_similarities, color="darkorchid", label="Harmful queries")
        ax.plot(x, harmless_similarities, color="darkorange", label="Harmless queries")
        fig.patch.set_alpha(0)  # Transparent figure background
        ax.set_facecolor('none')  # Transparent plot background
        plt.xlabel('Layers')
        plt.ylabel('Cosine similarity')
        plt.title('Cosine similarity with the refusal direction')
        plt.xticks(x)
        plt.axhline(y=0, color='k', linestyle='--')
        plt.axvline(x=15, color='k', linestyle='--')
        plt.legend()
        plt.show()
