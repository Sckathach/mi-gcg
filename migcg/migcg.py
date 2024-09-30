import logging
from dataclasses import dataclass
from typing import List

from torch.nn import CosineSimilarity
from tqdm import tqdm
import gc

import torch
import torch.nn.functional as F
from torch import Tensor
from transformer_lens import HookedTransformer

from migcg.gcg import GCGResult
from migcg.utils import get_nonascii_toks, find_executable_batch_size
from migcg.refusal_demo import RefusalLab

logger = logging.getLogger("migcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class GCGMIConfig:
    num_steps: int = 250
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0  # Only 0 accepted
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"


@dataclass
class GCGMIResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]


class AttackBufferMI:
    def __init__(self, size: int):
        self.buffer = []  # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]

    def log_buffer(self, model: HookedTransformer):
        message = "buffer:"
        for loss, ids in self.buffer:
            optim_str = model.to_string(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | string: {optim_str}"
        logger.info(message)

def sample_ids_from_grad(
        ids: Tensor,
        grad: Tensor,
        search_width: int,
        topk: int = 256,
        n_replace: int = 1,
        not_allowed_ids: Tensor = False,
):
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device)
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

    return new_ids

class GCGMI(RefusalLab):
    def __init__(self, model: HookedTransformer, config: GCGMIConfig, device: str = "cuda"):
        super().__init__(model)
        self.config = config

        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(
            model.tokenizer,
            device=device
        )

        self.prefix_cache = None
        self.stop_flag = False
        self.prompt_embeds = None

    def init_prompt(self, prompt):
        self.prompt_embeds = torch.embedding(self.model.W_E, self.model.to_tokens(prompt))

    def init_buffer(self):
        model = self.model
        config = self.config

        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")

        buffer = AttackBufferMI(config.buffer_size)
        init_buffer_ids = model.to_tokens(config.optim_str_init, prepend_bos=False)
        buffer.add(1, init_buffer_ids)
        buffer.log_buffer(model)
        logger.info("Initialized attack buffer.")

        return buffer

    def compute_gradients(self, adversarial_one_hot):
        self.model.train()

        adversarial_embeds = adversarial_one_hot @ self.model.W_E

        if adversarial_one_hot.grad is not None:
            adversarial_one_hot.grad.zero_()

        full_embeds = torch.cat([self.prompt_embeds, adversarial_embeds], dim=1)

        output = self.model.forward(full_embeds, start_at_layer=0, stop_at_layçàer=15).squeeze()[-1]
        loss_similarity = self.cos(output, self.refusal_dir)

        loss = loss_similarity

        loss.backward(retain_graph=True)

        return adversarial_one_hot.grad.detach()

    def compute_candidates_loss(
            self,
            search_batch_size: int,
            input_embeds: Tensor,
    ) -> Tensor:
        """Computes the loss on all candidate token id sequences.

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            input_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
        """
        all_loss = []
        cos = CosineSimilarity(dim=1)

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i + search_batch_size]

                outputs = self.model.forward(input_embeds_batch, start_at_layer=0, stop_at_layer=15).squeeze(0)[:, -1, :]
                loss = cos(outputs, self.refusal_dir)

                all_loss.append(loss)

                # Clear cache
                del outputs
                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)

    def run(self):
        config = self.config
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()
        optim_ids_onehot = F.one_hot(optim_ids, self.model.cfg.d_vocab).half().detach()
        optim_ids_onehot.requires_grad = True

        losses = []
        optim_strings = []

        for _ in tqdm(range(config.num_steps)):
            optim_ids_onehot_grad = self.compute_gradients(optim_ids_onehot)
            sampled_ids = sample_ids_from_grad(
                optim_ids.squeeze(0),
                optim_ids_onehot_grad.squeeze(0),
                config.search_width,
                config.topk,
                config.n_replace,
                not_allowed_ids=self.not_allowed_ids,
            )

            new_search_width = sampled_ids.shape[0]

            input_embeds = torch.cat([
                self.prompt_embeds.repeat(new_search_width, 1, 1),
                torch.embedding(self.model.W_E, sampled_ids)
            ], dim=1)

            loss = find_executable_batch_size(self.compute_candidates_loss, new_search_width)(input_embeds)
            current_loss = loss.min().item()

            optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

            losses.append(current_loss)
            if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = self.model.to_string(optim_ids)
            optim_strings.append(optim_str)

            buffer.log_buffer(self.model)

        min_loss_index = losses.index(min(losses))

        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
        )

        return result


























