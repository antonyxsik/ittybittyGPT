import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Int, Float
import transformers

# these all allow for a nice way to save the model and load it back up
from muutils.json_serialize import SerializableDataclass, serializable_dataclass, serializable_field
from zanj.torchutil import ConfiguredModel, set_config_class
from zanj import ZANJ

# GPT config class
@serializable_dataclass(kw_only=True)
class GPTConfig(SerializableDataclass):
    """Here we configure the dimensions of
    our model. We'll set the defaults as the 
     dims from GPT2 """
    d_model: int = serializable_field(default=768) # dimension of residual stream, the vectors it internally passes around
    d_vocab: int = serializable_field(default=50257) # defines the number of different tokens that can be represented as inputs (vocabulary size)
    n_context: int = serializable_field(default=1024) # maximum sequence length (context window size)
    n_blocks: int = serializable_field(default=12) # number of transformer blocks, frequently called n_layers but I don't like that
    n_head: int = serializable_field(default=12) # number of attention heads
    head_bias: bool = serializable_field(default=True) # whether to use bias in attention heads
    mlp_expansion: int = serializable_field(default=4) # expansion factor in MLP (they go from small to big to small, this is how many times bigger the middle layer is)

    tokenizer: transformers.PreTrainedTokenizer = serializable_field(
        default=transformers.AutoTokenizer.from_pretrained("gpt2"),
        serialization_fn = lambda x: x.name_or_path,
        loading_fn = lambda x: transformers.AutoTokenizer.from_pretrained(x["tokenizer"]),
    ) # tokenizer for the model

    # model dimension must be divisible by number of heads
    @property
    def d_head(self):
        assert self.d_model % self.n_head == 0, f"'{self.d_model = }' must be divisible by '{self.n_head = }': {self.d_model} % {self.n_head} == {self.d_model % self.n_head}"
        return self.d_model // self.n_head
    
    @property
    def params_shapes(self) -> dict:
        return dict(
            token_embeddings=(self.d_vocab, self.d_model),
            positional_embeddings=(self.n_context, self.d_model),
            attention_weights=(
                self.n_blocks,
                4,
                self.d_model,
                self.d_model,
            ),
            attention_bias=(
                self.n_blocks,
                int(self.head_bias),
                self.d_model,
            ),
            mlp_weights=(
                self.n_blocks,
                2,
                self.d_model,
                self.d_model * self.mlp_expansion,
            ),
            mlp_bias=(
                self.n_blocks,
                self.mlp_expansion + 1,
                self.d_model,
            ),
            block_layernorms=(
                self.n_blocks,
                2,
                2,
                self.d_model,
            ),
            output_layernorm=(2, self.d_model),
            lm_head=(self.d_model, self.d_vocab),
        )
    
    @property
    def params_numel(self) -> dict:
        return {
            k: int(torch.tensor(v).prod())
            for k, v in self.params_shapes.items()
        }

    # will return the total number of parameters in the model
    @property
    def n_params(self) -> int:
        return sum([v for v in self.params_numel.values()])
    

# attention head
@set_config_class(GPTConfig)
class AttentionHead(ConfiguredModel[GPTConfig]):

    def __init__(self, config: GPTConfig):
        super().__init__(zanj_model_config=config)

        # store dimensions
        self.n_head: int = config.n_head
        self.d_model: int = config.d_model
        self.n_context: int = config.n_context

        # concatenating the outputs of the heads should give us d_model, but this check is done in GPTConfig
        self.d_head: int = config.d_head
        self.head_bias: bool = config.head_bias

        # magic coefficient for scaling the dot product of the query and key in the attention calculation
        self.sqrt_dim: float = 1.0 / math.sqrt(self.d_head)
    

        # key, query, value projections
        self.W_K: nn.Module = nn.Linear(self.d_model, self.d_head, bias = self.head_bias)
        self.W_Q: nn.Module = nn.Linear(self.d_model, self.d_head, bias = self.head_bias)
        self.W_V: nn.Module = nn.Linear(self.d_model, self.d_head, bias = self.head_bias)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        # `register_buffer` means it's not a trainable parameter
        # the point here is to not allow the model to "look into the future" when making predictions
        self.register_buffer(
            "causal_mask", 
            torch.tril(
                torch.ones(config.n_context, config.n_context)
            )
            .view(1, 1, config.n_context, config.n_context)
        )


    def forward(self, x: Float[torch.Tensor, "batch n_ctx d_model"]) -> Float[torch.Tensor, "batch n_ctx d_head"]:
        assert x.ndim == 3, str(x.shape)
        B, n_ctx, d_model = x.shape # batch size, sequence length, embedding dimensionality (d_model)
        assert d_model == self.d_model, str(x.shape)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q: Float[torch.Tensor, "batch n_ctx d_head"] = self.W_Q(x)
        k: Float[torch.Tensor, "batch n_ctx d_head"] = self.W_K(x)
        v: Float[torch.Tensor, "batch n_ctx d_head"] = self.W_V(x)

        # self-attention
        att = (q @ k.transpose(-2, -1)) * self.sqrt_dim
        
        # autoregressive (causal) masking
        att = att.masked_fill(
            self.causal_mask[:,:n_ctx,:n_ctx] == 0, 
            float('-inf'),
        )

        # softmax
        att = F.softmax(att, dim=-1)

        # apply the self-attention to the values
        output = att @ v
        return output.view(B, n_ctx, self.d_head)
    

#multi headed attention
@set_config_class(GPTConfig)
class MultiHeadedAttention(ConfiguredModel[GPTConfig]):
    def __init__(self, config: GPTConfig):
        super().__init__(zanj_model_config=config)
        self.n_head: int = config.n_head
        self.d_head: int = config.d_model // config.n_head
        self.d_model: int = config.d_model

        # attention heads from previous class
        self.attention_heads: nn.ModuleList = nn.ModuleList([
            AttentionHead(config) 
            for _ in range(self.n_head)
        ])

        # output projection
        self.W_O: nn.Module = nn.Linear(self.d_model, self.d_model)


    def forward(self, x: Float[torch.Tensor, "batch n_ctx d_model"]) -> Float[torch.Tensor, "batch n_ctx d_model"]:
        assert x.ndim == 3, str(x.shape)
        # apply all attention heads and concatenate their outputs
        # note: in reality, you would do this all in one tensor
        # we split the attention heads up to make it easier to understand
        att = torch.cat(
            [
                head(x) 
                for head in self.attention_heads
            ],
            dim=-1,
        )
        assert len(att.shape) == 3, str(att.shape)

        # output projection
        output = self.W_O(att)
        assert output.shape == x.shape, str(output.shape)
        return output


#transformer block
@set_config_class(GPTConfig)
class TransformerBlock(ConfiguredModel[GPTConfig]):
    def __init__(self, config: GPTConfig):
        super().__init__(zanj_model_config=config)

        # layernorm, attention, another layernorm, mlp
        self.ln_1: nn.Module = nn.LayerNorm(config.d_model)
        self.attention: nn.Module = MultiHeadedAttention(config)
        self.ln_2: nn.Module = nn.LayerNorm(config.d_model)
        self.mlp: nn.Module = nn.Sequential(
            nn.Linear(config.d_model, config.mlp_expansion * config.d_model),
            nn.GELU(),
            nn.Linear(config.mlp_expansion * config.d_model, config.d_model),
        )

    def forward(self, x: Float[torch.Tensor, "batch n_ctx d_model"]) -> Float[torch.Tensor, "batch n_ctx d_model"]:
        z = x + self.attention(self.ln_1(x))
        return z + self.mlp(self.ln_2(z))
      


# GPT model (putting it all together)
@set_config_class(GPTConfig)
class GPT(ConfiguredModel[GPTConfig]):
    def __init__(self, config: GPTConfig):
        super().__init__(zanj_model_config=config)

        self.config: GPTConfig = config
        self.tokenizer: transformers.PreTrainedTokenizer = config.tokenizer
        assert config.d_vocab >= self.tokenizer.vocab_size

        # token and positional embeddings
        self.token_embeddings: nn.Module = nn.Embedding(config.d_vocab, config.d_model)
        self.positional_embeddings: nn.Module = nn.Embedding(config.n_context, config.d_model)

        # transformer
        self.transformer_blocks: nn.ModuleList = nn.ModuleList([
            TransformerBlock(config) 
            for _ in range(config.n_blocks)
        ])

        # language model head
        self.ln_f: nn.Module = nn.LayerNorm(config.d_model)
        self.lm_head: nn.Module = nn.Linear(config.d_model, config.d_vocab, bias=False)

    def forward(
            self, 
            x: Int[torch.Tensor, "batch n_ctx"],
            targets: Int[torch.Tensor, "batch n_ctx"]|None = None,
        ) -> tuple:
        """returns a tuple of (logits, loss) where loss=None if targets is None"""
        assert x.ndim == 2, str(x.shape)

        # calculate token and positional embeddings and sum them
        x_res: Float[torch.Tensor, "batch n_ctx d_model"] = self.token_embeddings(x) + self.positional_embeddings(torch.arange(x.size(1), device=x.device))

        assert x_res.ndim == 3, str(x.shape)

        # transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            x_res = block(x_res)

        # language model head
        logits: Float[torch.Tensor, "batch n_ctx d_vocab"] = self.lm_head(self.ln_f(x_res))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.transpose(1, 2),
                targets,
                ignore_index=-1,
            )

        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str|list[int]|Int[torch.Tensor, "* n_ctx"],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> str:

        # convert prompt to string and tensor versions
        prompt_str: str
        prompt_tensor: Int[torch.Tensor, "1 n_ctx"]
        if isinstance(prompt, str):
            prompt_str = prompt
            prompt_tensor = torch.tensor(self.tokenizer.encode(prompt_str), dtype=torch.long).unsqueeze(0) # add batch dim
        elif isinstance(prompt, list):
            prompt_str = self.tokenizer.decode(prompt)
            prompt_tensor = torch.tensor(prompt, dtype=torch.long).unsqueeze(0) # add batch dim
        elif isinstance(prompt, torch.Tensor):
            if prompt.ndim == 1:
                prompt = prompt.unsqueeze(0) # add batch dim
            assert prompt.ndim == 2

            prompt_str = self.tokenizer.decode(prompt[0].tolist())
            prompt_tensor = prompt
        else:
            raise ValueError(f"prompt must be a string, list of ints, or PyTorch tensor")
        
        # check tensor dims
        assert isinstance(prompt_str, str) 
        assert isinstance(prompt_tensor, torch.Tensor)
        assert prompt_tensor.ndim == 2 
        assert prompt_tensor.shape[0] == 1

        #  device
        prompt_tensor = prompt_tensor.to(self.device)

        # pad the prompt if necessary
        if prompt_tensor.shape[1] < self.config.n_context:
            prompt_tensor = F.pad(prompt_tensor, (0, self.config.n_context - prompt_tensor.shape[1]), value=self.tokenizer.pad_token_id)

        assert prompt_tensor.shape[1] == self.config.n_context

        # iterate until max_new_tokens is reached, or an end-of-sequence token is generated
        completions: list[int] = list()
        for _ in range(max_new_tokens):
            # truncate sequence to block size
            prompt_len: int = prompt_tensor.shape[1]
            if prompt_len > self.config.n_context:
                prompt_tensor = prompt_tensor[:, -self.config.n_context:]

            # forward the model to get the logits for the index in the sequence
            logits, _ = self(prompt_tensor)

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((prompt_tensor, idx_next), dim=1)

            # append the token to the running completions
            completions.append(int(idx_next[0, 0]))

            # check if end of sequence token is generated
            if idx_next == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(completions)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    @property
    def device(self) -> torch.device:
        device_set: set[torch.device] = set(p.device for p in self.parameters())
        assert len(device_set) == 1, device_set
        return next(iter(device_set))