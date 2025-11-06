import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        # no bias parameter

    def forward(self, x):
        # F.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5)
        return F.layer_norm(x, self.weight.shape, self.weight, None, 1e-5)
    
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(
        self, 
        x, 
        return_attn=False,
        scale_mask: torch.Tensor | None = None,
        qk_mask : torch.Tensor | None = None, 
        v_mask: torch.Tensor | None = None, 
        c_proj_ablate: bool = False
        ):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B,H,T,hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # V-mask: blocks writes from certain source positions. 
        if v_mask is not None:
            # Check (H, T). 
            if v_mask.shape != (self.n_head, T):
                raise ValueError(
                    f"v_mask must have shape (H,T)=({self.n_head},{T}), got {tuple(v_mask.shape)}"
                )
            # Transform --> (1,H,T,1)
            v_mask = v_mask.view(1, self.n_head, T, 1)
            v = v * v_mask.to(v.dtype).to(v.device)

        # The usual attention-pattern
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B,H,T,T)
        att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device), 1).bool(), float('-inf'))

        # QK-masking
        if qk_mask is not None: 
            if qk_mask.shape != (self.n_head, T, T):
                raise ValueError(
                    f"v_mask must have shape (H,T,T)=({self.n_head},{T},{T}), got {tuple(qk_mask.shape)}"
                )
            att = att.masked_fill(qk_mask, float('-inf'))
            print("attention-mask:", att[0, 0, -1])

        # The softmax and value-transformation
        attn_probs = F.softmax(att, dim=-1)
        y = attn_probs @ v  # (B,H,T,hs)

        # Scaing head outputs
        if scale_mask is not None:
            if scale_mask.shape != (self.n_head):
                raise ValueError(
                    f"scale_mask must have shape (H,)=({self.n_head}), got {tuple(scale_mask)}"
                )
            scale_mask = scale_mask.view(1, -1, 1, 1)
            scale_mask = scale_mask.view(1, -1, 1, 1).to(y.dtype).to(y.device)
            y = y * scale_mask

        # Projecting out. 
        y = y.transpose(1, 2).contiguous().view(B, T, C)   # (B,T,C)
        if not c_proj_ablate:
            y = self.c_proj(y)
        return (y, attn_probs) if return_attn else (y, None)  # Put attn_probs back here!

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu    = nn.GELU()
        self.relu    = nn.ReLU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        # x = self.relu(x)
        x = self.relu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd) if config.use_ln else nn.Identity()
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd) if config.use_ln else nn.Identity()
        self.mlp = MLP(config)

    def forward(
        self, 
        x, 
        return_attn=False, 
        scale_mask=None, 
        qk_mask=None, 
        v_mask=None,
        c_proj_ablate:bool=False
    ):
        y, attn_probs = self.attn(
            x=self.ln_1(x), 
            return_attn=return_attn, 
            scale_mask=scale_mask, 
            qk_mask=qk_mask, 
            v_mask=v_mask, 
            c_proj_ablate=c_proj_ablate,
        )
        x = x + y + self.mlp(self.ln_2(x + y))
        return (x, attn_probs) if return_attn else (x, None)

class CausalTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd) if config.use_ln else nn.Identity(),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embed:
            # https://paperswithcode.com/method/weight-tying
            self.transformer.wte.weight = self.lm_head.weight 

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def calc_loss(self, logits, targets): 
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return loss
    
    def forward(
        self, 
        idx: torch.Tensor, 
        return_hidden:bool=False,
        scale_masks: dict[int, torch.Tensor] = {}, # per-layer mask (H,)
        qk_masks: dict[int, torch.Tensor] = {},    # per-layer mask (H,T,T)
        v_masks: dict[int, torch.Tensor] = {},     # per-layer mask (H,T)
        c_proj_ablate:bool=False
    ):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Beginning the forward with embeddings. 
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))

        # For book-keeping 
        hidden_states = [x.detach()] if return_hidden else None
        attn_probs_all = [] if return_hidden else None

        # Passing through the blocks
        for block_i, block in enumerate(self.transformer.h):
            sm = scale_masks.get(block_i, None)
            qkm = qk_masks.get(block_i, None)
            vm  = v_masks.get(block_i, None)

            x, probs = block(
                x,
                return_attn=return_hidden,
                scale_mask=sm,
                qk_mask=qkm,
                v_mask=vm, 
                c_proj_ablate=c_proj_ablate
            )

            if return_hidden:
                attn_probs_all.append(probs.detach())
                hidden_states.append(x.detach())

        # Last layer-norm and unembedding. 
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Formatting
        if return_hidden:
            return {"logits": logits, "hidden_states": hidden_states, "attn_probs": attn_probs_all}
        else:
            return logits
    
    def generate(
        self, 
        ids:torch.Tensor, 
        max_new_tokens:int, 
        temperature=1.0, 
        ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):

            # if the sequence context is growing too long we must crop it at block_size
            ids_cond = ids if ids.size(1) <= self.config.block_size else ids[:, -self.config.block_size:]
            logits = self.forward(
                ids_cond, 
            )
            last_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(last_token_logits, dim=-1)
            ids_next = torch.multinomial(probs, num_samples=1)
            ids = torch.cat((ids, ids_next), dim=1)
        return ids