import torch 

# Constants: 
MAX_LEN = 32

class AdditionTokenizer: 

    def __init__(self):
        self.vocab = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "=", "<BOS>", "<EOS>", "<PAD>", "."]
        self.enc_to_char = {i:k for i,k in enumerate(self.vocab)}
        self.char_to_enc = {k:i for i,k in enumerate(self.vocab)}
        self.pad_enc = self.char_to_enc["<PAD>"] 
        self.bos_enc = self.char_to_enc["<BOS>"] 
        self.eos_enc = self.char_to_enc["<EOS>"] 

    def encode(self, s:str, max_len:int, padding:bool): 
        # encoding and adding BOS and EOS
        encoded = [self.bos_enc] + [self.char_to_enc[char] for char in list(s)] + [self.eos_enc]
        if padding: 
            encoded = encoded + [self.pad_enc] * max(0, max_len-len(encoded))
        return encoded

    def decode(self, ids):
        return "".join(self.enc_to_char[int(i)] for i in ids)

class AdditionConfig():

    def __init__(
        self,
        block_size: int,
        vocab_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float,
        use_ln:bool,
        tie_embed:bool
    ):
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.use_ln = use_ln
        self.tie_embed = tie_embed

    def to_dict(self) -> dict:
        return {
            "block_size": self.block_size,
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "dropout": self.dropout, 
            "use_ln": self.use_ln,
            "tie_embed":self.tie_embed
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AdditionConfig":
        return cls(
            block_size=d["block_size"],
            vocab_size=d["vocab_size"],
            n_layer=d["n_layer"],
            n_head=d["n_head"],
            n_embd=d["n_embd"],
            dropout=d["dropout"], 
            use_ln = d["use_ln"],
            tie_embed = d["tie_embed"]
        )
    
def format_equation(
        int1:int, 
        int2:int, 
        result:int | None,
        zero_pad_digits:bool, 
        summand_digits:int, 
        reverse_digits:bool, 
        thinking_tokens:int,

    ):
    int1 = str(int1)
    int2 = str(int2)
    res = str(result)

    if zero_pad_digits:
        int1 = "0"*max(0, summand_digits - len(int1)) + int1 
        int2 = "0"*max(0, summand_digits - len(int2)) + int2
        res = "0"*max(0, summand_digits + 1 - len(res)) + res
    if reverse_digits: 
        int1 = int1[::-1]
        int2 = int2[::-1]
        res = res[::-1]
    equation = int1 + "+" + int2 + "."*max(0, thinking_tokens) + "="
    return equation if (result is None) else equation + res

def generate_random_additions(
        tokenizer,
        batch_size:int, 
        max_len:int,
        low:int, 
        high:int,
        zero_pad_digits:bool, 
        summand_digits:int, 
        reverse_digits:bool, 
        thinking_tokens:int
    ):
    xs = torch.randint(low=low, high=high, size=(batch_size, 2))
    ys = torch.sum(xs, axis=1)
    encoded = []
    for i in range(batch_size):

        # Generating the equation-string. 
        int1 = int(xs[i, 0])
        int2 = int(xs[i, 1])
        result = int(ys[i])
        equation = format_equation(
            int1=int1, 
            int2=int2, 
            result=result,
            zero_pad_digits=zero_pad_digits, 
            summand_digits=summand_digits, 
            reverse_digits=reverse_digits, 
            thinking_tokens=thinking_tokens
        )
        # Applying tokenizer
        tokenized = tokenizer.encode(equation, padding=True, max_len=max_len)
        encoded.append(tokenized)
    X = torch.tensor(encoded, dtype=torch.long)
    return X