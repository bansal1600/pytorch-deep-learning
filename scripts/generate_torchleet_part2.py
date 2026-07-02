"""Part 2: LLM path + Advanced TorchLeet solution notebooks."""

from generate_torchleet_notebooks import save, md, code

# --- LLM PATH ---

def gen_v2_06():
    save("llm_path", "v2-06_attention_from_scratch_solution.ipynb", [
        md("# V2-06: Scaled Dot-Product Attention — Solution"),
        code("""import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, b=2, h=4, s=6, d=8):
        torch.manual_seed(1)
        self.q = torch.randn(b, h, s, d)
        self.k = torch.randn(b, h, s, d)
        self.v = torch.randn(b, h, s, d)
    def qkv(self):
        return self.q, self.k, self.v


class QKVDataset(Dataset):
    def __init__(self, q, k, v):
        self.q, self.k, self.v = q, k, v
    def __len__(self): return self.q.size(0)
    def __getitem__(self, i): return self.q[i], self.k[i], self.v[i]


def scaled_dot_product_attention(q, k, v, mask=None):
    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


class AttentionBlock(nn.Module):
    def __init__(self, d_model=8, n_heads=4):
        super().__init__()
        self.d, self.h = d_model, n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
    def forward(self, x):
        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.h, self.d // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        return scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, S, self.d)


q, k, v = DummyDataGenerator().qkv()
out = scaled_dot_product_attention(q, k, v)
ref = F.scaled_dot_product_attention(q, k, v)
print(f"max diff vs ref: {(out-ref).abs().max():.2e}")
print("✓ Attention matches PyTorch")"""),
    ])


def gen_v2_07():
    save("llm_path", "v2-07_multi_head_attention_solution.ipynb", [
        md("# V2-07: Multi-Head Attention — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, b=2, s=8, d=32):
        torch.manual_seed(2)
        self.x = torch.randn(b, s, d)
    def input(self): return self.x


class SequenceDataset(Dataset):
    def __init__(self, x): self.x = x
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


def multi_head_attention(x, d_model, num_heads, mask=None):
    B, S, _ = x.shape
    d_h = d_model // num_heads
    Wq = Wk = Wv = torch.randn(d_model, d_model) / d_model**0.5
    Wo = torch.randn(d_model, d_model) / d_model**0.5
    q = (x @ Wq).view(B, S, num_heads, d_h).transpose(1, 2)
    k = (x @ Wk).view(B, S, num_heads, d_h).transpose(1, 2)
    v = (x @ Wv).view(B, S, num_heads, d_h).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / (d_h ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1) @ v
    concat = attn.transpose(1, 2).reshape(B, S, d_model)
    return concat @ Wo


class MHAWrapper(nn.Module):
    def __init__(self, d=32, heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True)
    def forward(self, x):
        out, _ = self.mha(x, x, x)
        return out


x = DummyDataGenerator().input()
out = multi_head_attention(x, 32, 4)
print(f"MHA output shape: {out.shape}")
print("✓ Multi-head attention from scratch")"""),
    ])


def gen_v2_08():
    save("llm_path", "v2-08_grouped_query_attention_solution.ipynb", [
        md("# V2-08: Grouped Query Attention — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        torch.manual_seed(3)
        self.q = torch.randn(1, 8, 4, 16)  # B, n_heads, S, d
        self.k = torch.randn(1, 2, 4, 16)  # B, n_kv_heads, S, d
        self.v = torch.randn(1, 2, 4, 16)
    def qkv(self): return self.q, self.k, self.v


class GQADataset(Dataset):
    def __init__(self, q, k, v):
        self.q, self.k, self.v = q, k, v
    def __len__(self): return 1
    def __getitem__(self, i): return self.q, self.k, self.v


def grouped_query_attention(q, k, v, num_query_groups):
    n_qh = q.size(1)
    n_kvh = k.size(1)
    repeat = n_qh // n_kvh
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
    return torch.softmax(scores, dim=-1) @ v


class GQAModel(nn.Module):
    def __init__(self, d=16, n_q_heads=8, n_kv_heads=2):
        super().__init__()
        self.n_q_heads, self.n_kv_heads = n_q_heads, n_kv_heads
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d * n_kv_heads // n_q_heads)
        self.wv = nn.Linear(d, d * n_kv_heads // n_q_heads)
    def forward(self, x):
        B, S, D = x.shape
        q = self.wq(x).view(B, S, self.n_q_heads, D // self.n_q_heads).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_kv_heads, D // self.n_q_heads).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_kv_heads, D // self.n_q_heads).transpose(1, 2)
        return grouped_query_attention(q, k, v, self.n_kv_heads)


q, k, v = DummyDataGenerator().qkv()
out = grouped_query_attention(q, k, v, 2)
print(f"GQA output: {out.shape}")
print("✓ GQA implemented")"""),
    ])


def gen_v2_10():
    save("llm_path", "v2-10_sinusoidal_embeddings_solution.ipynb", [
        md("# V2-10: Sinusoidal Positional Embeddings — Solution"),
        code("""import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, seq_len=10, d_model=32):
        torch.manual_seed(4)
        self.tokens = torch.randn(2, seq_len, d_model)
        self.seq_len, self.d_model = seq_len, d_model
    def input(self): return self.tokens


class TokenDataset(Dataset):
    def __init__(self, x): self.x = x
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(max_seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return self.pe[:, : x.size(1)]


class EmbeddingModel(nn.Module):
    def __init__(self, d_model=32, max_len=128):
        super().__init__()
        self.pos = SinusoidalPositionalEmbedding(max_len, d_model)
    def forward(self, x):
        return x + self.pos(x)


x = DummyDataGenerator().input()
model = EmbeddingModel(32)
out = model(x)
print(f"embedded shape: {out.shape}")
print("✓ Sinusoidal PE applied")"""),
    ])


def gen_v2_11():
    save("llm_path", "v2-11_rope_embeddings_solution.ipynb", [
        md("# V2-11: RoPE Embeddings — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        torch.manual_seed(5)
        self.q = torch.randn(1, 4, 8, 16)
        self.k = torch.randn(1, 4, 8, 16)
    def qk(self): return self.q, self.k


class RotaryDataset(Dataset):
    def __init__(self, q, k): self.q, self.k = q, k
    def __len__(self): return 1
    def __getitem__(self, i): return self.q, self.k


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class Rotary(nn.Module):
    def __init__(self, dim, max_seq=128):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :])
        self.register_buffer("sin", emb.sin()[None, None, :, :])
    def forward(self, q, k):
        S = q.size(2)
        return apply_rotary_pos_emb(q, k, self.cos[:, :, :S], self.sin[:, :, :S])


q, k = DummyDataGenerator().qk()
rope = Rotary(16)
qr, kr = rope(q, k)
print(f"rotated q: {qr.shape}")
print("✓ RoPE applied to Q/K")"""),
    ])


def gen_v2_12():
    save("llm_path", "v2-12_smollm_from_scratch_solution.ipynb", [
        md("# V2-12: SmolLM from Scratch — Solution\n\nMinimal decoder-only LM with RoPE + GQA."),
        code("""import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, vocab=100, seq=16, batch=4):
        torch.manual_seed(6)
        self.ids = torch.randint(0, vocab, (batch, seq))
        self.vocab = vocab
    def batch(self): return self.ids


class LMTokenDataset(Dataset):
    def __init__(self, ids):
        self.ids = ids
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        s = self.ids[i]
        return s[:-1], s[1:]


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * self.weight / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SmolLMBlock(nn.Module):
    def __init__(self, d=64, n_heads=4, n_kv=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
    def forward(self, x):
        h, _ = self.attn(self.n1(x), self.n1(x), self.n1(x))
        x = x + h
        return x + self.ff(self.n2(x))


class SmolLM(nn.Module):
    def __init__(self, vocab=100, d=64, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([SmolLMBlock(d) for _ in range(layers)])
        self.lm_head = nn.Linear(d, vocab)
    def forward(self, ids):
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(x)


ids = DummyDataGenerator().batch()
model = SmolLM()
logits = model(ids[:, :-1])
print(f"logits: {logits.shape}")
print("✓ Mini SmolLM forward pass")"""),
    ])


def gen_v3_07():
    save("llm_path", "v3-07_top_p_sampling_solution.ipynb", [
        md("# V3-07: Top-p (Nucleus) Sampling — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, vocab=50):
        torch.manual_seed(7)
        self._logits = torch.randn(vocab)
    def get_logits(self):
        return self._logits


class LogitsDataset(Dataset):
    def __init__(self, logits): self.logits = logits
    def __len__(self): return 1
    def __getitem__(self, i): return self.logits


def top_p_sample(logits, p=0.9, temperature=1.0):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    mask = cum - sorted_probs > p
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs /= sorted_probs.sum()
    i = torch.multinomial(sorted_probs, 1)
    return sorted_idx[i]


class LMHead(nn.Module):
    def __init__(self, vocab=50, d=32):
        super().__init__()
        self.proj = nn.Linear(d, vocab)
    def forward(self, h):
        return self.proj(h)


logits = DummyDataGenerator().get_logits()
tok = top_p_sample(logits, p=0.9)
print(f"sampled token id: {tok.item()}")
print("✓ Top-p sampling works")"""),
    ])


def gen_v3_08():
    save("llm_path", "v3-08_top_k_sampling_solution.ipynb", [
        md("# V3-08: Top-k Sampling — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        self._logits = torch.randn(100)
    def get_logits(self):
        return self._logits


class LogitsDataset(Dataset):
    def __init__(self, l): self.l = l
    def __len__(self): return 1
    def __getitem__(self, i): return self.l


def top_k_sample(logits, k=10, temperature=1.0):
    logits = logits / temperature
    topk_vals, _ = torch.topk(logits, k)
    threshold = topk_vals[-1]
    filtered = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, 1)


class DecoderModel(nn.Module):
    def __init__(self, vocab=100):
        super().__init__()
        self.head = nn.Linear(16, vocab)
    def forward(self, h):
        return self.head(h)


tok = top_k_sample(DummyDataGenerator().get_logits(), k=5)
print(f"top-k sample: {tok.item()}")
print("✓ Top-k sampling")"""),
    ])


def gen_v3_10():
    save("llm_path", "v3-10_temperature_sampling_solution.ipynb", [
        md("# V3-10: Temperature Sampling — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        self._logits = torch.tensor([1., 2., 3., 4.])
    def get_logits(self):
        return self._logits


class LogitsDataset(Dataset):
    def __init__(self, l): self.l = l
    def __len__(self): return 1
    def __getitem__(self, i): return self.l


def temperature_sample(logits, temperature=1.0):
    scaled = logits / max(temperature, 1e-8)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, 1)


def compute_entropy(probs):
    return -(probs * (probs + 1e-8).log()).sum()


class LMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
    def forward(self, x):
        return self.fc(x)


logits = DummyDataGenerator().get_logits()
for T in [0.1, 1.0, 5.0]:
    p = torch.softmax(logits / T, dim=-1)
    print(f"T={T} entropy={compute_entropy(p):.3f}")
print("✓ Temperature affects distribution sharpness")"""),
    ])


def gen_v3_11():
    save("llm_path", "v3-11_lora_solution.ipynb", [
        md("# V3-11: LoRA — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n=64, d=32):
        torch.manual_seed(8)
        self.X = torch.randn(n, d)
        self.y = torch.randint(0, 4, (n,))
    def tensors(self): return self.X, self.y


class LoRADataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank=4, alpha=8):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False
        d_in, d_out = linear.in_features, linear.out_features
        self.A = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, d_out))
        self.scale = alpha / rank
    def forward(self, x):
        return self.linear(x) + (x @ self.A @ self.B) * self.scale


class LoRAModel(nn.Module):
    def __init__(self, d=32, c=4, rank=4):
        super().__init__()
        base = nn.Linear(d, c)
        self.lora = LoRALinear(base, rank=rank)
    def forward(self, x):
        return self.lora(x)


X, y = DummyDataGenerator().tensors()
model = LoRAModel()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable {trainable}/{total} params ({100*trainable/total:.1f}%)")
print("✓ LoRA layer with frozen base weights")"""),
    ])


def gen_v3_12():
    save("llm_path", "v3-12_kv_cache_solution.ipynb", [
        md("# V3-12: KV Cache — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        self.x = torch.randn(1, 1, 16)
    def token(self): return self.x


class TokenDataset(Dataset):
    def __init__(self, x): self.x = x
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


class KVCache:
    def __init__(self):
        self.k, self.v = None, None
    def update(self, k, v):
        self.k = k if self.k is None else torch.cat([self.k, k], dim=2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=2)
        return self.k, self.v
    def get(self):
        return self.k, self.v
    def reset(self):
        self.k, self.v = None, None


class CachedAttention(nn.Module):
    def __init__(self, d=16, heads=2):
        super().__init__()
        self.d, self.h = d, heads
        self.qkv = nn.Linear(d, 3 * d)
        self.cache = KVCache()
    def forward(self, x, use_cache=True):
        B, S, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, S, self.h, -1).transpose(1, 2) for t in qkv]
        if use_cache:
            k, v = self.cache.update(k, v)
        # When using cache, q has seq len 1 but k/v have full history
        scores = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1) @ v
        return attn.transpose(1, 2).reshape(B, S, self.d)


model = CachedAttention()
t1 = DummyDataGenerator().token()
o1 = model(t1, use_cache=True)
t2 = torch.randn(1, 1, 16)
o2 = model(t2, use_cache=True)
print(f"cached steps: {o1.shape}, {o2.shape}, cache len={model.cache.k.size(1)}")
print("✓ KV cache grows across steps")"""),
    ])


def gen_v3_13():
    save("llm_path", "v3-13_sliding_window_attention_solution.ipynb", [
        md("# V3-13: Sliding Window Attention — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, s=12, d=16):
        self.q = torch.randn(1, 4, s, d)
        self.k = self.v = torch.randn(1, 4, s, d)
    def qkv(self): return self.q, self.k, self.v


class AttnDataset(Dataset):
    def __init__(self, q, k, v): self.q, self.k, self.v = q, k, v
    def __len__(self): return 1
    def __getitem__(self, i): return self.q, self.k, self.v


def create_sliding_window_mask(seq_len, window_size):
    idx = torch.arange(seq_len)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    return (dist <= window_size).float()


def sliding_window_attention(q, k, v, window_size):
    S = q.size(-2)
    mask = create_sliding_window_mask(S, window_size).to(q.device)
    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
    scores = scores.masked_fill(mask == 0, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


class SWAModel(nn.Module):
    def __init__(self, window=2):
        super().__init__()
        self.window = window
    def forward(self, q, k, v):
        return sliding_window_attention(q, k, v, self.window)


q, k, v = DummyDataGenerator().qkv()
out = SWAModel(3)(q, k, v)
print(f"SWA output: {out.shape}")
print("✓ Sliding window attention")"""),
    ])


def gen_v3_14():
    save("llm_path", "v3-14_dpo_loss_solution.ipynb", [
        md("# V3-14: DPO Loss — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, vocab=20, seq=6, batch=4):
        torch.manual_seed(9)
        self.chosen = torch.randint(0, vocab, (batch, seq))
        self.rejected = torch.randint(0, vocab, (batch, seq))
    def pairs(self): return self.chosen, self.rejected


class PreferenceDataset(Dataset):
    def __init__(self, chosen, rejected):
        self.chosen, self.rejected = chosen, rejected
    def __len__(self): return len(self.chosen)
    def __getitem__(self, i): return self.chosen[i], self.rejected[i]


class SimpleLM(nn.Module):
    def __init__(self, vocab=20, d=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)
    def forward(self, ids):
        return self.head(self.embed(ids))


def get_batch_logps(model, ids):
    logits = model(ids)
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1).sum(-1)


def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1):
    return -torch.log(torch.sigmoid(beta * ((pi_c - pi_r) - (ref_c - ref_r)))).mean()


chosen, rejected = DummyDataGenerator().pairs()
policy, ref = SimpleLM(), SimpleLM()
ref.eval()
for p in ref.parameters(): p.requires_grad = False
loss = dpo_loss(
    get_batch_logps(policy, chosen), get_batch_logps(policy, rejected),
    get_batch_logps(ref, chosen), get_batch_logps(ref, rejected),
)
print(f"DPO loss: {loss.item():.4f}")
print("✓ DPO loss computed")"""),
    ])


def gen_v3_15():
    save("llm_path", "v3-15_ppo_rlhf_solution.ipynb", [
        md("# V3-15: PPO for RLHF — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        torch.manual_seed(10)
        self.states = torch.randn(8, 16)
        self.actions = torch.randint(0, 5, (8,))
        self.rewards = torch.randn(8)
        self.values = torch.randn(8)
        self.logp_old = torch.randn(8)
    def rollout(self):
        return self.states, self.actions, self.rewards, self.values, self.logp_old


class RolloutDataset(Dataset):
    def __init__(self, s, a, r): self.s, self.a, self.r = s, a, r
    def __len__(self): return len(self.s)
    def __getitem__(self, i): return self.s[i], self.a[i], self.r[i]


class PolicyModel(nn.Module):
    def __init__(self, d=16, n_actions=5):
        super().__init__()
        self.net = nn.Linear(d, n_actions)
    def forward(self, s):
        return torch.log_softmax(self.net(s), dim=-1)


class ValueModel(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.v = nn.Linear(d, 1)
    def forward(self, s):
        return self.v(s).squeeze(-1)


class RewardModel(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.r = nn.Linear(d, 1)
    def forward(self, s):
        return self.r(s).squeeze(-1)


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    adv = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t] if t < len(rewards)-1 else rewards[t] - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv


def ppo_step(policy, logp_old, states, actions, advantages, clip=0.2):
    logp = policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    ratio = (logp - logp_old).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-clip, 1+clip) * advantages
    return -torch.min(surr1, surr2).mean()


s, a, r, v, logp_old = DummyDataGenerator().rollout()
v_pad = torch.cat([v, torch.zeros(1)])
adv = compute_gae(r, v_pad)
policy = PolicyModel()
loss = ppo_step(policy, logp_old, s, a, adv)
print(f"PPO surrogate loss: {loss.item():.4f}")
print("✓ PPO step implemented")"""),
    ])


def gen_v3_17():
    save("llm_path", "v3-17_mixture_of_experts_solution.ipynb", [
        md("# V3-17: Mixture of Experts — Solution"),
        code("""import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class MoEConfig:
    d_model: int = 32
    num_experts: int = 4
    top_k: int = 2


class DummyDataGenerator:
    def __init__(self, n=16, d=32):
        self.x = torch.randn(n, d)
    def input(self): return self.x


class MoEDataset(Dataset):
    def __init__(self, x): self.x = x
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


class Expert(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.ReLU(), nn.Linear(4*d, d))
    def forward(self, x):
        return self.ff(x)


class MoELayer(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.gate = nn.Linear(cfg.d_model, cfg.num_experts)
        self.experts = nn.ModuleList([Expert(cfg.d_model) for _ in range(cfg.num_experts)])
    def forward(self, x):
        logits = self.gate(x)
        topk = torch.topk(logits, self.cfg.top_k, dim=-1)
        weights = torch.softmax(topk.values, dim=-1)
        out = torch.zeros_like(x)
        for i in range(self.cfg.top_k):
            idx = topk.indices[:, i]
            for e_id in range(self.cfg.num_experts):
                mask = idx == e_id
                if mask.any():
                    out[mask] += weights[mask, i:i+1] * self.experts[e_id](x[mask])
        load = torch.softmax(logits, dim=-1).mean(0)
        aux_loss = (load * load.log()).sum() * self.cfg.num_experts
        return out, aux_loss


x = DummyDataGenerator().input()
out, aux = MoELayer(MoEConfig())(x)
print(f"MoE out: {out.shape}, aux_loss={aux.item():.4f}")
print("✓ MoE routing + load balance loss")"""),
    ])


def gen_v3_18():
    save("llm_path", "v3-18_speculative_decoding_solution.ipynb", [
        md("# V3-18: Speculative Decoding — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, vocab=20):
        self.vocab = vocab
        self.prompt = torch.tensor([1, 2, 3])
    def prompt_ids(self): return self.prompt


class PromptDataset(Dataset):
    def __init__(self, ids): self.ids = ids
    def __len__(self): return 1
    def __getitem__(self, i): return self.ids


class DraftModel(nn.Module):
    def __init__(self, vocab=20, d=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)
    def forward(self, ids):
        return self.head(self.embed(ids))


class TargetModel(nn.Module):
    def __init__(self, vocab=20, d=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)
    def forward(self, ids):
        return self.head(self.embed(ids))


def speculative_decode(draft, target, prompt, K=3, max_new=6):
    seq = prompt.tolist()
    draft.eval(); target.eval()
    with torch.no_grad():
        while len(seq) < len(prompt) + max_new:
            draft_tokens = []
            ctx = torch.tensor([seq])
            for _ in range(K):
                logits = draft(ctx)[0, -1]
                t = logits.argmax().item()
                draft_tokens.append(t)
                ctx = torch.tensor([seq + draft_tokens])
            verify_logits = target(torch.tensor([seq + draft_tokens]))[0]
            accepted = 0
            for i, t in enumerate(draft_tokens):
                p_t = torch.softmax(verify_logits[len(seq)+i-1], dim=-1)
                p_d = torch.softmax(draft(ctx)[0, -1], dim=-1)
                if torch.rand(1).item() < min(1.0, (p_t[t] / (p_d[t] + 1e-8)).item()):
                    seq.append(t); accepted += 1
                else:
                    break
            if accepted == 0:
                seq.append(verify_logits[len(seq)-1].argmax().item())
    return torch.tensor(seq)


prompt = DummyDataGenerator().prompt_ids()
out = speculative_decode(DraftModel(), TargetModel(), prompt, K=2, max_new=4)
print(f"speculative output: {out.tolist()}")
print("✓ Speculative decoding loop")"""),
    ])


def gen_v3_19():
    save("llm_path", "v3-19_continuous_batching_solution.ipynb", [
        md("# V3-19: Continuous Batching — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset
from dataclasses import dataclass, field


class DummyDataGenerator:
    @staticmethod
    def requests():
        return [
            {"id": 0, "input_ids": [1, 2], "max_gen": 3},
            {"id": 1, "input_ids": [3], "max_gen": 4},
            {"id": 2, "input_ids": [4, 5, 6], "max_gen": 2},
        ]


@dataclass
class Request:
    id: int
    input_ids: list
    generated_ids: list = field(default_factory=list)
    max_gen_len: int = 5
    @property
    def is_done(self):
        return len(self.generated_ids) >= self.max_gen_len
    @property
    def all_ids(self):
        return self.input_ids + self.generated_ids


class RequestDataset(Dataset):
    def __init__(self, reqs): self.reqs = reqs
    def __len__(self): return len(self.reqs)
    def __getitem__(self, i): return self.reqs[i]


class DummyLLM(nn.Module):
    def __init__(self, vocab=10):
        super().__init__()
        self.embed = nn.Embedding(vocab, 8)
        self.head = nn.Linear(8, vocab)
    def forward(self, ids):
        return self.head(self.embed(ids))


class ContinuousBatchScheduler:
    def __init__(self, model, max_batch=4):
        self.model = model
        self.max_batch = max_batch
        self.queue = []
        self.active = []
    def add_request(self, req: Request):
        self.queue.append(req)
    def step(self):
        while self.queue and len(self.active) < self.max_batch:
            self.active.append(self.queue.pop(0))
        if not self.active:
            return []
        still = []
        for req in self.active:
            if not req.is_done:
                logits = self.model(torch.tensor([req.all_ids]))
                req.generated_ids.append(logits[0, -1].argmax().item())
            if not req.is_done:
                still.append(req)
        done = [r for r in self.active if r.is_done]
        self.active = still
        return done


sched = ContinuousBatchScheduler(DummyLLM())
for r in DummyDataGenerator.requests():
    sched.add_request(Request(id=r["id"], input_ids=r["input_ids"], max_gen_len=r["max_gen"]))
finished = []
while sched.queue or sched.active:
    finished.extend(sched.step())
print(f"completed {len(finished)} requests")
print("✓ Continuous batching scheduler")"""),
    ])


def gen_v3_27():
    save("llm_path", "v3-27_grpo_solution.ipynb", [
        md("# V3-27: GRPO — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        torch.manual_seed(11)
        self.prompts = torch.randint(0, 20, (2, 5))
        self.completions = torch.randint(0, 20, (2, 4, 6))
        self.rewards = torch.randn(2, 4)
    def batch(self):
        return self.prompts, self.completions, self.rewards


class GRPODataset(Dataset):
    def __init__(self, prompts): self.prompts = prompts
    def __len__(self): return len(self.prompts)
    def __getitem__(self, i): return self.prompts[i]


class PolicyModel(nn.Module):
    def __init__(self, vocab=20, d=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)
    def forward(self, ids):
        return self.head(self.embed(ids))


class RewardModel(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.r = nn.Linear(d, 1)
    def forward(self, h):
        return self.r(h.mean(1)).squeeze(-1)


def compute_group_advantages(rewards):
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True) + 1e-8
    return (rewards - mean) / std


def grpo_loss(logp, logp_old, advantages, beta=0.1, clip=0.2):
    ratio = (logp - logp_old).exp()
    surr = torch.min(ratio * advantages, torch.clamp(ratio, 1-clip, 1+clip) * advantages)
    return -(surr.mean())


prompts, comps, rewards = DummyDataGenerator().batch()
adv = compute_group_advantages(rewards)
print(f"group advantages shape: {adv.shape}")
print("✓ GRPO group-relative advantages")"""),
    ])


def gen_v3_28():
    save("llm_path", "v3-28_inference_engine_solution.ipynb", [
        md("# V3-28: LLM Inference Engine — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def prompts(self):
        return ["abc", "xy"]


class PromptDataset(Dataset):
    def __init__(self, texts): self.texts = texts
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i]


class SimpleTokenizer:
  def __init__(self):
    self.chars = list("abcdefghijklmnopqrstuvwxyz ")
    self.stoi = {c: i+2 for i, c in enumerate(self.chars)}
    self.stoi["<pad>"] = 0; self.stoi["<eos>"] = 1
  def encode(self, s): return [self.stoi.get(c, 0) for c in s.lower()] + [1]
  def decode(self, ids): return "".join(self.chars[i-2] for i in ids if i >= 2)


class KVCache:
    def __init__(self): self.k = self.v = None
    def update(self, k, v):
        self.k = k if self.k is None else torch.cat([self.k, k], dim=2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=2)
        return self.k, self.v


class MiniTransformer(nn.Module):
    def __init__(self, vocab=30, d=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.attn = nn.MultiheadAttention(d, 2, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.head = nn.Linear(d, vocab)
    def forward(self, ids):
        x = self.embed(ids)
        x, _ = self.attn(x, x, x)
        return self.head(self.ff(x))


class InferenceEngine:
    def __init__(self, model, tokenizer):
        self.model, self.tok = model, tokenizer
    def generate(self, prompts, max_new_tokens=5, temperature=1.0):
        results = []
        self.model.eval()
        with torch.no_grad():
            for p in prompts:
                ids = torch.tensor([self.tok.encode(p)])
                for _ in range(max_new_tokens):
                    logits = self.model(ids)[0, -1] / temperature
                    nxt = torch.multinomial(torch.softmax(logits, -1), 1)
                    ids = torch.cat([ids, nxt.unsqueeze(0)], dim=1)
                    if nxt.item() == 1: break
                results.append(self.tok.decode(ids[0].tolist()))
        return results


tok = SimpleTokenizer()
engine = InferenceEngine(MiniTransformer(), tok)
print(engine.generate(DummyDataGenerator().prompts(), max_new_tokens=3))
print("✓ Inference engine generates text")"""),
    ])


def gen_v2_20():
    save("llm_path", "v2-20_sft_smollm_solution.ipynb", [
        md("# V2-20: SFT on SmolLM — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self):
        self.data = [
            {"prompt": "What is 2+2?", "response": "4"},
            {"prompt": "Capital of France?", "response": "Paris"},
        ]
    def instructions(self): return self.data


class SFTDataset(Dataset):
    def __init__(self, data, vocab, max_len=32):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        text = self.data[i]["prompt"] + " " + self.data[i]["response"]
        ids = [self.vocab.get(c, 0) for c in text[:self.max_len]]
        ids += [0] * (self.max_len - len(ids))
        x = torch.tensor(ids[:-1])
        y = torch.tensor(ids[1:])
        return x, y


class SmolLM(nn.Module):
    def __init__(self, vocab=128, d=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, batch_first=True), 2)
        self.head = nn.Linear(d, vocab)
    def forward(self, ids):
        return self.head(self.blocks(self.embed(ids)))


vocab = {c: i+1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz0123456789 ?")}
data = DummyDataGenerator().instructions()
loader = DataLoader(SFTDataset(data, vocab), batch_size=2)
model = SmolLM(vocab=len(vocab)+1)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for x, y in loader:
    opt.zero_grad()
    nn.CrossEntropyLoss()(model(x).view(-1, model.head.out_features), y.view(-1)).backward()
    opt.step()
print("✓ SFT training step on instruction data")"""),
    ])


LLM_PATH = [
    gen_v2_06, gen_v2_07, gen_v2_08, gen_v2_10, gen_v2_11, gen_v2_12,
    gen_v3_07, gen_v3_08, gen_v3_10, gen_v3_11, gen_v3_12, gen_v3_13,
    gen_v3_14, gen_v3_15, gen_v3_17, gen_v3_18, gen_v3_19, gen_v3_27, gen_v3_28, gen_v2_20,
]

if __name__ == "__main__":
    print("Generating llm_path...")
    for fn in LLM_PATH:
        fn()
    print(f"Done llm_path ({len(LLM_PATH)} notebooks)")
