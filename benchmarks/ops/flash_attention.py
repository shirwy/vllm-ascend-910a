import torch
import torch_npu


# prefill

torch.manual_seed(0)


bs = 256
seqlen = 8
head_num = 32
kv_head_num = 8
head_dim = 128

dtype = torch.float16
device = "npu"

query = torch.randn(bs * seqlen, head_num, head_dim, dtype=dtype, device=device)
key = torch.randn(bs * seqlen, kv_head_num, head_dim, dtype=dtype, device=device)
value = torch.randn(bs * seqlen, kv_head_num, head_dim, dtype=dtype, device=device)
mask = torch.ones(seqlen, seqlen, dtype=dtype, device=device)
mask = mask.tril()
mask = mask.unsqueeze(0).expand(bs, -1, -1)  # (bs, seqlen, seqlen)

output = torch.zeros_like(query)
seq_lens = torch.tensor([seqlen] * bs, dtype=torch.int32, device='cpu')
scale = 0.08838834764831845

torch_npu._npu_flash_attention(
    query=query,
    key=key,
    value=value,
    mask=mask,
    seq_len=seq_lens,
    scale_value=scale,
    num_heads=head_num,
    num_kv_heads=kv_head_num,
    out=output)


torch.npu.synchronize()
print(output)
