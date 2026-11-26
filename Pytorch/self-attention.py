import torch
import torch.nn.functional as F

def self_attention_demo():
    
    X = torch.tensor([
        [1.0,0.0,1.0,0.0],
        [0.0,2.0,0.0,2.0],
        [1.0,1.0,1.0,1.0],
    ])
    print(f"输入X的shape：{X.shape}")
    print(f"输入X：\n{X}\n")

    torch.manual_seed(42)
    d_model = 4
    d_k = 2
    W_q = torch.randn(d_model,d_k)
    W_k = torch.randn(d_model,d_k)
    W_v = torch.randn(d_model,d_k)
    print(f"W_q shape：{W_q.shape}")

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    print(f"\nQ shape: {Q.shape}")  # torch.Size([3, 2])
    print(f"Q (每个词的查询向量):\n{torch.round(Q,decimals=2)}")
    print(f"\nK shape: {K.shape}")
    print(f"K (每个词的索引向量):\n{torch.round(K,decimals=2)}")
    print(f"\nV shape: {V.shape}")
    print(f"V (每个词的内容向量):\n{torch.round(V,decimals=2)}")

    scores = Q @ K.T
    print(f"\n注意力分数 Q·K^T shape: {scores.shape}")
    print(f"注意力分数矩阵：\n{torch.round(scores,decimals=2)}")

    scores_scaled = scores / torch.sqrt(torch.tensor(d_k,dtype=torch.float32))
    print(f"\n缩放后分数 (除以 sqrt({d_k})):\n{torch.round(scores_scaled,decimals=2)}")

    attn_weights = F.softmax(scores_scaled,dim=-1)
    print(f"\n注意力权重 (softmax后, 每行和为1):\n{torch.round(attn_weights,decimals=3)}")

    output = attn_weights @ V
    print(f"\n输出 shape: {output.shape}")  # torch.Size([3, 2])
    print(f"最终输出 (每个词的新表示):\n{torch.round(output,decimals=2)}")

    print("\n" + "=" * 60)
    print("怎么看这个结果？")
    print(f"词1 的新表示 = {torch.round(output[0],decimals=2)}")
    print(f"  = 权重 {torch.round(attn_weights[0],decimals=3)} · 所有词的 V")
    print(f"  → 词1 对词1的关注度 {attn_weights[0][0]:.1%}，对词2 {attn_weights[0][1]:.1%}，对词3 {attn_weights[0][2]:.1%}")

    return output

self_attention_demo()