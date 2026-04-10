import argparse
import json
import os
import random

import torch
import torch.nn as nn
from glob import glob

class JointScorer(nn.Module):
    def __init__(self, head_dim=128, num_heads=56, embed_dim=64, attn_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # per-head token embedding
        self.token_embed = nn.Sequential(
            nn.Linear(head_dim * 2, embed_dim),
            nn.ReLU(),
        )

        # cross-head attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=attn_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

        # scoring head
        self.score_head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        seq_len, num_heads, _ = x.shape

        x = self.token_embed(x)

        attn_out, _ = self.cross_attn(x, x, x)
        x = self.norm(x + attn_out)

        scores = self.score_head(x).squeeze(-1)

        return scores


def compute_future_utility(attn_scores, num_kv_heads, num_query_heads, num_layers):
    heads_per_kv = num_query_heads // num_kv_heads
    _, steps, _, seq_len = attn_scores.shape

    utilities = []
    for layer in range(num_layers):
        layer_attn = attn_scores[layer]
        for kv_idx in range(num_kv_heads):
            q_start = kv_idx * heads_per_kv
            q_end = q_start + heads_per_kv
            kv_attn = layer_attn[:, q_start:q_end, :].mean(dim=1)
            utility = kv_attn.mean(dim=0)
            utilities.append(utility)

    return torch.stack(utilities, dim=0)


def pairwise_ranking_loss(scores, utility, margin=0.1, num_pairs=1000):
    seq_len, num_heads = scores.shape

    losses = []
    for _ in range(num_pairs):
        h = random.randint(0, num_heads - 1)
        i = random.randint(0, seq_len - 1)
        j = random.randint(0, seq_len - 1)

        u_i = utility[h, i]
        u_j = utility[h, j]
        s_i = scores[i, h]
        s_j = scores[j, h]

        if u_i > u_j:
            loss = torch.relu(s_j - s_i + margin)
            losses.append(loss)
        elif u_j > u_i:
            loss = torch.relu(s_i - s_j + margin)
            losses.append(loss)

    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def build_input(trace):
    keys = trace['keys']
    values = trace['values']

    num_layers, num_kv_heads, seq_len, head_dim = keys.shape

    kv = torch.cat([keys, values], dim=-1)

    kv = kv.permute(2, 0, 1, 3)
    kv = kv.reshape(seq_len, num_layers * num_kv_heads, head_dim * 2)

    return kv


def load_traces(trace_dir):
    paths = sorted(glob(os.path.join(trace_dir, "trace_*.pt")))
    traces = []
    for p in paths:
        trace = torch.load(p, weights_only=False)
        traces.append(trace)
    return traces


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(traces, config, epochs=10, lr=1e-3):
    num_layers = config['num_layers']
    num_kv_heads = config['num_kv_heads']
    num_query_heads = config['num_heads']
    head_dim = config['head_dim']
    total_heads = num_layers * num_kv_heads

    model = JointScorer(head_dim=head_dim, num_heads=total_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"joint scorer params: {count_parameters(model):,}")
    print(f"training for {epochs} epochs on {len(traces)} traces")

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for trace in traces:
            x = build_input(trace)
            utility = compute_future_utility(
                trace['attn_scores'], num_kv_heads, num_query_heads, num_layers
            )

            scores = model(x)
            loss = pairwise_ranking_loss(scores, utility)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"epoch {epoch + 1}/{epochs} loss={avg_loss:.4f}")

    return model


def evaluate(model, trace, budget_frac=0.5):
    config = trace['config']
    num_layers = config['num_layers']
    num_kv_heads = config['num_kv_heads']
    num_query_heads = config['num_heads']
    total_heads = num_layers * num_kv_heads

    x = build_input(trace)
    utility = compute_future_utility(
        trace['attn_scores'], num_kv_heads, num_query_heads, num_layers
    )

    seq_len = x.shape[0]
    budget = int(seq_len * budget_frac)

    model.eval()
    with torch.no_grad():
        scores = model(x)

    results = {'joint': [], 'random': [], 'recency': []}

    for h in range(total_heads):
        u = utility[h]
        s = scores[:, h]

        _, gt_top = torch.topk(u, budget)
        gt_set = set(gt_top.tolist())
        _, joint_top = torch.topk(s, budget)
        joint_set = set(joint_top.tolist())
        joint_recall = len(joint_set & gt_set) / budget

        rand_top = set(random.sample(range(seq_len), budget))
        rand_recall = len(rand_top & gt_set) / budget

        # recency for comparison
        recency_top = set(range(seq_len - budget, seq_len))
        recency_recall = len(recency_top & gt_set) / budget

        results['joint'].append(joint_recall)
        results['random'].append(rand_recall)
        results['recency'].append(recency_recall)

    for k in results:
        results[k] = sum(results[k]) / len(results[k])

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=str, default="./traces")
    parser.add_argument("--output_dir", type=str, default="./agents/joint")
    parser.add_argument("--results_path", type=str, default="./results/joint_eval.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--budget_frac", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)

    traces = load_traces(args.trace_dir)
    if not traces:
        print(f"no traces found in {args.trace_dir}")
        return

    config = traces[0]['config']
    num_layers = config['num_layers']
    num_kv_heads = config['num_kv_heads']
    head_dim = config['head_dim']
    total_heads = num_layers * num_kv_heads

    print(f"loaded {len(traces)} traces")
    print(f"layers={num_layers} kv_heads={num_kv_heads} total_heads={total_heads} head_dim={head_dim}")
    print(f"input shape: [seq_len, {total_heads}, {head_dim * 2}]")

    # param comparison
    kvp_params = total_heads * (256 * 64 + 64 + 64 * 1 + 1)
    print(f"kvp baseline params: {kvp_params:,} (56 separate agents)")

    model = train(traces, config, epochs=args.epochs, lr=args.lr)

    path = os.path.join(args.output_dir, "joint_scorer.pt")
    torch.save(model.state_dict(), path)
    print(f"saved to {path}")

    eval_trace = traces[-1]
    results = evaluate(model, eval_trace, args.budget_frac)

    kvp_results_path = "./results/kvp_eval.json"
    if os.path.exists(kvp_results_path):
        with open(kvp_results_path) as f:
            kvp_results = json.load(f)
        results['kvp'] = kvp_results.get('kvp', 0)

    print(f"\nrecall@{args.budget_frac:.0%} budget:")
    print(f"  joint:   {results['joint']:.4f}")
    if 'kvp' in results:
        print(f"  kvp:     {results['kvp']:.4f}")
    print(f"  recency: {results['recency']:.4f}")
    print(f"  random:  {results['random']:.4f}")

    with open(args.results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"saved results to {args.results_path}")


if __name__ == "__main__":
    main()
