#!/usr/bin/env python3
import argparse
import json
import os
import random

import torch
import torch.nn as nn
from glob import glob
from tqdm import tqdm

class TokenScorer(nn.Module):
    def __init__(self, head_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(head_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_future_utility(attn_scores, num_kv_heads, num_query_heads):
    heads_per_kv = num_query_heads // num_kv_heads
    steps, _, seq_len = attn_scores.shape

    utility = torch.zeros(num_kv_heads, seq_len)
    for kv_idx in range(num_kv_heads):
        q_start = kv_idx * heads_per_kv
        q_end = q_start + heads_per_kv
        kv_attn = attn_scores[:, q_start:q_end, :].mean(dim=1)
        utility[kv_idx] = kv_attn.mean(dim=0)

    return utility


def pairwise_ranking_loss(scores, utility, margin=0.1, num_pairs=1000):
    seq_len = scores.shape[0]

    # find pairs with utility difference
    losses = []
    for _ in range(num_pairs):
        i = random.randint(0, seq_len - 1)
        j = random.randint(0, seq_len - 1)
        if utility[i] > utility[j]:
            # want score_i > score_j
            loss = torch.relu(scores[j] - scores[i] + margin)
            losses.append(loss)
        elif utility[j] > utility[i]:
            loss = torch.relu(scores[i] - scores[j] + margin)
            losses.append(loss)

    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def load_traces(trace_dir):
    paths = sorted(glob(os.path.join(trace_dir, "trace_*.pt")))
    traces = []
    for p in paths:
        trace = torch.load(p, weights_only=False)
        traces.append(trace)
    return traces


def train_agents(traces, num_layers, num_kv_heads, head_dim, epochs=10, lr=1e-3):
    # one agent per (layer, kv_head)
    agents = {}
    optimizers = {}
    for layer in range(num_layers):
        for kv_head in range(num_kv_heads):
            key = (layer, kv_head)
            agents[key] = TokenScorer(head_dim)
            optimizers[key] = torch.optim.Adam(agents[key].parameters(), lr=lr)

    num_query_heads = traces[0]['config']['num_heads']
    print(f"training {len(agents)} agents for {epochs} epochs on {len(traces)} traces")

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for trace in traces:
            keys = trace['keys']
            values = trace['values']
            attn = trace['attn_scores']

            for layer in range(num_layers):
                layer_attn = attn[layer]
                utility = compute_future_utility(layer_attn, num_kv_heads, num_query_heads)

                for kv_head in range(num_kv_heads):
                    key = (layer, kv_head)
                    agent = agents[key]
                    opt = optimizers[key]

                    # build input: concat k and v
                    k = keys[layer, kv_head]
                    v = values[layer, kv_head]
                    x = torch.cat([k, v], dim=-1)

                    scores = agent(x)

                    u = utility[kv_head]
                    loss = pairwise_ranking_loss(scores, u)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    total_loss += loss.item()
                    num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"epoch {epoch + 1}/{epochs} loss={avg_loss:.4f}")

    return agents


def evaluate(agents, trace, budget_frac=0.5):
    keys = trace['keys']
    values = trace['values']
    attn = trace['attn_scores']

    config = trace['config']
    num_layers = config['num_layers']
    num_kv_heads = config['num_kv_heads']
    num_query_heads = config['num_heads']

    results = {'kvp': [], 'random': [], 'recency': [], 'frequency': []}

    for layer in range(num_layers):
        layer_attn = attn[layer]
        utility = compute_future_utility(layer_attn, num_kv_heads, num_query_heads)

        for kv_head in range(num_kv_heads):
            key = (layer, kv_head)
            agent = agents[key]
            agent.eval()

            k = keys[layer, kv_head]
            v = values[layer, kv_head]
            seq_len = k.shape[0]
            budget = int(seq_len * budget_frac)

            x = torch.cat([k, v], dim=-1)
            u = utility[kv_head]

            # ground truth: top-B by utility
            _, gt_top = torch.topk(u, budget)
            gt_set = set(gt_top.tolist())

            # kvp: top-B by agent score
            with torch.no_grad():
                scores = agent(x)
            _, kvp_top = torch.topk(scores, budget)
            kvp_set = set(kvp_top.tolist())
            kvp_recall = len(kvp_set & gt_set) / budget

            rand_top = set(random.sample(range(seq_len), budget))
            rand_recall = len(rand_top & gt_set) / budget

            recency_top = set(range(seq_len - budget, seq_len))
            recency_recall = len(recency_top & gt_set) / budget

            _, freq_top = torch.topk(u, budget)
            freq_set = set(freq_top.tolist())
            freq_recall = len(freq_set & gt_set) / budget

            results['kvp'].append(kvp_recall)
            results['random'].append(rand_recall)
            results['recency'].append(recency_recall)
            results['frequency'].append(freq_recall)

    for k in results:
        results[k] = sum(results[k]) / len(results[k])

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=str, default="./traces")
    parser.add_argument("--output_dir", type=str, default="./agents/kvp")
    parser.add_argument("--results_path", type=str, default="./results/kvp_eval.json")
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

    print(f"loaded {len(traces)} traces")
    print(f"layers={num_layers} kv_heads={num_kv_heads} head_dim={head_dim}")
    print(f"keys shape: {traces[0]['keys'].shape}")
    print(f"attn shape: {traces[0]['attn_scores'].shape}")

    agents = train_agents(
        traces,
        num_layers,
        num_kv_heads,
        head_dim,
        epochs=args.epochs,
        lr=args.lr,
    )

    for (layer, kv_head), agent in agents.items():
        path = os.path.join(args.output_dir, f"layer{layer}_head{kv_head}.pt")
        torch.save(agent.state_dict(), path)
    print(f"saved {len(agents)} agents to {args.output_dir}")

    eval_trace = traces[-1]
    results = evaluate(agents, eval_trace, args.budget_frac)
    print(f"\nrecall@{args.budget_frac:.0%} budget:")
    for method, recall in results.items():
        print(f"  {method}: {recall:.4f}")

    with open(args.results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"saved results to {args.results_path}")


if __name__ == "__main__":
    main()
