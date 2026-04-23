import json
import os
import random

import torch
import torch.nn as nn
from glob import glob
import matplotlib.pyplot as plt


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


def kvp_compute_future_utility(attn_scores, num_kv_heads, num_query_heads):
    heads_per_kv = num_query_heads // num_kv_heads
    steps, _, seq_len = attn_scores.shape

    utility = torch.zeros(num_kv_heads, seq_len)
    for kv_idx in range(num_kv_heads):
        q_start = kv_idx * heads_per_kv
        q_end = q_start + heads_per_kv
        kv_attn = attn_scores[:, q_start:q_end, :].mean(dim=1)
        utility[kv_idx] = kv_attn.mean(dim=0)

    return utility


def kvp_pairwise_ranking_loss(scores, utility, margin=0.1, num_pairs=1000):
    seq_len = scores.shape[0]

    losses = []
    for _ in range(num_pairs):
        i = random.randint(0, seq_len - 1)
        j = random.randint(0, seq_len - 1)
        if utility[i] > utility[j]:
            loss = torch.relu(scores[j] - scores[i] + margin)
            losses.append(loss)
        elif utility[j] > utility[i]:
            loss = torch.relu(scores[i] - scores[j] + margin)
            losses.append(loss)

    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def train_kvp_agents(traces, num_layers, num_kv_heads, head_dim, epochs=10, lr=1e-3, verbose=False):
    agents = {}
    optimizers = {}
    for layer in range(num_layers):
        for kv_head in range(num_kv_heads):
            key = (layer, kv_head)
            agents[key] = TokenScorer(head_dim)
            optimizers[key] = torch.optim.Adam(agents[key].parameters(), lr=lr)

    num_query_heads = traces[0]['config']['num_heads']

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for trace in traces:
            keys = trace['keys']
            values = trace['values']
            attn = trace['attn_scores']

            for layer in range(num_layers):
                layer_attn = attn[layer]
                utility = kvp_compute_future_utility(layer_attn, num_kv_heads, num_query_heads)

                for kv_head in range(num_kv_heads):
                    key = (layer, kv_head)
                    agent = agents[key]
                    opt = optimizers[key]

                    k = keys[layer, kv_head]
                    v = values[layer, kv_head]
                    x = torch.cat([k, v], dim=-1)

                    scores = agent(x)

                    u = utility[kv_head]
                    loss = kvp_pairwise_ranking_loss(scores, u)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    total_loss += loss.item()
                    num_batches += 1

        if verbose:
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"  kvp epoch {epoch + 1}/{epochs} loss={avg_loss:.4f}")

    return agents


def evaluate_kvp(agents, eval_traces, budget_frac=0.5):
    all_recalls = []

    for trace in eval_traces:
        keys = trace['keys']
        values = trace['values']
        attn = trace['attn_scores']

        config = trace['config']
        num_layers = config['num_layers']
        num_kv_heads = config['num_kv_heads']
        num_query_heads = config['num_heads']

        for layer in range(num_layers):
            layer_attn = attn[layer]
            utility = kvp_compute_future_utility(layer_attn, num_kv_heads, num_query_heads)

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

                _, gt_top = torch.topk(u, budget)
                gt_set = set(gt_top.tolist())

                with torch.no_grad():
                    scores = agent(x)
                _, kvp_top = torch.topk(scores, budget)
                kvp_set = set(kvp_top.tolist())
                kvp_recall = len(kvp_set & gt_set) / budget

                all_recalls.append(kvp_recall)

    return sum(all_recalls) / len(all_recalls)


class JointScorer(nn.Module):
    def __init__(self, head_dim=128, num_heads=56, embed_dim=64, attn_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.token_embed = nn.Sequential(
            nn.Linear(head_dim * 2, embed_dim),
            nn.ReLU(),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=attn_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.score_head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        seq_len, num_heads, _ = x.shape

        x = self.token_embed(x)

        attn_out, _ = self.cross_attn(x, x, x)
        x = self.norm(x + attn_out)

        scores = self.score_head(x).squeeze(-1)

        return scores


def joint_compute_future_utility(attn_scores, num_kv_heads, num_query_heads, num_layers):
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


def joint_pairwise_ranking_loss(scores, utility, margin=0.1, num_pairs=1000):
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


def build_joint_input(trace):
    keys = trace['keys']
    values = trace['values']

    num_layers, num_kv_heads, seq_len, head_dim = keys.shape

    kv = torch.cat([keys, values], dim=-1)

    kv = kv.permute(2, 0, 1, 3)
    kv = kv.reshape(seq_len, num_layers * num_kv_heads, head_dim * 2)

    return kv


def train_joint(traces, config, epochs=10, lr=1e-3, verbose=False):
    num_layers = config['num_layers']
    num_kv_heads = config['num_kv_heads']
    num_query_heads = config['num_heads']
    head_dim = config['head_dim']
    total_heads = num_layers * num_kv_heads

    model = JointScorer(head_dim=head_dim, num_heads=total_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for trace in traces:
            x = build_joint_input(trace)
            utility = joint_compute_future_utility(
                trace['attn_scores'], num_kv_heads, num_query_heads, num_layers
            )

            scores = model(x)
            loss = joint_pairwise_ranking_loss(scores, utility)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if verbose:
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"  joint epoch {epoch + 1}/{epochs} loss={avg_loss:.4f}")

    return model


def evaluate_joint(model, eval_traces, budget_frac=0.5):
    model.eval()
    all_recalls = []

    for trace in eval_traces:
        config = trace['config']
        num_layers = config['num_layers']
        num_kv_heads = config['num_kv_heads']
        num_query_heads = config['num_heads']
        total_heads = num_layers * num_kv_heads

        x = build_joint_input(trace)
        utility = joint_compute_future_utility(
            trace['attn_scores'], num_kv_heads, num_query_heads, num_layers
        )

        seq_len = x.shape[0]
        budget = int(seq_len * budget_frac)

        with torch.no_grad():
            scores = model(x)

        for h in range(total_heads):
            u = utility[h]
            s = scores[:, h]

            _, gt_top = torch.topk(u, budget)
            gt_set = set(gt_top.tolist())
            _, joint_top = torch.topk(s, budget)
            joint_set = set(joint_top.tolist())
            joint_recall = len(joint_set & gt_set) / budget

            all_recalls.append(joint_recall)

    return sum(all_recalls) / len(all_recalls)


def load_traces(trace_dir):
    paths = sorted(glob(os.path.join(trace_dir, "trace_*.pt")))
    traces = []
    for p in paths:
        trace = torch.load(p, weights_only=False)
        traces.append(trace)
    return traces


def run_scaling_experiment():
    random.seed(42)
    torch.manual_seed(42)

    trace_dir = "./traces"
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    all_traces = load_traces(trace_dir)
    print(f"Loaded {len(all_traces)} traces")

    eval_traces = all_traces[-5:]
    train_pool = all_traces[:25]
    print(f"Training pool: {len(train_pool)} traces")
    print(f"Eval set: {len(eval_traces)} traces (held out)")

    config = all_traces[0]['config']
    num_layers = config['num_layers']
    num_kv_heads = config['num_kv_heads']
    head_dim = config['head_dim']

    print(f"Config: layers={num_layers} kv_heads={num_kv_heads} head_dim={head_dim}")
    print()

    trace_counts = [2, 5, 10, 15, 20, 25]
    results = {
        'trace_counts': trace_counts,
        'kvp_recalls': [],
        'joint_recalls': [],
    }

    for n_traces in trace_counts:
        print(f"Training with {n_traces} traces")

        # sample traces (same seed ensures reproducibility)
        random.seed(42)
        train_traces = random.sample(train_pool, n_traces)

        # train KVP
        print(f"Training KVP (56 agents)")
        kvp_agents = train_kvp_agents(
            train_traces,
            num_layers,
            num_kv_heads,
            head_dim,
            epochs=10,
            lr=1e-3,
            verbose=False
        )
        kvp_recall = evaluate_kvp(kvp_agents, eval_traces, budget_frac=0.5)
        print(f"  KVP recall@50%: {kvp_recall:.4f}")

        # train joint
        print(f"Training JointScorer")
        joint_model = train_joint(
            train_traces,
            config,
            epochs=10,
            lr=1e-3,
            verbose=False
        )
        joint_recall = evaluate_joint(joint_model, eval_traces, budget_frac=0.5)
        print(f"  Joint recall@50%: {joint_recall:.4f}")

        results['kvp_recalls'].append(kvp_recall)
        results['joint_recalls'].append(joint_recall)
        print()

    print("RESULTS")
    print(f"{'traces':<8} | {'kvp':<8} | {'joint':<8} | {'delta':<8}")

    crossover_point = None
    for i, n in enumerate(trace_counts):
        kvp = results['kvp_recalls'][i]
        joint = results['joint_recalls'][i]
        delta = joint - kvp
        print(f"{n:<8} | {kvp:<8.4f} | {joint:<8.4f} | {delta:+.4f}")

        if i > 0 and crossover_point is None:
            prev_kvp = results['kvp_recalls'][i-1]
            prev_joint = results['joint_recalls'][i-1]
            if prev_kvp > prev_joint and joint > kvp:
                crossover_point = (trace_counts[i-1], trace_counts[i])

    if crossover_point:
        print(f"Crossover between {crossover_point[0]} and {crossover_point[1]} traces")
    else:
        final_kvp = results['kvp_recalls'][-1]
        final_joint = results['joint_recalls'][-1]
        if final_joint > final_kvp:
            print("Joint outperforms KVP at all tested trace counts")
        else:
            print("KVP outperforms Joint at all tested trace counts")

    # raw results
    results_path = os.path.join(results_dir, "scaling_experiment.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(trace_counts, results['kvp_recalls'], 'b-o', label='KVP (56 agents)', linewidth=2, markersize=8)
    plt.plot(trace_counts, results['joint_recalls'], 'r-s', label='JointScorer', linewidth=2, markersize=8)

    if crossover_point:
        # interpolate crossover point
        i = trace_counts.index(crossover_point[1])
        x1, x2 = crossover_point
        y1_kvp, y2_kvp = results['kvp_recalls'][i-1], results['kvp_recalls'][i]
        y1_joint, y2_joint = results['joint_recalls'][i-1], results['joint_recalls'][i]

        slope_kvp = (y2_kvp - y1_kvp) / (x2 - x1)
        slope_joint = (y2_joint - y1_joint) / (x2 - x1)
        if slope_joint != slope_kvp:
            x_cross = x1 + (y1_kvp - y1_joint) / (slope_joint - slope_kvp)
            y_cross = y1_kvp + slope_kvp * (x_cross - x1)
            plt.axvline(x=x_cross, color='gray', linestyle='--', alpha=0.7)
            plt.scatter([x_cross], [y_cross], color='green', s=100, zorder=5, label=f'Crossover (~{x_cross:.1f} traces)')

    plt.xlabel('Number of Training Traces', fontsize=12)
    plt.ylabel('Recall@50%', fontsize=12)
    plt.title('KVP vs JointScorer: Scaling with Training Data', fontsize=14)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(trace_counts)

    plot_path = os.path.join(results_dir, "scaling_experiment.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")

    plt.close()


if __name__ == "__main__":
    run_scaling_experiment()
