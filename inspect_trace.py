#!/usr/bin/env python3
import argparse
import sys
import torch

def inspect_trace(trace_path):
    print(f"loading {trace_path}")
    trace = torch.load(trace_path, weights_only=False)

    config = trace.get('config', {})
    print(f"config: layers={config.get('num_layers')} heads={config.get('num_heads')} kv_heads={config.get('num_kv_heads')} head_dim={config.get('head_dim')}")

    input_ids = trace.get('input_ids')
    generated_ids = trace.get('generated_ids')
    if input_ids is not None:
        print(f"tokens: input={input_ids.shape[-1]} total={generated_ids.shape[-1]} new={generated_ids.shape[-1] - input_ids.shape[-1]}")

    keys = trace.get('keys')
    values = trace.get('values')
    if keys is not None:
        print(f"final kv: keys={list(keys.shape)} values={list(values.shape)}")

    attn_scores = trace.get('attn_scores')
    if attn_scores is not None:
        print(f"attn_scores: {list(attn_scores.shape)} (layers, steps, heads, seq)")
        print(f"  layer0 max={attn_scores[0].max():.4f} mean={attn_scores[0].mean():.4f}")

    keys_per_step = trace.get('keys_per_step')
    if keys_per_step is not None:
        num_layers = len(keys_per_step)
        num_steps = len(keys_per_step[0]) if keys_per_step[0] else 0
        print(f"per-step: {num_layers} layers x {num_steps} steps")
        if num_steps > 0:
            print(f"  step0={list(keys_per_step[0][0].shape)} step-1={list(keys_per_step[0][-1].shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_path", type=str)
    args = parser.parse_args()

    try:
        inspect_trace(args.trace_path)
    except FileNotFoundError:
        print(f"not found: {args.trace_path}")
        sys.exit(1)
    except Exception as e:
        print(f"error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
