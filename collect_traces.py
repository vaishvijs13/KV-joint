import argparse
import os

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_model_config(model):
    config = model.config
    return {
        'num_layers': config.num_hidden_layers,
        'num_heads': config.num_attention_heads,
        'num_kv_heads': getattr(config, 'num_key_value_heads', config.num_attention_heads),
        'head_dim': config.hidden_size // config.num_attention_heads,
    }


def collect_trace_for_sequence(model, tokenizer, input_text, max_new_tokens=128, device="cuda"):
    config = get_model_config(model)
    num_layers = config['num_layers']

    all_keys = [[] for _ in range(num_layers)]
    all_values = [[] for _ in range(num_layers)]
    all_attn_scores = [[] for _ in range(num_layers)]

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    input_ids = inputs.input_ids.to(device)

    generated_ids = input_ids.clone()
    past_key_values = None

    with torch.no_grad():
        for step in range(max_new_tokens):
            if past_key_values is None:
                curr_input_ids = generated_ids
                position_ids = None
            else:
                curr_input_ids = generated_ids[:, -1:]
                seq_len = past_key_values[0][0].shape[2] + 1
                position_ids = torch.tensor([[seq_len - 1]], device=device)

            outputs = model(
                input_ids=curr_input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )

            logits = outputs.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            past_key_values = outputs.past_key_values
            for layer_idx in range(num_layers):
                key_states = past_key_values[layer_idx][0]
                value_states = past_key_values[layer_idx][1]
                all_keys[layer_idx].append(key_states[0].detach().cpu())
                all_values[layer_idx].append(value_states[0].detach().cpu())

            if outputs.attentions is not None:
                for layer_idx, attn in enumerate(outputs.attentions):
                    # [batch, heads, q_len, kv_len] -> [heads, kv_len]
                    attn_weights = attn[0, :, -1, :].detach().cpu()
                    attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-9)
                    all_attn_scores[layer_idx].append(attn_weights)

            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    result = {
        'input_text': input_text,
        'generated_text': generated_text,
        'input_ids': input_ids.cpu(),
        'generated_ids': generated_ids.cpu(),
        'config': config,
    }

    if all_keys[0]:
        keys_final = []
        values_final = []
        attn_stacked = []

        for layer_idx in range(num_layers):
            keys_final.append(all_keys[layer_idx][-1])
            values_final.append(all_values[layer_idx][-1])

            if all_attn_scores[layer_idx]:
                max_seq = max(t.shape[-1] for t in all_attn_scores[layer_idx])
                padded_attn = []
                for t in all_attn_scores[layer_idx]:
                    if t.shape[-1] < max_seq:
                        pad = torch.zeros(t.shape[0], max_seq - t.shape[-1])
                        t = torch.cat([t, pad], dim=-1)
                    padded_attn.append(t)
                layer_attn = torch.stack(padded_attn, dim=0)
                attn_stacked.append(layer_attn)

        # [layers, num_kv_heads, seq_len, head_dim]
        result['keys'] = torch.stack(keys_final, dim=0)
        result['values'] = torch.stack(values_final, dim=0)

        if attn_stacked:
            # [layers, steps, num_heads, seq_len]
            result['attn_scores'] = torch.stack(attn_stacked, dim=0)

        result['keys_per_step'] = all_keys
        result['values_per_step'] = all_values

    return result


def load_long_context_data(num_sequences, max_length):
    try:
        dataset = load_dataset("pg19", split="train", streaming=True)
        texts = []
        for i, item in enumerate(dataset):
            if i >= num_sequences:
                break
            # approx 4 chars per token
            text = item['text'][:max_length * 4]
            texts.append(text)
        return texts
    except Exception as e:
        print(f"pg19 failed: {e}, using synthetic text")
        base_text = "The quick brown fox jumps over the lazy dog. " * 20
        return [(base_text * (max_length // 50)) + f" seq {i}." for i in range(num_sequences)]


def save_trace(trace, output_path):
    save_data = {
        'keys': trace.get('keys'),
        'values': trace.get('values'),
        'attn_scores': trace.get('attn_scores'),
        'keys_per_step': trace.get('keys_per_step'),
        'values_per_step': trace.get('values_per_step'),
        'input_ids': trace.get('input_ids'),
        'generated_ids': trace.get('generated_ids'),
        'config': trace.get('config'),
    }
    torch.save(save_data, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--num_sequences", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="./traces")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print(f"device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # qwen/qwen2.5-1.5B
    print(f"loading {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32

    # eager attn to get attention weights
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map=device if device != "mps" else None,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    if device == "mps":
        model = model.to(device)
    model.eval()

    config = get_model_config(model)
    print(f"layers={config['num_layers']} heads={config['num_heads']} kv_heads={config['num_kv_heads']} head_dim={config['head_dim']}")

    texts = load_long_context_data(args.num_sequences, args.max_length)
    print(f"loaded {len(texts)} sequences")

    for i, text in enumerate(tqdm(texts)):
        trace = collect_trace_for_sequence(
            model=model,
            tokenizer=tokenizer,
            input_text=text,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )

        output_path = os.path.join(args.output_dir, f"trace_{i:04d}.pt")
        save_trace(trace, output_path)

        if 'keys' in trace and trace['keys'] is not None:
            k, v = trace['keys'].shape, trace['values'].shape
            a = trace['attn_scores'].shape if trace.get('attn_scores') is not None else None
            print(f"trace {i}: keys={list(k)} values={list(v)} attn={list(a) if a else None}")

    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
