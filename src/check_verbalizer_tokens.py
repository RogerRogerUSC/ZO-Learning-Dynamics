"""
Check whether candidate verbalizer words tokenize as a single token
in OPT and GPT2 tokenizers. Prints a table showing token IDs and
whether each word is single-token (safe for last-token loss/logit extraction).
"""

from transformers import AutoTokenizer

MODELS = {
    "opt-125m": "facebook/opt-125m",
    "gpt2": "gpt2",
}

# SST-5 candidate verbalizer words (with leading space, as used in prompts)
CANDIDATES = {
    "option_B": {
        0: " awful",
        1: " bad",
        2: " neutral",
        3: " good",
        4: " excellent",
    },
    "option_A": {
        0: " terrible",
        1: " bad",
        2: " okay",
        3: " good",
        4: " great",
    },
    "alternative": {
        0: " horrible",
        1: " bad",
        2: " okay",
        3: " good",
        4: " wonderful",
    },
}

SST2_VERBALIZER = {0: " bad", 1: " good"}  # for reference


def check_verbalizer(tokenizer_name, tokenizer, verbalizer):
    results = {}
    for label, word in verbalizer.items():
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        results[label] = {
            "word": word,
            "token_ids": token_ids,
            "is_single_token": len(token_ids) == 1,
        }
    return results


def print_table(model_name, option_name, results):
    print(f"\n  [{model_name}] {option_name}")
    print(f"  {'Label':<6} {'Word':<14} {'Token IDs':<20} {'Single Token?'}")
    print(f"  {'-'*55}")
    all_single = True
    for label, info in results.items():
        ok = "YES" if info["is_single_token"] else "NO  <-- PROBLEM"
        if not info["is_single_token"]:
            all_single = False
        print(f"  {label:<6} {info['word']:<14} {str(info['token_ids']):<20} {ok}")
    print(f"  => All single-token: {'YES' if all_single else 'NO'}")


def main():
    tokenizers = {}
    for name, hf_id in MODELS.items():
        print(f"Loading tokenizer: {hf_id} ...")
        tok = AutoTokenizer.from_pretrained(hf_id, padding_side="left")
        if hf_id == "gpt2":
            tok.pad_token = tok.eos_token
        tokenizers[name] = tok

    print("\n" + "=" * 60)
    print("SST-2 reference verbalizer (should all be single-token)")
    print("=" * 60)
    for model_name, tokenizer in tokenizers.items():
        results = check_verbalizer(model_name, tokenizer, SST2_VERBALIZER)
        print_table(model_name, "sst2_reference", results)

    print("\n" + "=" * 60)
    print("SST-5 candidate verbalizers")
    print("=" * 60)
    for option_name, verbalizer in CANDIDATES.items():
        for model_name, tokenizer in tokenizers.items():
            results = check_verbalizer(model_name, tokenizer, verbalizer)
            print_table(model_name, option_name, results)

    # Summary: which options are fully single-token across all models?
    print("\n" + "=" * 60)
    print("SUMMARY — options that are fully single-token across all models")
    print("=" * 60)
    for option_name, verbalizer in CANDIDATES.items():
        safe = True
        for tokenizer in tokenizers.values():
            for word in verbalizer.values():
                ids = tokenizer.encode(word, add_special_tokens=False)
                if len(ids) != 1:
                    safe = False
                    break
        status = "SAFE" if safe else "UNSAFE"
        print(f"  {option_name:<20} {status}")


if __name__ == "__main__":
    main()
