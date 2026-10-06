import sys
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]

# Same reason as train.py: the model files import each other by bare name
# (`from decoder import ...`), which only resolves with src/models on sys.path.
sys.path.insert(0, str(ROOT / "src" / "models"))

from models.decoder import generate_causal_mask
from models.transformer import Transformer


def greedy_decode(model, src, src_mask, max_len, sos_id, eos_id, device):
    model.eval()
    with torch.no_grad():
        encoder_output = model.encoder(src, src_mask)

        tgt = torch.tensor([[sos_id]], dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt_mask = generate_causal_mask(tgt.size(1)).to(device)

            decoder_output = model.decoder(tgt, encoder_output, src_mask, tgt_mask)
            logits = model.output_projection(decoder_output)

            next_token_logits = logits[:, -1, :]  # [batch, vocab_size] — only the newest position
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)  # [batch, 1]

            tgt = torch.cat([tgt, next_token], dim=1)

            if next_token.item() == eos_id:
                break

    return tgt.squeeze().tolist()


def beam_search_decode(model, src, src_mask, max_len, sos_id, eos_id, device,
                       beam_size=4, alpha=0.6):
    """Returns (ids, finished). ids has the same format as greedy_decode:
    starts with sos, includes eos if it was produced. finished is True
    if the winning sequence ended with eos."""
    model.eval()
    with torch.no_grad():
        # Encode the source once; every beam reuses this same encoder output.
        encoder_output = model.encoder(src, src_mask)

        # Each beam is (token ids so far, sum of log-probs). Start with sos only.
        beams = [([sos_id], 0.0)]
        finished = []

        for _ in range(max_len):
            if not beams:
                break

            # Expand every live beam by its top `beam_size` next tokens.
            candidates = []
            for tokens, score in beams:
                tgt = torch.tensor([tokens], dtype=torch.long, device=device)
                tgt_mask = generate_causal_mask(tgt.size(1)).to(device)

                decoder_output = model.decoder(tgt, encoder_output, src_mask, tgt_mask)
                logits = model.output_projection(decoder_output)

                # Newest position only, as log-probabilities (log_softmax, not softmax then log).
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)[0]
                top_log_probs, top_ids = torch.topk(log_probs, beam_size)

                for log_prob, token_id in zip(top_log_probs.tolist(), top_ids.tolist()):
                    candidates.append((tokens + [token_id], score + log_prob))

            # Keep the best beam_size candidates overall. Those ending in eos are
            # finished and stop growing; the rest stay live.
            candidates.sort(key=lambda c: c[1], reverse=True)
            beams = []
            for tokens, score in candidates[:beam_size]:
                if tokens[-1] == eos_id:
                    finished.append((tokens, score))
                else:
                    beams.append((tokens, score))

        # Prefer finished sequences; fall back to live ones if none finished (hit max_len).
        pool = finished if finished else beams
        if not pool:
            return [sos_id], False

        # Length normalization: divide by generated-token count ** alpha.
        # The sos is not counted. alpha = 0 means raw score, no normalization.
        best_tokens, _ = max(
            pool,
            key=lambda c: c[1] / (len(c[0]) - 1) ** alpha,
        )

    return best_tokens, best_tokens[-1] == eos_id


# ── Test-Block ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.from_file(str(ROOT / "data" / "bpe_tokenizer.json"))
    special = config["tokenizer"]["special_tokens"]
    PAD_ID = tokenizer.token_to_id(special["pad"])
    SOS_ID = tokenizer.token_to_id(special["sos"])
    EOS_ID = tokenizer.token_to_id(special["eos"])

    model = Transformer(
        vocab_size=config["tokenizer"]["vocab_size"],
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        num_layers=config["model"]["num_layers"],
        max_len=config["model"]["max_len"],
        dropout=config["model"]["dropout"],
        pad_id=PAD_ID,
    ).to(device)

    checkpoint = torch.load(ROOT / "experiments" / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, val_loss {checkpoint['best_val_loss']:.4f}")

    # Try one real German sentence
    from datasets import load_dataset

    dataset_name = config["dataset"]["name"]
    src_lang = config["dataset"]["source_lang"]
    tgt_lang = config["dataset"]["target_lang"]

    test_set = load_dataset(dataset_name, split="test")

    NUM_EXAMPLES = 10

    for i in range(NUM_EXAMPLES):
        german_sentence = test_set[i][src_lang]
        reference = test_set[i][tgt_lang]

        src_ids = tokenizer.encode(german_sentence).ids
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = (src != PAD_ID).unsqueeze(1)

        output_ids = greedy_decode(model, src, src_mask, max_len=50, sos_id=SOS_ID, eos_id=EOS_ID, device=device)
        translation = tokenizer.decode(output_ids)
        stopped_early = len(output_ids) < 50

        print(f"\n[{i}]")
        print(f"German:      {german_sentence}")
        print(f"Predicted:   {translation}")
        print(f"Reference:   {reference}")
        print(f"Hit <eos>:   {stopped_early}")