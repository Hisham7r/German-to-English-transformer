from functools import partial
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# Think of the "class TranslationDataset"  a numbered shelf of examples.
#  Once built, you can say "give me example #57"  
#  and it hands you back one German sentence and  
#  one English sentence, already converted to numbers.

class TranslationDataset(Dataset):
    def __init__(self, split, tokenizer, sos_id, eos_id, source_lang, target_lang):
        self.source = []
        self.target = []
        for example in split:
            src_ids = [sos_id] + tokenizer.encode(example[source_lang]).ids + [eos_id]
            tgt_ids = [sos_id] + tokenizer.encode(example[target_lang]).ids + [eos_id]
            self.source.append(torch.tensor(src_ids, dtype=torch.long))
            self.target.append(torch.tensor(tgt_ids, dtype=torch.long))

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        target = self.target[idx]
        decoder_input = target[:-1]
        decoder_target = target[1:]
        return self.source[idx], decoder_input, decoder_target

#Padding is necessary because the sentences in a batch can have different lengths.
def collate_fn(batch, pad_id):
    src_batch, decoder_input_batch, decoder_target_batch = zip(*batch)

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_id) #padding_value=pad_id → fill the gaps with 0.
    decoder_input_padded = pad_sequence(decoder_input_batch, batch_first=True, padding_value=pad_id)
    decoder_target_padded = pad_sequence(decoder_target_batch, batch_first=True, padding_value=pad_id)

    src_padding_mask = src_padded != pad_id
    decoder_input_padding_mask = decoder_input_padded != pad_id

    return src_padded, decoder_input_padded, decoder_target_padded, src_padding_mask, decoder_input_padding_mask


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]

    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tokenizer = Tokenizer.from_file(str(ROOT / "data" / "bpe_tokenizer.json"))
    special = config["tokenizer"]["special_tokens"]
    sos_id = tokenizer.token_to_id(special["sos"])
    eos_id = tokenizer.token_to_id(special["eos"])
    pad_id = tokenizer.token_to_id(special["pad"])

    ds = load_dataset(config["dataset"]["name"], split="train")

    train_dataset = TranslationDataset(
        ds,
        tokenizer,
        sos_id,
        eos_id,
        config["dataset"]["source_lang"],
        config["dataset"]["target_lang"],
    )

    loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_id=pad_id),
    )

    src, decoder_input, decoder_target, src_mask, decoder_input_mask = next(iter(loader))
    print("source batch shape:", src.shape)
    print("decoder_input batch shape:", decoder_input.shape)
    print("decoder_target batch shape:", decoder_target.shape)
    print("decoder_input:\n", decoder_input)
    print("decoder_target:\n", decoder_target)
