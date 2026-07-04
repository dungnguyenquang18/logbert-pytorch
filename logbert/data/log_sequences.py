"""Default data type: device-incident log sequences from the old repo's PKL parts."""
import glob
import os
import pickle
import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def load_pkl_dir(folder_path: str) -> list:
    files = glob.glob(os.path.join(folder_path, "**", "*.pkl"), recursive=True)
    seqs = []
    for file in tqdm(sorted(files), desc=f"Loading pkl from {folder_path}"):
        with open(file, "rb") as f:
            seqs.extend(pickle.load(f))
    return seqs


def _normalize_tuple(item):
    """PKL tuples come as (tokens, window, dev, label) with 2/3-element variants."""
    if len(item) == 4:
        k, window, d, y = item
    elif len(item) == 3:
        k, window, d = item
        y = -1
    elif len(item) == 2:
        k, d = item
        window, y = [], -1
    else:
        raise ValueError(f"Unexpected sequence tuple of len {len(item)}")
    return k, window, d, int(y)


class LogSequenceDataset(Dataset):
    def __init__(self, sequences, vocab, mask_ratio: float = 0.0, predict_mode: bool = False):
        self.sequences = sequences
        self.vocab = vocab
        self.mask_ratio = mask_ratio
        self.predict_mode = predict_mode
        self.lengths = [len(_normalize_tuple(s)[0]) + 1 for s in sequences]  # +1 for SOS

    def __len__(self):
        return len(self.sequences)

    def _to_id(self, tok):
        if isinstance(tok, int):
            return tok if 0 <= tok < len(self.vocab) else self.vocab.unk_index
        return self.vocab.stoi.get(tok, self.vocab.unk_index)

    def _mask_inputs(self, tokens):
        """Optionally perturb INPUT tokens (BERT-style). Labels are NOT produced
        here — the collator always builds causal next-token labels, matching the
        old pipeline where collate_fn overwrote the MLM labels."""
        if self.mask_ratio <= 0:
            return [self._to_id(t) for t in tokens]
        out = []
        for tok in tokens:
            prob = random.random()
            if prob < self.mask_ratio:
                if self.predict_mode:
                    out.append(self.vocab.mask_index)
                    continue
                prob /= self.mask_ratio
                if prob < 0.8:
                    out.append(self.vocab.mask_index)
                elif prob < 0.9:
                    out.append(random.randrange(len(self.vocab)))
                else:
                    out.append(self._to_id(tok))
            else:
                out.append(self._to_id(tok))
        return out

    def __getitem__(self, idx):
        k, window, d, y = _normalize_tuple(self.sequences[idx])
        tokens = [self.vocab.sos_index] + self._mask_inputs(k)
        dev_cls = d[0] if len(d) > 0 else 0
        device_ids = [dev_cls] + list(d[: len(tokens) - 1])
        device_ids += [self.vocab.pad_index] * (len(tokens) - len(device_ids))
        return {
            "tokens": tokens,
            "device_ids": device_ids,
            "label": y,
            "window_end": int(window[0]) if window else 0,
        }


class LogCollator:
    def __init__(self, vocab, seq_len=None):
        self.vocab = vocab
        self.seq_len = seq_len

    def __call__(self, samples):
        cap = max(len(s["tokens"]) for s in samples)
        if self.seq_len is not None:
            cap = min(cap, self.seq_len)
        pad = self.vocab.pad_index
        input_ids, causal_labels, device_ids, labels, window_end = [], [], [], [], []
        for s in samples:
            k = s["tokens"][:cap]
            lab = k[1:] + [self.vocab.eos_index]          # next-token target
            d = s["device_ids"][:cap]
            npad = cap - len(k)
            input_ids.append(k + [pad] * npad)
            causal_labels.append(lab + [pad] * npad)
            device_ids.append(d + [pad] * (cap - len(d)))
            labels.append(s["label"])
            window_end.append(s["window_end"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "causal_labels": torch.tensor(causal_labels, dtype=torch.long),
            "device_ids": torch.tensor(device_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "window_end": torch.tensor(window_end, dtype=torch.long),
        }
