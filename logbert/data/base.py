"""Data contract for pluggable data types.

A data type module provides two things:

1. A torch Dataset whose __getitem__ returns a dict:
       tokens:      list[int]   token ids, SOS already prepended
       device_ids:  list[int]   same length as tokens
       label:       int         sequence label, -1 if unknown
       window_end:  int         unix ts of the window end, 0 if N/A
   and exposes `lengths: list[int]` (per-sample token count) so
   BucketBatchSampler can bucket by length without touching items.

2. A collator, Callable[[list[dict]], dict[str, Tensor]], producing:
       input_ids   LongTensor [B, L]
       mlm_labels  LongTensor [B, L]   next-token labels, pad = 0
       device_ids  LongTensor [B, L]
       labels      LongTensor [B]
       window_end  LongTensor [B]

Trainer and the predict script depend ONLY on this batch dict, so a new
data type = one new file implementing this contract; no trainer changes.
"""
