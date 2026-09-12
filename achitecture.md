# Kiến trúc Model — LogBERT (PyTorch)

Tài liệu này trích xuất kiến trúc hiện tại của model từ `logbert/model/`.

## Tổng quan: `LogBertClassifier` (`logbert/model/logbert.py:65`)

Model đa nhiệm (multi-task): vừa **phân loại chuỗi log** (anomaly / incident
detection) vừa có head **dự đoán log kế tiếp** (causal LM). `forward` trả về một
`dict` thay vì `SequenceClassifierOutput` của HuggingFace.

```
input_ids [B,L]  (+ device_ids tuỳ chọn)
        │
     ┌──▼─────────────────────────┐
     │  BERT backbone             │  → x [B, L, H]
     └──┬─────────────────────────┘
        ├──────────────► CausalLogModel (causal_lm_head)  → loss_causal
        │
   AttentionPooling(x, mask)  → pooled [B,H], alpha [B,L]
        │
     Dropout → cls_head (Linear H→2) → logits [B,2]  → loss_cls
        │
   loss = loss_cls + alpha_causal_lm · loss_causal      (alpha = 0.1)

forward → {logits, loss, loss_cls, loss_causal, attn_weights=alpha}
```

## 1. Backbone `BERT` (`logbert/model/logbert.py:9`)

`BERTEmbedding` + `N × TransformerBlock`. Key-padding mask lấy từ `input_ids > 0`.

### `BERTEmbedding` (`logbert/model/embedding.py:60`)

Tổng của các embedding:

- **`TokenEmbedding`** — `nn.Embedding(vocab_size, H, padding_idx=0)`
- **`PositionalEmbedding`** — sinusoidal sin/cos, **cố định (không học)**, lưu dưới
  dạng buffer với độ dài `max_len`
- **`DeviceEmbedding`** (tuỳ chọn, khi `is_device=True`) —
  `nn.Embedding(num_devices, H, padding_idx=0)`
- `SegmentEmbedding` và `TimeEmbedding` có sẵn code nhưng **đang tắt** (comment out)
- `_safe_clamp_indices` clamp id ngoài vocab để tránh CUDA illegal memory access
  dưới DDP (crash guard thật, không phải defensive noise)

### `TransformerBlock` (`logbert/model/transformer.py:8`)

Kiến trúc **pre-norm** (LayerNorm trước sublayer):

- `MultiHeadedAttention` → `input_sublayer` (residual + LayerNorm)
- `PositionwiseFeedForward` → `output_sublayer` (residual + LayerNorm)
- FFN hidden = `hidden × 2`, activation **GELU**

### `MultiHeadedAttention` (`logbert/model/attention.py:36`)

- 3 × `Linear(d_model, d_model)` cho query/key/value + `output_linear`
- Dùng `F.scaled_dot_product_attention`
- `causal=True` bật `is_causal` → attention một chiều (autoregressive next-log)

> ⚠️ **Lưu ý cần kiểm tra:** trong `Attention.forward` (`attention.py:28`), tham số
> `attn_mask=attn_mask` **đang bị comment out**, nên key-padding mask hiện **không**
> được truyền vào SDPA — chỉ có `is_causal` hoạt động. Cần xác nhận đây có phải chủ
> ý hay không.

## 2. `AttentionPooling` (`logbert/model/logbert.py:33`)

### Vấn đề cần giải quyết

Sau backbone, ta có `x` shape `[B, L, H]` — mỗi chuỗi log là **L vector** (mỗi
token 1 vector H chiều). Nhưng `cls_head` cần **1 vector duy nhất** `[H]` cho cả
chuỗi để phân loại. Vậy phải "nén" trục `L` thành 1 vector.

`AttentionPooling` làm việc đó bằng cách **học** token nào quan trọng rồi lấy
**trung bình có trọng số** (thay cho average pooling coi mọi token như nhau, hoặc
lấy 1 token [CLS] cố định). Đây là kế thừa của `Interpolate + linear1` cũ.

### Cơ chế: 4 bước

Ký hiệu: `B`=batch, `L`=độ dài chuỗi, `H`=hidden.

1. **Chấm điểm mỗi token** — `score = Linear(H, 1, bias=False)` là 1 vector trọng số
   `w [H]`. Với mỗi token: `scores[i] = w · x[i]` (tích vô hướng) → 1 số = "độ liên
   quan". Kết quả `[B, L]`.
2. **Che token không hợp lệ** — `scores.masked_fill(attn_mask == 0, -inf)`.
   `attn_mask` đánh dấu `1`=token thật, `0`=**padding** hoặc **vị trí SOS**
   (dựng ở `logbert.py:105-106`, `attn_mask[:, 0] = 0`). Chỗ `0` bị gán `-inf`.
3. **Chuẩn hóa thành trọng số** — `alpha = softmax(scores, dim=1)`, cộng lại = 1.
   Token bị `-inf` → `exp(-inf) = 0` → `alpha = 0` (loại hoàn toàn). `alpha [B, L]`
   chính là **độ quan trọng mỗi token**, trả ra ngoài (`attn_weights`) cho
   root-cause analysis / `visualize_weights.py`.
4. **Trung bình có trọng số** — `pooled = (alpha.unsqueeze(-1) * x).sum(dim=1)`:
   nhân mỗi token với trọng số của nó rồi cộng dồn theo trục `L` → `[B, H]`.

### Ví dụ số (H=2, L=4, vị trí 0 = SOS bị che)

```
x = [[9,9](SOS), [1,0](A), [0,2](B), [3,3](C)]
scores  → [-inf, 0.5, 0.1, 2.0]
softmax → [0,   0.17, 0.11, 0.72]        (cộng = 1, SOS = 0)
pooled  = 0.17·[1,0] + 0.11·[0,2] + 0.72·[3,3] = [2.33, 2.38]
```
→ token C quan trọng nhất nên vector kết quả nghiêng về nó.

### `init_as_avg_pooling()` — vì sao khởi tạo weight = 0 (không phải 1)?

Cần phân biệt hai thứ khác nhau:

- **`w`** = tham số học được (`score.weight`, cỡ `[H]`), nhân với *nội dung token*
  `x[i]` để ra điểm số. Đây là thứ được init về 0.
- **`alpha`** = kết quả softmax, KHÔNG phải tham số. Đây mới là "trọng số trung bình"
  của từng token. `w = 0` **không** làm `alpha = 0`.

Nếu `w = 0` thì mọi `scores[i] = 0` bất kể `x[i]` → softmax ra **đều nhau**
`1/(số token thật)`. Đó chính là average pooling (mỗi `alpha = 1/N`, cộng lại = 1 —
**không phải** `alpha = 1` cho mọi token, vì thế thì tổng = N, hết là "trung bình").

Vì sao **không** phải `w = 1`? Vì `w = 1` làm `scores[i] = tổng feature của x[i]` —
phụ thuộc nội dung token nên mỗi token ra điểm khác nhau → softmax lệch, không đều.
Chỉ khi scores bằng nhau ở mọi token thì softmax mới đều, mà cách duy nhất để scores
luôn bằng nhau bất kể `x[i]` là `w = 0`.

```
w = [1,1]: scores [1, 2, 6]  → softmax [0.006, 0.018, 0.976]  ❌ lệch hẳn về 1 token
w = [0,0]: scores [0, 0, 0]  → softmax [0.33,  0.33,  0.33 ]  ✅ đều = average
           (ví dụ token A=[1,0], B=[0,2], C=[3,3])
```

**Vẫn học được gì?** `w = 0` chỉ là điểm xuất phát, KHÔNG bị đóng băng
(`score.weight` vẫn `requires_grad=True`). Ngay tại `w = 0`, gradient `∂loss/∂w`
vẫn khác 0 nên train đẩy `w` rời khỏi 0; khi `w ≠ 0`, scores khác nhau → `alpha`
lệch → model đã học được token nào quan trọng. Ý tưởng: **bắt đầu từ baseline trung
tính (avg pooling) rồi học phần lệch ra từ đó** — mẹo init phổ biến (giống LoRA B=0,
zero-init cổng residual), ổn định hơn init ngẫu nhiên.

> Thứ tự init: `self.apply(self._init_weights)` (gán `normal(0,0.02)` cho mọi Linear)
> chạy trước, rồi `self.pool.init_as_avg_pooling()` chạy **sau** ghi đè `score.weight`
> về 0 — nên avg-init luôn thắng.

## 3. Hai head

- **`cls_head`** — `Linear(H → num_labels=2)`, loss `CrossEntropyLoss`
  (hỗ trợ `class_weight` **chỉ khi train**)
- **`causal_lm_head` = `CausalLogModel`** (`logbert/model/logbert.py:52`) —
  `Linear(H → vocab_size)` + `LogSoftmax`, loss `NLLLoss(ignore_index=0)`;
  chỉ tính khi `use_causal_lm=True`

Tổng loss: `loss = loss_cls + alpha_causal_lm · loss_causal`.

## 4. Khởi tạo & chi tiết

- `_init_weights`: `normal(0, 0.02)` cho Linear/Embedding, bias = 0
  (mirror HF `post_init`), áp dụng **sau** avg-init của pooling
- `LayerNorm` tự viết (`logbert/model/utils.py:12`), không dùng `nn.LayerNorm`
- `set_class_weight` đăng ký buffer `class_weight` (chỉ dùng ở train mode);
  `_save_final` loại bỏ nó khỏi `model_final.pt`

## Cấu hình mặc định (`logbert/config.py` — `ModelConfig`)

| Field                | Default | Ghi chú                                        |
|----------------------|---------|------------------------------------------------|
| `vocab_size`         | —       | điền lúc runtime từ `len(vocab)`               |
| `hidden`             | 256     | H                                              |
| `layers`             | 1       | số `TransformerBlock`                          |
| `attn_heads`         | 4       |                                                |
| `max_seq_len`        | 36000   | buffer cho `PositionalEmbedding`               |
| `dropout`            | 0.1     |                                                |
| `causal`             | True    | attention một chiều                            |
| `use_causal_lm`      | True    | bật head dự đoán log kế tiếp                    |
| `alpha_causal_lm`    | 0.1     | trọng số của causal LM loss                    |
| `num_labels`         | 2       |                                                |
| `is_device`          | False   | bật device embedding                           |
| `num_devices`        | 1       | điền lúc runtime từ `len(device_vocab)`        |
