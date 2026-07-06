from logbert.config import ModelConfig, DataConfig, TrainConfig

def test_model_config_defaults():
    m = ModelConfig(vocab_size=100)
    assert m.hidden == 256 and m.layers == 1 and m.attn_heads == 4
    assert m.causal is True and m.use_causal_lm is True
    assert m.alpha_causal_lm == 0.1
    assert m.max_seq_len == 36000 and m.num_labels == 2
    assert m.is_device is False and m.num_devices == 1

def test_train_config_defaults():
    t = TrainConfig(output_dir="/tmp/x")
    assert t.epochs == 20 and t.lr == 5e-5 and t.max_grad_norm == 1.0
    assert t.warmup_ratio == 0.1 and t.scheduler_type == "linear"
    assert t.grad_accum_steps == 32 and t.bf16 is True
    assert t.eval_every == 0 and t.save_every == 0   # 0 = per epoch
    assert t.early_stop_patience == 4 and t.min_delta == 0.002
    assert t.resume_from is None and t.log_initial_loss is True

def test_data_config_defaults():
    d = DataConfig(train_dir="a", vocab_path="b", device_vocab_path="c")
    assert d.seq_len == 36000 and d.mask_ratio == 0.0
    assert d.batch_size == 8
