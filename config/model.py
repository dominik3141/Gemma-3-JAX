from types import SimpleNamespace

gemma_3_27b = SimpleNamespace(
    num_attention_heads=32,
    num_key_value_heads=16,
    num_queries_per_group=2,
    num_layers=62,
    d_model=5376,
    d_kvq=128,
    head_dim=128,
    d_mlp=21504,
    sliding_window=1024,
)

gemma_3_1b = SimpleNamespace(
    num_attention_heads=4,
    num_key_value_heads=1,
    num_queries_per_group=4,
    num_layers=26,
    d_model=1152,
    d_kvq=256,
    head_dim=256,
    d_mlp=6912,
    sliding_window=1024,
)
