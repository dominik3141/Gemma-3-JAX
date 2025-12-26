from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int = 1152
    n_layers: int = 26
    n_query_heads: int = 4
    n_kv_heads: int = 1
    head_dim: int = 256
    vocab_size: int = 262144
    d_mlp: int = 6912
    # soft config (changing these will not break anything, but might degrade accuracy)
    local_window_size: int = 1024
    max_context_length: int = 32768
