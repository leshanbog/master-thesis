{
    "tokenizer_model_path": "/data/aobuhtijarov/models/rubert_cased_L-12_H-768_A-12_pt",
    "enc_model_path": "/data/aobuhtijarov/models/pretrained_enc_8_layers",
    "dec_model_path": "/data/aobuhtijarov/models/pretrained_dec_6_layers",
    "agency_list": ["ТАСС", "РИАМО", "RT на русском", "Новости Мойка78"],
    "max_tokens_text": 250,
    "max_tokens_title": 48,
    "gradient_accumulation_steps": 50,
    "batch_size": 4,
    "eval_steps": 500,
    "save_steps": 500,
    "logging_steps": 100,
    "enc_lr": 0.00002,
    "dec_lr": 0.0002,
    "warmup_steps": 1800,
    "max_steps": 20000,
}
