# Master thesis

RIA-Lenta cluster dataset: https://disk.yandex.com/d/-Bxxa59MVwtMYQ


### Alignement method

Below are examples for Lenta & RIA datasets.

1. Train a Bottleneck Encoder-Decoder model

```bash
python src/train_gen_title.py --run-name lenta-ria-clustering-model --config-file configs/gen_title.jsonnet --train-file /datasets/ --dataset-type lenta-ria --output-model-path model_checkpoints/lenta_ria_clustering_model --enable-bottleneck
```

2. Construct a cluster dataset

```bash
python make_lenta_ria_dataset.py
```

