# CamemBERT: a French BERT

## Introduction

CamemBERT is a pretrained language model trained on 138GB of French text based on RoBERTa.

Also available in [github.com/huggingface/transformers](https://github.com/huggingface/transformers/).

## Pre-trained models

| Model                          | #params | vocab size | Download                                                                                                                 | Description                                                   |
|--------------------------------|---------|------------|--------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| `camembert` / `camembert-base` | 110M    | 32k        | [camembert-base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/camembert-base.tar.gz)                             | Base model trained on OSCAR (138 GB of raw text)              |
| `camembert-large`              | 335M    | 32k        | [camembert-large.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/camembert-large.tar.gz)                           | Large model trained on CCNet (135 GB of raw text)             |
| `camembert-base-ccnet`         | 110M    | 32k        | [camembert-base-ccnet.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/camembert-base-ccnet.tar.gz)                 | Base model trained on CCNet (135 GB of raw text)              |
| `camembert-base-wikipedia-4gb` | 110M    | 32k        | [camembert-base-wikipedia-4gb.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/camembert-base-wikipedia-4gb.tar.gz) | Base model trained on Wikipedia (4 GB of raw text)            |
| `camembert-base-oscar-4gb`     | 110M    | 32k        | [camembert-base-oscar-4gb.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/camembert-base-oscar-4gb.tar.gz)         | Base model trained on a subsample of OSCAR (4 GB of raw text) |
| `camembert-base-ccnet-4gb`     | 110M    | 32k        | [camembert-base-ccnet-4gb.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/camembert-base-ccnet-4gb.tar.gz)         | Base model trained on a subsample of CCNet (4 GB of raw text) |

## Example usage

### fairseq
##### Load CamemBERT from torch.hub (PyTorch >= 1.1):
```python
import torch
camembert = torch.hub.load('pytorch/fairseq', 'camembert')
camembert.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Load CamemBERT (for PyTorch 1.0 or custom models):
```python
# Download camembert model
wget https://dl.fbaipublicfiles.com/fairseq/models/camembert.tar.gz
tar -xzvf camembert.tar.gz

# Load the model in fairseq
from fairseq.models.roberta import CamembertModel
camembert = CamembertModel.from_pretrained('/path/to/camembert')
camembert.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Filling masks:
```python
masked_line = 'Le camembert est <mask> :)'
camembert.fill_mask(masked_line, topk=3)
# [('Le camembert est délicieux :)', 0.4909118115901947, ' délicieux'),
#  ('Le camembert est excellent :)', 0.10556942224502563, ' excellent'),
#  ('Le camembert est succulent :)', 0.03453322499990463, ' succulent')]
```

##### Extract features from Camembert:
```python
# Extract the last layer's features
line = "J'aime le camembert !"
tokens = camembert.encode(line)
last_layer_features = camembert.extract_features(tokens)
assert last_layer_features.size() == torch.Size([1, 10, 768])

# Extract all layer's features (layer 0 is the embedding layer)
all_layers = camembert.extract_features(tokens, return_all_hiddens=True)
assert len(all_layers) == 13
assert torch.all(all_layers[-1] == last_layer_features)
```
