# Regression-based-Analytic-Incremental-Learning
Official implementation of [Advancing Cross-domain Discriminability in Continual Learning of Vision-Language Models](https://arxiv.org/pdf/2406.18868)

The paper has accepted by NeurIPS 2024.

### Installation

Create a conda environment and install dependencies:

```bash
git clone https://github.com/linghan1997/Regression-based-Analytic-Incremental-Learning.git
cd RAIL

conda create -n rail python=3.8
conda activate tip_adapter

pip install -r requirements.txt
```

### Data Preparation

We suggest putting all required datasets under the folder
```bash
RAIL/
|-- datasets/
```

Please refer to the following guides for setting up datasets:
[CoOp](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md)

## Acknowledgement

Our repo benefits from [CLIP](https://github.com/openai/CLIP) and [CoOp](https://github.com/KaiyangZhou/CoOp). We thank them for their wonderful works.

---
