# LiLT (ACL 2022)

LiLT is pre-trained on the visually-rich documents of a single language (English) and can be directly fine-tuned on other languages with the corresponding off-the-shelf monolingual/multilingual pre-trained textual models. We hope the public availability of this work can help document intelligence researches.

## Installation

For CUDA 11.X: 

~~~bash
pip install -r requirements.txt
~~~
Or check [PyTorch](https://pytorch.org/get-started/previous-versions/) versions and modify the --extra-index-url line accordingly.

## Datasets

In this repository, we provide the fine-tuning codes for FUNSD dataset.


## Available pre-trained model combined with RoBERTa

| Model                         | Language  | Size  | Download                                                                          | 
| ----------------------------- | --------- | ----- |-----------------------------------------------------------------------------------|
| `lilt-roberta-en-base`        | EN        | 293MB | [HuggingFace](https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base/tree/main) | 


## Fine-tuning


### Semantic Entity Recognition on FUNSD

~~~pycon
runfile('path/to/main_process.py', wdir='pat/to/src/folder')
~~~

### Multi-task Semantic Entity Recognition on FUNSD


### Multi-task Relation Extraction on FUNSD



## Results

### Semantic Entity Recognition on FUNSD
overall_recall has value:      0.7021806853582554
overall_f1 has value:          0.6592570927171687

### Language-specific Fine-tuning on FUNSD

### Cross-lingual Zero-shot Transfer on FUNSD

### Multitask Fine-tuning on FUNSD




## Citation
```
@inproceedings{wang-etal-2022-lilt,
    title = "{L}i{LT}: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding",
    author={Wang, Jiapeng and Jin, Lianwen and Ding, Kai},
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.534",
    doi = "10.18653/v1/2022.acl-long.534",
    pages = "7747--7757",
}
```

