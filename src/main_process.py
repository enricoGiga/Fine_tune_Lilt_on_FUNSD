
from functools import partial

from datasets import Features, Sequence, Value, Array2D, load_dataset
from datasets.formatting.formatting import LazyBatch

from transformers import LayoutLMv3Processor, LayoutLMv3FeatureExtractor, LayoutLMv3Tokenizer, LayoutLMv3ImageProcessor, \
    AutoTokenizer, AutoModel

from lilt_fine_tune.build_trainer import build_trainer

if __name__ == "__main__":
    dataset = load_dataset("./lilt_fine_tune/funsdloader.py")
    label_list = dataset["train"].features["ner_tags"].feature.names

    column_names = dataset["train"].column_names

    # we need to define custom features for `set_format` (used later on) to work properly
    features = Features({
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'labels': Sequence(feature=Value(dtype='int64')),
    })



    tokenizer = AutoTokenizer.from_pretrained("C:\\Users\\enrico\\PycharmProjects\\fine_tune_Lilt_on_FUNSD\\data\\models\\Lilt")


    def prepare_dataset(sample):

        ## sample can be a single dict, or a list of dict, depending upon if we don't use batch or use batch
        if isinstance(sample, dict):
            tokens = sample["words"]
            bbox = sample["bboxes"]
            ner_tags = sample["ner_tags"]

            encoding = tokenizer(text=tokens, boxes=bbox, word_labels=ner_tags, truncation=True, padding="max_length")

        elif isinstance(sample, LazyBatch):
            tokens = sample["words"]
            bbox = sample["bboxes"]
            ner_tags = sample["ner_tags"]

            encoding = tokenizer(text=tokens, boxes=bbox, word_labels=ner_tags, truncation=True, padding="max_length")

        else:
            tokens = [item["words"] for item in sample]
            bbox = [item["bboxes"] for item in sample]
            ner_tags = [item["ner_tags"] for item in sample]

            encoding = tokenizer(text=tokens, boxes=bbox, word_labels=ner_tags, truncation=True, padding="max_length")

        return encoding


    train_dataset = dataset["train"].map(
        prepare_dataset,
        batched=True,
        remove_columns=column_names,
        features=features,
    )

    eval_dataset = dataset["test"].map(
        prepare_dataset,
        batched=True,
        remove_columns=column_names,
        features=features,
    )
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'bbox', 'labels'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'bbox', 'labels'])

    ckpt_filename = "lilt_best"
    build_trainer(labels=label_list, train_dataset=train_dataset, eval_dataset=eval_dataset,
                  checkpoint_filename=ckpt_filename)
    # evaluate_model(label_list, train_dataset=train_dataset, eval_dataset=eval_dataset,checkpoint_filename=ckpt_filename)
    # inference(eval_dataset=eval_dataset, dataset=dataset, labels=label_list, checkpoint_filename=ckpt_filename)

