
import evaluate
import torch
from tqdm.auto import tqdm

from lilt_fine_tune.data_module import DataModule
from lilt_fine_tune.Lit_module import LitModule
from lilt_fine_tune.prediction_utility import get_labels


def evaluate_model(label_list, train_dataset, eval_dataset, checkpoint_filename):
    pl_dl = DataModule(train_dataset=train_dataset, eval_dataset=eval_dataset)
    pl_model = LitModule(label_list)

    checkpoint = torch.load(f"/workspace/enrico/LilLt/data/checkpoint/{checkpoint_filename}.ckpt")
    pl_model.load_state_dict(checkpoint['state_dict'])

    evaluate_results(label_list, pl_dl, pl_model)


def evaluate_results(label_list, pl_dl, pl_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_metric = evaluate.load("seqeval")
    pl_model.eval()
    model = pl_model.model.to(device)
    for idx, batch in enumerate(tqdm(pl_dl.val_dataloader())):
        # move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(batch)

        predictions = outputs['logits'].argmax(-1)
        true_predictions, true_labels = get_labels(label_list, predictions, batch["labels"])
        eval_metric.add_batch(references=true_labels, predictions=true_predictions)
    results = eval_metric.compute()
    # for key in ['overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']:
    for key in ['ART_NO', 'ESCRIPTION', 'ESC_CODE', 'NIT_PRICE', 'TEM_NO', 'UANTITY', 'UESTION']:
        print(key + ":")
        for value in results[key]:
            print_statement = '{0: <30}'.format(str(value) + " has value:")
            print(print_statement, results[key][value])


def inference(eval_dataset, dataset, labels, checkpoint_filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl_model = LitModule(labels)
    checkpoint_filename = torch.load(f"/workspace/enrico/LilLt/data/checkpoint/{checkpoint_filename}.ckpt")
    pl_model.load_state_dict(checkpoint_filename['state_dict'])
    model = pl_model.model.to(device)
    for k, eval_value in enumerate(eval_dataset):

        sample = eval_value
        for key in list(sample.keys()):
            sample[key] = sample[key].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(sample)
        predictions = outputs['logits'].argmax(-1)
        pad_token_id = 0
        for i, j in enumerate(sample["input_ids"][0]):
            if j == 1:
                pad_token_id = i
                break

        predictions = predictions.squeeze(0)[:pad_token_id]
        actual_prediction = [i.item() for i in predictions]

        for key in list(sample.keys()):
            sample[key] = sample[key].squeeze(0)

        sample['ner_tags'] = actual_prediction

        sample['image'] = dataset['test'][k]['image'].convert("RGB").resize((1000, 1000))
        sample.pop("attention_mask")
        sample['bboxes'] = sample.pop("bbox").tolist()
        img = plot_visualization(sample, labels)

        # plt.imshow(img)
        # plt.show()

        img.save(f"/workspace/enrico/LilLt/data/images/{k}.png")


## Visualizing the bounding boxes

def plot_visualization(sample: dict, labels):
    from PIL import ImageDraw, ImageFont

    img = sample["image"]
    bbox = sample["bboxes"]
    ner_tags = sample["ner_tags"]

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    converter = {
        0: "O",
        1: "DESCRIPTION",
        2: "ITEM_NO",
        3: "PART_NO",
        4: "QUANTITY",
        5: "MESC_CODE",
        6: "UNIT_PRICE",
        7: "QUESTION",
    }
    label2color = {
        0: "pink",
        1: "yellow",
        2: "green",
        3: "blue",
        4: "red",
        5: "green",
        6: "grey",
        7: "black",
    }

    for box, predicted_label in zip(bbox, ner_tags):
        ## The bounding box has been rescaled to the range of [0, 1000] considering an image is [1000, 1000]
        box[0] = int(img.size[0] * box[0] / 1000)
        box[1] = int(img.size[1] * box[1] / 1000)
        box[2] = int(img.size[0] * box[2] / 1000)
        box[3] = int(img.size[1] * box[3] / 1000)
        draw.rectangle(box, outline="black")
        draw.rectangle(box, outline=label2color[predicted_label])
        draw.text((box[0] + 10, box[1] - 10), text=converter[predicted_label], fill=label2color[predicted_label],
                  font=font)

    return img


