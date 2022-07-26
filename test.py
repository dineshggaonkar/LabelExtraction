import torch
import numpy as np
import argparse
import utils
import yaml

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset

parser = argparse.ArgumentParser(description="path to test_data.csv")
parser.add_argument("-p", "--path", help="Path")
args = parser.parse_args()

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("./train_config.yaml")

dataset = load_dataset('csv', data_files={'test': args.path})

tokenizer = AutoTokenizer.from_pretrained(config["model_save_path"])


def preprocess_data(examples):
    text = examples["transcription"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=config["max_length"])
    action = examples["action"]
    obj = examples["object"]
    location = examples["location"]
    labels_matrix = np.zeros((len(text), len(config["labels"])))
    for index, i, j, k in zip(range(len(text)), action, obj, location):
        for i in utils.get_id(i, j, k, config):
            labels_matrix[index][i] = 1

    encoding["labels"] = labels_matrix.tolist()

    return encoding


encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['test'].column_names)
encoded_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(config["model_save_path"],
                                                           problem_type=config["problem_type"],
                                                           num_labels=len(config["labels"]),
                                                           id2label=config["id2label"],
                                                           label2id=config["label2id"])

args = TrainingArguments(
    output_dir=config["model_save_path"],
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=config["test_batch_size"],
    num_train_epochs=config["num_train_epochs"],
    weight_decay=config["weight_decay"],
    metric_for_best_model=config["metric"],
)


def multi_label_metrics(predictions, labels):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = []
    for prob in probs:
      y_pred_sample = np.zeros(prob.shape)
      for i in utils.get_out_labels(prob):
        y_pred_sample[i] = 1
      y_pred.append(y_pred_sample.tolist())
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'f1': f1_micro_average,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    print(f"len of preds = {len(preds)}")
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result

trainer = Trainer(
    model,
    args,
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print(trainer.evaluate())