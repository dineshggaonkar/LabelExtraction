import torch
import numpy as np
#import config
import utils
import yaml
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from sklearn.metrics import f1_score, accuracy_score

parser = argparse.ArgumentParser(description="path to config file")
parser.add_argument("-p", "--path", help="Path")
args = parser.parse_args()


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config


config = load_config(args.path)

dataset = load_dataset('csv', data_files={'train': config["train_data_path"], 'valid': config["valid_data_path"]})

tokenizer = AutoTokenizer.from_pretrained(config["pretrained_lm"])


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


encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)

encoded_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(config["pretrained_lm"],
                                                           problem_type=config["problem_type"],
                                                           num_labels=len(config["labels"]),
                                                           id2label=config["id2label"],
                                                           label2id=config["label2id"])

args = TrainingArguments(
    output_dir = config["model_save_path"],
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate = config["learning_rate"],
    per_device_train_batch_size = config["train_batch_size"],
    per_device_eval_batch_size = config["test_batch_size"],
    num_train_epochs = config["num_train_epochs"],
    weight_decay = config["weight_decay"],
    save_total_limit = 1,
    seed=0,
    metric_for_best_model = config["metric"],
    logging_dir = config["log_path"],
)


def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'f1': f1_micro_average,
               'accuracy': accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result


trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["valid"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(config['model_save_path'])

#trainer.evaluate()
