from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import utils
import yaml

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("./train_config.yaml")

tokenizer = AutoTokenizer.from_pretrained(config["model_save_path"])

model = AutoModelForSequenceClassification.from_pretrained(config["model_save_path"])

text = "Turn the lamp on in washroom"

input_ids = tokenizer.encode(text, return_tensors='pt')

with torch.no_grad():
    output = model(input_ids).logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(output.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    for i in utils.get_out_labels(probs):
        predictions[i] = 1

    pred_labels = [config["id2label"][idx] for idx, label in enumerate(predictions) if label == 1.0]
    print(pred_labels)
