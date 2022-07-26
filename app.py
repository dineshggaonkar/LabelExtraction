import torch
import uvicorn
import numpy as np
import utils
import yaml

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("./train_config.yaml")

class ExtractLabels(BaseModel):
    text: str


tokenizer = AutoTokenizer.from_pretrained(config["model_save_path"])

model = AutoModelForSequenceClassification.from_pretrained(config["model_save_path"])


@app.post('/get-labels')
def get_labels(extract_labels: ExtractLabels):
    input_ids = tokenizer.encode(extract_labels.text, return_tensors='pt')
    output = model(input_ids).logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(output.squeeze().cpu())
    print(probs)
    predictions = np.zeros(probs.shape)
    for i in utils.get_out_labels(probs):
        predictions[i] = 1
    pred_labels = [config["id2label"][idx] for idx, label in enumerate(predictions) if label == 1.0]

    return pred_labels


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8017)


#curl -X POST http://localhost:8017/get-labels -H "Content-Type: application/json" -d '{"text":"Switch on the kitchen lights"}'
