# LabelExtraction
Transformer based multi label classification

### Setup environment and install requirements
```commandline
python3 -m venv venv_label_extraction
source venv_label_extraction/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Train label extraction model
```commandline
python3 train.py --path ./train_config.yaml
```

### Evaluate model on test data
```commandline
python3 test.py --path ./data/test_data.csv 
```

### Inference on single utterence
##### Start the label extraction server
```commandline
python3 app.py
```
#### send request using curl
```commandline
curl -X POST http://localhost:8017/get-labels -H "Content-Type: application/json" -d '{"text":"Switch on the kitchen lights"}'
```
sample output : ["activate","lights","kitchen"]

#### tensorboard logs
```commandline
https://drive.google.com/drive/folders/1---4NW79iBzisE851RlpDF-zS3IxIc-r?usp=sharing
```