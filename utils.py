import torch

def get_id(action, obj, location, config):
    if obj=='none':
        obj="none_o"
    if location=='none':
        location="none_l"
    a_id = config["label2id"][action]
    o_id = config["label2id"][obj]
    l_id = config["label2id"][location]
    return [a_id, o_id, l_id]

def get_out_labels(probs):
    return [torch.argmax(probs[0:6]).item(), torch.argmax(probs[6:20]).item()+6, torch.argmax(probs[20:24]).item()+20]