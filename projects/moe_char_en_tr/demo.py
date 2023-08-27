from mingpt.model import GPT
import argparse
import pathlib 
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
import json
import torch
from projects.moe_char_en_tr.chargpt import CharDataset

set_seed(3407)


if __name__ == "__main__":

    #get work dir path using argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, help="path to work dir")
    args = parser.parse_args()
    work_dir = pathlib.Path(args.work_dir)

    #load config.json from work dir
    config = json.load(open(work_dir / "config.json"))

    model_config_node = CN(**config['model'])
    dataset_config_node = CN(**config['data'])
    
    #print config
    print(json.dumps(config, indent=2))

    #list files in checkpoints dir
    checkpoints = list((work_dir / "checkpoints").glob("*.pt"))
    checkpoints.sort(key=lambda x: float(x.stem.split("_")[-1]))
    checkpoint = checkpoints[0] #pick best checkpoint

    print(f"loading model from {checkpoint}")

    #load model
    model = GPT(config=model_config_node)
    model.load_state_dict(torch.load(checkpoint))

    #load dataset so that we can use stoi and itos
    text = open('en-tr/turkish-train.txt', 'r').read()
    full_dataset = CharDataset(CN(**config['data']), text) #TODO, save tokenizer as a part of the model or in config

    #generate text
    context = "\n"
    x = torch.tensor([full_dataset.stoi[s] for s in context], dtype=torch.long)[None,...]
    y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
    completion = ''.join([full_dataset.itos[int(i)] for i in y])
    print(completion)






    
