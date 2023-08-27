"""
Trains a character-level language model.
"""

import os
import sys
import pathlib
import json

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

from datetime import datetime
from pathlib import Path
# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = f'./out/moe-en-tr-{datetime.now().strftime("%m-%d-%H:%M:%S")}'

    # data
    C.data = CharDataset.get_default_config()
    C.data.validation_split = 5e-5
    C.data.dataset_path = './opus-100-corpus/v1.0/supervised/en-tr/opus.en-tr-train.tr'
    C.data.encoder_file = './opus-100-corpus/v1.0/supervised/en-tr/encoder.json'


    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gopher-44m'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 1e-4 # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------


class CharTokenizer:
    def __init__(self, encoder):
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}

    def __call__(self, text):
        # single string input for now, in the future potentially a list of strings
        assert isinstance(text, str)
        # encode and create a "batch dimension" of 1
        idx = torch.tensor([self.encoder[s] for s in text], dtype=torch.long)
        return idx
    
    def decode(self, idx):
        # ensure a simple 1D tensor for now
        assert idx.ndim == 1
        # decode indices to text
        text = ''.join([self.decoder[int(i)] for i in idx])
        return text
    
    def get_vocab_size(self):
        return len(self.encoder)

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 64
        return C

    def __init__(self, config, tokenizer):
        self.config = config
        
        # Load the dataset
        with open(config.dataset_path, 'r', encoding='utf-8') as f:
            data = f.read()

        data_size, vocab_size = len(data), tokenizer.get_vocab_size()
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = self.tokenizer(chunk)
        # return as tensors
        x = dix[:-1].clone().detach()
        y = dix[1:].clone().detach()
        return x, y



# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    set_seed(config.system.seed)

    # construct the training dataset
    encoder_path = Path(config.data.encoder_file)
    with encoder_path.open('r') as f:
        tokenizer = CharTokenizer(json.load(f))

    full_dataset = CharDataset(config.data, tokenizer)
    # construct the model
    config.model.vocab_size = full_dataset.get_vocab_size()
    config.model.block_size = full_dataset.get_block_size()
    model = GPT(config.model)

    #save config 
    setup_logging(config)
    print(config)

    # split the dataset into train and test
    train_size = int((1.0 - config.data.validation_split) * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    #print sizes of datasets
    print(f"train size: {train_size}, test size: {test_size}")

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset, test_dataset)

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 100 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                context = "\n"
                x = tokenizer(context)[None,...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = ''.join(tokenizer.decode(y))
                print(completion)

            #compute validation loss
            trainer.validate()

            # check current saved models
            checkpoint_dir = pathlib.Path(config.system.work_dir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            existing_models = list(checkpoint_dir.iterdir())

            #save this model if its validation loss is good enough
            should_save = False
            if len(existing_models) >= 5:
                # sort models by their reported loss in filename
                sorted_models = sorted(existing_models, key=lambda x: float(x.stem.split("_loss_")[1]))
                highest_loss_model = sorted_models[-1]
                highest_loss = float(highest_loss_model.stem.split("_loss_")[1])

                # if this model's validation loss is lower than the highest among top 3
                if trainer.val_loss < highest_loss:
                    # delete the file with the highest loss
                    highest_loss_model.unlink()
                    should_save = True
            else:
                should_save = True

            if should_save:
                print("saving model...")
                ckpt_path = checkpoint_dir / f"model_iter_{trainer.iter_num}_loss_{trainer.val_loss:.2f}.pt"

                torch.save(model.state_dict(), ckpt_path)
                # revert model to training mode
                model.train()
   
                
        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}, validation loss {trainer.val_loss:.5f}")


    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
