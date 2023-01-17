import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import time
import os

from utils import utils
from data import preprocess

def pretrain():
    start_time = time.time()
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels = 6) # label개수에 따라 변경
    model.cuda()
    # print("  Loading took: {:}".format(format_time(time.time() - start_time)))
    model = utils.initial_setting(model, seed_val=42)

    return model

def finetuning():
    path = path()
    device = utils.GPU_setting()

    train_dataloader, _ , validation_dataloader = preprocess()
    optimizer, epochs, total_steps, scheduler = utils.hyperparmeter_setting(model, train_dataloader, lr=2e-5, eps=1e-8, epochs=3)
    model = pretrain()

    model = utils.run_train(model, epochs, train_dataloader, validation_dataloader, optimizer, scheduler, device, path)
    return model

def save():
    path = path() 
    model = finetuning()

    torch.save(model.state_dict(), path+"model_new.pth")
    model.load_state_dict(torch.load(path+"model_new.pth"))
    model.eval()

    output_dir = path + 'model_save/'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    output_dir = path + 'model_save/'
    # print(output_dir)

    # Load a trained model and vocabulary that you have fine-tuned
    start_time = time.time()
    model_new = BertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir)

    # Copy the model to the GPU.
    device = utils.GPU_setting()
    model_new.to(device)
    print("  Loading took: {:}".format(utils.format_time(time.time() - start_time)))

    model = model_new

    return model, tokenizer