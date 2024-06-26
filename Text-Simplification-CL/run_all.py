import argparse
import pickle
import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from data import WikiDataset
from tokenizer import Tokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from transformers import EncoderDecoderModel, BertConfig, EncoderDecoderConfig, GPT2Tokenizer
import time
import tqdm
import logging
import gc
import shutil
import sari
from scheduler import get_batches

TRAIN_BATCH_SIZE = 4
N_EPOCH = 2
max_token_len = 80
LOG_EVERY = 10

import logging
from datetime import datetime

log_file_name = f"log_file_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(filename=log_file_name, level=logging.INFO,
                    format="%(asctime)s:%(levelname)s: %(message)s")

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-cased', 'gpt2')
model.decoder.config.use_cache = False
tokenizer = Tokenizer(max_token_len)
model.config.decoder_start_token_id = tokenizer.gpt2_tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.gpt2_tokenizer.eos_token_id
model.config.max_length = max_token_len
model.config.no_repeat_ngram_size = 3

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} as device")
model.to(device)


def collate_fn(batch):
    data_list, label_list, ref_list = [], [], []
    for _data, _label, _ref in batch:
        data_list.append(_data)
        label_list.append(_label)
        ref_list.append(_ref)
    return data_list, label_list, ref_list


def compute_bleu_score(logits, labels):
    refs = Tokenizer.get_sent_tokens(labels)
    weights = (1.0 / 2.0, 1.0 / 2.0,)
    score = corpus_bleu(refs, logits.tolist(), smoothing_function=SmoothingFunction(epsilon=1e-10).method1,
                        weights=weights)
    return score


def compute_sari(norm, pred_tensor, ref):
    pred = tokenizer.decode_sent_tokens(pred_tensor)
    score = 0
    for step, item in enumerate(ref):
        score += sari.SARIsent(norm[step], pred[step], item)
    return score / TRAIN_BATCH_SIZE


def evaluate(data_loader, e_loss, should_print=True):
    was_training = model.training
    model.eval()
    eval_loss = e_loss
    bleu_score = 0
    sari_score = 0
    softmax = nn.LogSoftmax(dim=-1)
    print_count = 0

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            src_tensors, src_attn_tensors, tgt_tensors, tgt_attn_tensors, labels = tokenizer.encode_batch(batch)
            loss, logits = model(input_ids=src_tensors.to(device),
                                 decoder_input_ids=tgt_tensors.to(device),
                                 attention_mask=src_attn_tensors.to(device),
                                 decoder_attention_mask=tgt_attn_tensors.to(device),
                                 labels=labels.to(device))[:2]
            outputs = softmax(logits)
            score = compute_bleu_score(torch.argmax(outputs, dim=-1), batch[1])
            s_score = compute_sari(batch[0], torch.argmax(outputs, dim=-1), batch[2])

            if step == 0:
                eval_loss = loss.item()
                bleu_score = score
                sari_score = s_score
            else:
                eval_loss = (1 / 2.0) * (eval_loss + loss.item())
                bleu_score = (1 / 2.0) * (bleu_score + score)
                sari_score = (1 / 2.0) * (sari_score + s_score)

            if should_print and print_count < 2:
                # Let's print the source, target and predicted sentences for the batch but
                # limit the number of sentences to 5
                print_batch = batch if len(batch) < 5 else batch[:5]
                print(f"Source: {print_batch[0]}")
                print(f"Target: {print_batch[1]}")
                predicted = tokenizer.decode_sent_tokens(torch.argmax(outputs, dim=-1).tolist())
                predicted = predicted if len(predicted) < 5 else predicted[:5]
                print(f"Predicted: {predicted}");
                print(
                    f"eval loss: {eval_loss:.5f} | blue score: {bleu_score} | sari score: {sari_score} | sum_op: {torch.sum(outputs)}")
                print("-" * 100)
                print("\n")
            print_count += 1

    if was_training:
        model.train()

    return eval_loss, bleu_score, sari_score


def load_checkpt(checkpt_path, optimizer=None):
    print(f"Loading checkpoint path: {checkpt_path}")
    checkpoint = torch.load(checkpt_path)
    if device == "cpu":
        model.load_state_dict(checkpoint["model_state_dict"], map_location=torch.device("cpu"))
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"], map_location=torch.device("cpu"))
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    eval_loss = checkpoint["eval_loss"]
    epoch = checkpoint["epoch"]

    return optimizer, eval_loss, epoch


def save_model_checkpt(state, is_best, check_pt_path, best_model_path):
    f_path = check_pt_path
    torch.save(state, f_path)

    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)


def task():
    ''' This is the documentation of the main file. This is the reference for executing this file.'''
    pass


def p_train(base_path="/content/drive/MyDrive/Sentence-Simplification-using-BERT-GPT2",
            src_train="/dataset/src_train.txt", tgt_train="/dataset/tgt_train.txt",
            src_valid="/dataset/src_valid.txt", tgt_valid="/dataset/tgt_valid.txt",
            ref_valid="/dataset/ref_valid.pkl", checkpoint_path="/checkpoint/model_ckpt.pt",
            best_model="/best_model/model.pt", seed=540):
    train_kwargs = {
        "base_path": base_path,
        "src_train": src_train,
        "tgt_train": tgt_train,
        "src_valid": src_valid,
        "tgt_valid": tgt_valid,
        "ref_valid": ref_valid,
        "best_model": best_model,
        "checkpoint_path": checkpoint_path,
        "seed": seed
    }

    # Call your train function with the kwargs
    train(**train_kwargs)


"""
$ run.py train --base_path "./" --src_train "dataset/src_train.txt" --src_valid "dataset/src_valid.txt" /
        --tgt_train "dataset/tgt_train.txt" --tgt_valid "dataset/tgt_valid.txt" /
        --ref_valid "dataset/ref_valid.pkl" --checkpoint_path "checkpoint/model_ckpt.pt" /
        --best_model "best_model/model.pt" --seed 540
"""


def ssh_train(base_path="./", src_train="dataset/INT.txt", tgt_train="dataset/ELE.txt",
              src_valid="dataset/src_valid.txt", tgt_valid="dataset/tgt_valid.txt",
              ref_valid="dataset/ref_valid.pkl", checkpoint_path="cp_ele_int/model_ckpt.pt",
              best_model="cp_ele_int/model.pt", seed=540):
    train_kwargs = {
        "base_path": base_path,
        "src_train": src_train,
        "tgt_train": tgt_train,
        "src_valid": src_valid,
        "tgt_valid": tgt_valid,
        "ref_valid": ref_valid,
        "best_model": best_model,
        "checkpoint_path": checkpoint_path,
        "seed": seed
    }

    print(f"Starting training with following parameters: {train_kwargs}")
    logging.info(f"Starting training with following parameters: {train_kwargs}")

    # Call your train function with the kwargs
    train(**train_kwargs)


def train(**kwargs):
    print("Loading datasets...")
    train_dataset = WikiDataset(kwargs['base_path'] + kwargs['src_train'], kwargs['base_path'] + kwargs['tgt_train'])
    valid_dataset = WikiDataset(kwargs['base_path'] + kwargs['src_valid'], kwargs['base_path'] + kwargs['tgt_valid'],
                                kwargs['base_path'] + kwargs['ref_valid'], ref=True)
    print("Dataset loaded successfully")

    train_dl = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    valid_dl = DataLoader(valid_dataset, batch_size=TRAIN_BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=3e-5)

    start_epoch, eval_loss = 0, 100
    if os.path.exists(kwargs['base_path'] + kwargs["checkpoint_path"]):
        try:
            optimizer, eval_loss, start_epoch = load_checkpt(kwargs['base_path'] + kwargs["checkpoint_path"], optimizer)
            print(f"Loading model from checkpoint with start epoch: {start_epoch} and loss: {eval_loss}")
            logging.info(f"Model loaded from saved checkpoint with start epoch: {start_epoch} and loss: {eval_loss}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    train_model(start_epoch, eval_loss, (train_dl, valid_dl), optimizer,
                kwargs['base_path'] + kwargs["checkpoint_path"], kwargs['base_path'] + kwargs["best_model"])


def test(**kwargs):
    print("Testing Model module executing...")
    logging.info(f"Test module invoked.")
    _, _, _ = load_checkpt(kwargs['base_path'] + kwargs["best_model"])
    print(f"Model loaded.")
    model.eval()
    test_dataset = WikiDataset(kwargs['base_path'] + kwargs['src_test'], kwargs['base_path'] + kwargs['tgt_test'],
                               kwargs['base_path'] + kwargs['ref_test'], ref=True)
    test_dl = DataLoader(test_dataset, batch_size=TRAIN_BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    test_start_time = time.time()
    test_loss, bleu_score, sari_score = evaluate(test_dl, 0)
    test_loss = test_loss / TRAIN_BATCH_SIZE
    print(
        f'Avg. eval loss: {test_loss:.5f} | blue score: {bleu_score} | sari score: {sari_score} | time elapsed: {time.time() - test_start_time}')
    logging.info(
        f'Avg. eval loss: {test_loss:.5f} | blue score: {bleu_score} | sari score: {sari_score} | time elapsed: {time.time() - test_start_time}')
    print("Test Complete!")


def decode(**kwargs):
    print("Decoding sentences module executing...")
    logging.info(f"Decode module invoked.")
    _, _, _ = load_checkpt(kwargs['base_path'] + kwargs["best_model"])
    print(f"Model loaded.")
    model.eval()
    dataset = WikiDataset(kwargs['base_path'] + kwargs['src_file'])
    predicted_list = []
    sent_tensors = tokenizer.encode_sent(dataset.src)
    print("Decoding Sentences...")
    for sent in sent_tensors:
        with torch.no_grad():
            predicted = model.generate(sent[0].to(device), attention_mask=sent[1].to(device),
                                       decoder_start_token_id=model.config.decoder.decoder_start_token_id)
            predicted_list.append(predicted.squeeze())

    output = tokenizer.decode_sent_tokens(predicted_list)
    with open(kwargs['base_path'] + kwargs["output"], "w") as f:
        for sent in output:
            f.write(sent + "\n")
    print("Output file saved successfully.")


def decode_live(base_path, best_model):
    # This loads  the model from the checkpoint and decodes the sentences from the user input
    print("Decoding sentences module executing...")
    logging.info(f"Decode module invoked.")
    _, _, _ = load_checkpt(base_path + best_model)
    print(f"Model loaded.")
    model.eval()

    print("Decoding Sentences...")
    should_continue = True
    while should_continue:
        print("Enter the sentence to be decoded: Enter 'exit' to exit")
        sent = input()
        if sent == "exit":
            should_continue = False
            break
        sent_tensor = tokenizer.encode_sent([sent])
        predicted = model.generate(sent_tensor[0][0].to(device), attention_mask=sent_tensor[0][1].to(device),
                                   decoder_start_token_id=model.config.decoder.decoder_start_token_id)
        output = tokenizer.decode_sent_tokens([predicted.squeeze()])
        print("Simplified Sentence:", output[0])

    print("Okay, byee.")


def train_model(start_epoch, eval_loss, loaders, optimizer, check_pt_path, best_model_path):
    best_eval_loss = eval_loss
    sari_per_epoch = []
    bleu_per_epoch = []
    loss_per_step = []
    print("Evaluating the model before training...")
    epoch_eval_loss, bleu_score, sari_score = evaluate(loaders[1], eval_loss)
    print(f"Initial eval loss: {epoch_eval_loss:.5f} | blue score: {bleu_score} | sari score: {sari_score}")
    print("Model training started...")
    for epoch in range(start_epoch, N_EPOCH):
        print(f"Epoch {epoch} running...")
        epoch_start_time = time.time()
        epoch_train_loss = 0
        epoch_eval_loss = 0
        model.train()
        for step, batch in enumerate(loaders[0]):
            src_tensors, src_attn_tensors, tgt_tensors, tgt_attn_tensors, labels = tokenizer.encode_batch(batch)
            optimizer.zero_grad()
            model.zero_grad()
            loss = model(input_ids=src_tensors.to(device),
                         decoder_input_ids=tgt_tensors.to(device),
                         attention_mask=src_attn_tensors.to(device),
                         decoder_attention_mask=tgt_attn_tensors.to(device),
                         labels=labels.to(device))[0]
            if step == 0:
                epoch_train_loss = loss.item()
            else:
                epoch_train_loss = (1 / 2.0) * (epoch_train_loss + loss.item())

            loss.backward()
            optimizer.step()

            loss_per_step.append(epoch_train_loss)

            if (step + 1) % LOG_EVERY == 0:
                print(
                    f'Epoch: {epoch} | iter: {step + 1} | avg. train loss: {epoch_train_loss} | time elapsed: {time.time() - epoch_start_time}')
                logging.info(
                    f'Epoch: {epoch} | iter: {step + 1} | avg. train loss: {epoch_train_loss} | time elapsed: {time.time() - epoch_start_time}')

        eval_start_time = time.time()
        epoch_eval_loss, bleu_score, sari_score = evaluate(loaders[1], epoch_eval_loss)
        epoch_eval_loss = epoch_eval_loss / TRAIN_BATCH_SIZE
        print(
            f'Completed Epoch: {epoch} | avg. eval loss: {epoch_eval_loss:.5f} | blue score: {bleu_score} | Sari score: {sari_score} | time elapsed: {time.time() - eval_start_time}')
        logging.info(
            f'Completed Epoch: {epoch} | avg. eval loss: {epoch_eval_loss:.5f} | blue score: {bleu_score}| Sari score: {sari_score} | time elapsed: {time.time() - eval_start_time}')

        check_pt = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'eval_loss': epoch_eval_loss,
            'sari_score': sari_score,
            'bleu_score': bleu_score
        }

        bleu_per_epoch.append(bleu_score)
        sari_per_epoch.append(sari_score)

        print("Saving Checkpoint.......")
        check_pt_time = time.time()
        try:
            if epoch_eval_loss < best_eval_loss:
                print("New best model found")
                logging.info(f"New best model found")
                best_eval_loss = epoch_eval_loss
                save_model_checkpt(check_pt, False, check_pt_path, best_model_path)
            else:
                save_model_checkpt(check_pt, False, check_pt_path, best_model_path)
        except Exception as e:
            print(f"Error while saving checkpoint: {e}")
            logging.error(f"Error while saving checkpoint: {e}")
            continue  # Continue training even if checkpoint is not saved
        print(f"Checkpoint saved successfully with time: {time.time() - check_pt_time}")
        logging.info(f"Checkpoint saved successfully with time: {time.time() - check_pt_time}")

        gc.collect()
        torch.cuda.empty_cache()

    logging.info(f"bleu score per epoch: {bleu_per_epoch}")
    logging.info(f"sari score per epoch: {sari_per_epoch}")
    logging.info(f"loss per step: {loss_per_step}")
    print("Training Complete!")
    print(f"bleu score per epoch: {bleu_per_epoch}")
    print(f"sari score per epoch: {sari_per_epoch}")
    print(f"loss per step: {loss_per_step}")


if __name__ == "__main__":
    # Extract command line arguments
    parser = argparse.ArgumentParser(description="Training script for your model")

    parser.add_argument("--base_path", default="./", help="Base path")
    parser.add_argument("--src_train", default="dataset/INT.txt", help="Source training data path")
    parser.add_argument("--tgt_train", default="dataset/ELE.txt", help="Target training data path")
    parser.add_argument("--src_valid", default="dataset/src_valid.txt", help="Source validation data path")
    parser.add_argument("--tgt_valid", default="dataset/tgt_valid.txt", help="Target validation data path")
    parser.add_argument("--ref_valid", default="dataset/ref_valid.pkl", help="Reference validation data path")
    parser.add_argument("--checkpoint_path", default="checkpoint/simp_sch/model_ckpt.pt", help="Checkpoint path")
    parser.add_argument("--best_model", default="cp_ele_int/model.pt", help="Best model path")
    parser.add_argument("--seed", type=int, default=540, help="Random seed")

    train_type_config = {
        "ele_int": {"src_train": "dataset/ELE_INT/ELE.txt",
                    "tgt_train": "dataset/ELE_INT/INT.txt",
                    # "checkpoint_path": "cp/ele_int/model_ckpt.pt",
                    "best_model": "cp/ele_int/model.pt",
                    "log_file": "log/ele_int.log",
                    },
        "int_adv": {
            "src_train": "dataset/ADV_INT/INT.txt",
            "tgt_train": "dataset/ADV_INT/ADV.txt",
            # "checkpoint_path": "cp/int_adv/model_ckpt.pt",
            "best_model": "cp/int_adv/model.pt",
            "log_file": "log/int_adv.log",
        },
        "ele_adv": {"src_train": "dataset/ADV_ELE/ELE.txt",
                    "tgt_train": "dataset/ADV_ELE/ADV.txt",
                    # "checkpoint_path": "cp/ele_adv/model_ckpt.pt",
                    "best_model": "cp/ele_adv/model.pt",
                    "log_file": "log/ele_adv.log"
                    }
    }

    args = parser.parse_args()
    train_types = ["ele_int", "int_adv", "ele_adv"]
    for train_type in train_types:
        logging.info(f"*****************Starting training for {train_type}*******************************")
        config = train_type_config[train_type]
        for train_args in train_type_config[train_type]:
            args.__dict__[train_args] = train_type_config[train_type][train_args]

        ssh_train(base_path=args.base_path, src_train=args.src_train, tgt_train=args.tgt_train,
                  src_valid=args.src_valid, tgt_valid=args.tgt_valid,
                  ref_valid=args.ref_valid, checkpoint_path=args.checkpoint_path,
                  best_model=args.best_model, seed=args.seed)