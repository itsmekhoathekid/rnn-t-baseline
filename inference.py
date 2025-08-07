import os
import csv
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from models.model import Transducer
from utils.dataset import Speech2Text, speech_collate_fn
from jiwer import wer, cer
from tqdm import tqdm

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ids_to_text(ids, itos, eos_id=None):
    tokens = []
    for idx in ids:
        if eos_id is not None and idx == eos_id:
            break
        token = itos.get(idx, '')
        if token in ['<pad>','<s>','</s>','<unk>','<blank>']:
            continue
        tokens.append(token)
    return ' '.join(tokens)

def main():
    parser = argparse.ArgumentParser(description="Inference script for RNN-T speech-to-text model")
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    args = parser.parse_args()

    full_cfg = load_config(args.config)
    model_cfg = full_cfg.get('model', full_cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===Load Checkpoint===
    checkpoint = torch.load(full_cfg["training"]["save_path"] + "/conv-rnnt_epoch_2", map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    #===Load Model===
    model = Transducer(model_cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    #===Load Data===
    dataset = Speech2Text(full_cfg["training"]["test_path"], full_cfg["training"]["vocab_path"])
    itos    = dataset.vocab.itos
    eos_id  = dataset.vocab.get_eos_token()

    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        collate_fn=speech_collate_fn)

    pred_texts = []
    true_texts = []

    with open(full_cfg["training"]["result"], 'w', encoding='utf-8') as fout:
        for batch in loader:
            fbanks     = batch['fbank'].to(device)
            fbank_lens = batch['fbank_len'].to(device)

            with torch.no_grad():
                batch_preds = model.recognize(fbanks, fbank_lens)

            for i in range(len(batch_preds)):
                pred_ids = batch_preds[i]
                true_ids = batch['text'][i].tolist()

                pred_text = ids_to_text(pred_ids, itos, eos_id=eos_id)
                true_text = ids_to_text(true_ids, itos, eos_id=eos_id)

                pred_texts.append(pred_text)
                true_texts.append(true_text)
                print(f"Predict text: {pred_text}")
                print(f"Ground truth: {true_text}")
                fout.write(f"Predict text: {pred_text}\n")
                fout.write(f"Ground truth: {true_text}\n")
                fout.write("---------------\n")

    print(f"Inference complete. Results saved to {full_cfg["training"]["result"]}")

    #===TÍNH WER VÀ CER===
    overall_wer = wer(true_texts, pred_texts)
    overall_cer = cer(true_texts, pred_texts)
    print(f"Word Error Rate (WER): {overall_wer:.4f}")
    print(f"Character Error Rate (CER): {overall_cer:.4f}")

if __name__ == '__main__':
    main()
    
# python /home/anhkhoa/rnn-t/inference.py --config /home/anhkhoa/rnn-t/configs/rnn_t.yaml --checkpoint /home/anhkhoa/rnn-t/save_path/rnn-t_epoch_67 --test_json /home/anhkhoa/transformer_transducer_speeQ/data/dev.json --vocab_json /home/anhkhoa/transformer_transducer_speeQ/data/vocab.json --batch_size 1 --output /home/anhkhoa/rnn-t/result_train.txt