

from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from collections import Counter

# 1. Load Dataset
dataset = load_dataset("bentrevett/multi30k")

# 2. Tokenizers
import spacy
en_nlp = spacy.load("en_core_web_sm")
de_nlp = spacy.load("de_core_news_sm")

def tokenize_en(text): return [tok.text.lower() for tok in en_nlp.tokenizer(text)]
def tokenize_de(text): return [tok.text.lower() for tok in de_nlp.tokenizer(text)]

# 3. Simple Vocab Class
class Vocab:
    def __init__(self, tokens_generator, specials):
        self.itos = specials[:]
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        
        counter = Counter()
        for tokens in tokens_generator:
            counter.update(tokens)
        
        for token, freq in counter.items():
            if freq >= 1: # You can increase this to prune rare words
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def __len__(self): return len(self.itos)
    def __getitem__(self, token): return self.stoi.get(token, 0) # 0 is <unk>

# 4. Build Vocabs
print("Building Vocabularies...")
vocab_en = Vocab((tokenize_en(x['en']) for x in dataset['train']), ['<unk>', '<pad>', '<bos>', '<eos>'])
vocab_de = Vocab((tokenize_de(x['de']) for x in dataset['train']), ['<unk>', '<pad>', '<bos>', '<eos>'])

PAD_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

# 5. Collate Function
def collate_fn(batch):
    src_list, tgt_list = [], []
    for item in batch:
        en_ids = torch.tensor([BOS_IDX] + [vocab_en[token] for token in tokenize_en(item['en'])] + [EOS_IDX])
        de_ids = torch.tensor([BOS_IDX] + [vocab_de[token] for token in tokenize_de(item['de'])] + [EOS_IDX])
        src_list.append(en_ids)
        tgt_list.append(de_ids)
    
    return pad_sequence(src_list, padding_value=PAD_IDX, batch_first=True), \
           pad_sequence(tgt_list, padding_value=PAD_IDX, batch_first=True)

train_loader = DataLoader(dataset['train'], batch_size=32, collate_fn=collate_fn, shuffle=True)
print(f"Vocab sizes: EN={len(vocab_en)}, DE={len(vocab_de)}")
