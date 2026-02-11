import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset

# Importing from your modular structure
from model.transformer import Transformer
from utils.data_utils import Vocab, collate_fn, tokenize_en, tokenize_de

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Dataset
    print("Loading Multi30k dataset...")
    dataset = load_dataset("bentrevett/multi30k")

    # 3. Build Vocabularies
    print("Building Vocabularies...")
    vocab_en = Vocab((tokenize_en(x['en']) for x in dataset['train']), ['<unk>', '<pad>', '<bos>', '<eos>'])
    vocab_de = Vocab((tokenize_de(x['de']) for x in dataset['train']), ['<unk>', '<pad>', '<bos>', '<eos>'])

    PAD_IDX, BOS_IDX, EOS_IDX = 1, 2, 3
    SRC_VOCAB_SIZE = len(vocab_en)
    TGT_VOCAB_SIZE = len(vocab_de)

    # 4. Create DataLoader
    # We use a lambda to pass vocab/idx arguments to our collate function
    train_loader = DataLoader(
        dataset['train'], 
        batch_size=32, 
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, vocab_en, vocab_de, PAD_IDX, BOS_IDX, EOS_IDX)
    )

    # 5. Initialize Model
    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE, 
        tgt_vocab_size=TGT_VOCAB_SIZE, 
        embed_dim=512, 
        heads=8, 
        num_layers=3, 
        dff=2048, 
        max_seq_length=100, 
        dropout=0.1
    ).to(device)

    # 6. Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # 7. Training Loop
    model.train()
    epochs = 5
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            
            # Teacher Forcing: Shift target
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            
            optimizer.zero_grad()
            logits = model(src, tgt_input)
            
            # Reshape for loss: [batch * seq_len, vocab_size]
            loss = criterion(logits.reshape(-1, TGT_VOCAB_SIZE), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

    # 8. Save the model
    torch.save(model.state_call(), "transformer_model.pt")
    print("Training complete. Model saved!")

if __name__ == "__main__":
    main()
