import torch
from model.transformer import Transformer
from utils.data_utils import Vocab, tokenize_en

def translate(model, sentence, vocab_en, vocab_de, device, max_length=50):
    model.eval()
    
    # 1. Tokenize English input
    tokens = [2] + [vocab_en[token] for token in tokenize_en(sentence)] + [3] # 2: BOS, 3: EOS
    src = torch.tensor(tokens).unsqueeze(0).to(device)
    
    # 2. Encoding
    src_mask = (src != 1).unsqueeze(1).unsqueeze(2) # 1: PAD_IDX
    with torch.no_grad():
        src_embedded = model.dropout(model.positional_encoding(model.encoder_embedding(src)))
        enc_output = src_embedded
        for enc_layer in model.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

    # 3. Decoding (Word by Word)
    outputs = [2] # Start with <bos>
    for _ in range(max_length):
        tgt_tensor = torch.tensor(outputs).unsqueeze(0).to(device)
        _, tgt_mask = model.generate_mask(src, tgt_tensor)
        
        with torch.no_grad():
            dec_output = model.dropout(model.positional_encoding(model.decoder_embedding(tgt_tensor)))
            for dec_layer in model.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            
            prediction = model.fc(dec_output)
            next_token = prediction[0, -1, :].argmax().item()
            
        outputs.append(next_token)
        if next_token == 3: # Stop at <eos>
            break
            
    return " ".join([vocab_de.itos[idx] for idx in outputs])

if __name__ == "__main__":
    # This block allows you to run it from the terminal
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Inference script ready. Usage: Provide a sentence to the translate function.")
