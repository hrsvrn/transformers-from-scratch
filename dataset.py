import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,src_lang,tgt_lang,seq_len)->None:
        super().__init__()
        self.seq_len=seq_len
        self.ds=ds
        self.tokenizer_src=tokenizer_src
        self.tokenizer_tgt=tokenizer_tgt
        self.src_lang=src_lang
        self.tgt_lang=tgt_lang

        # Store token ids as integers for source and target tokenizers
        self.sos_id_src = tokenizer_src.token_to_id('[SOS]')
        self.eos_id_src = tokenizer_src.token_to_id('[EOS]')
        self.pad_id_src = tokenizer_src.token_to_id('[PAD]')

        self.sos_id_tgt = tokenizer_tgt.token_to_id('[SOS]')
        self.eos_id_tgt = tokenizer_tgt.token_to_id('[EOS]')
        self.pad_id_tgt = tokenizer_tgt.token_to_id('[PAD]')
    
    def __len__(self):
        return len(self.ds)
    
    
    def __getitem__(self,idx):
        src_target_pair=self.ds[idx]
        src_text=src_target_pair['translation'][self.src_lang]
        tgt_text=src_target_pair['translation'][self.tgt_lang]

        #Transform Text into Token 
        enc_input_tokens=self.tokenizer_src.encode(src_text).ids
        dec_input_tokens=self.tokenizer_tgt.encode(tgt_text).ids

        #Adding sos,eos and pad to each sentence
        enc_num_padding_tokens= self.seq_len-len(enc_input_tokens)-2
        dec_num_padding_tokens=self.seq_len-len(dec_input_tokens)-1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence  is too long')
        

        # Build encoder input: [SOS] + src_ids + [EOS] + PADs
        encoder_input = torch.cat(
            [
                torch.tensor([self.sos_id_src], dtype=torch.int64),
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                torch.tensor([self.eos_id_src], dtype=torch.int64),
                torch.full((enc_num_padding_tokens,), self.pad_id_src, dtype=torch.int64),
            ],
            dim=0,
        )

        # Build decoder input: [SOS] + tgt_ids + PADs
        decoder_input = torch.cat(
            [
                torch.tensor([self.sos_id_tgt], dtype=torch.int64),
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.full((dec_num_padding_tokens,), self.pad_id_tgt, dtype=torch.int64),
            ],
            dim=0,
        )

        # Labels: tgt_ids + [EOS] + PADs
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.eos_id_tgt], dtype=torch.int64),
                torch.full((dec_num_padding_tokens,), self.pad_id_tgt, dtype=torch.int64),
            ],
            dim=0,
        )

        assert encoder_input.size(0)==self.seq_len
        assert decoder_input.size(0)==self.seq_len
        assert  label.size(0)==self.seq_len


        # Create boolean masks (True = allowed positions)
        encoder_pad_mask = (encoder_input != self.pad_id_src).unsqueeze(0).unsqueeze(0)
        decoder_pad_mask = (decoder_input != self.pad_id_tgt).unsqueeze(0)
        decoder_causal = causal_mask(decoder_input.size(0))
        decoder_mask = decoder_pad_mask & decoder_causal

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_pad_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def causal_mask(size):
    # Returns a boolean mask of shape (1, size, size) where True indicates allowed attention
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
