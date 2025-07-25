# Overview
This is a simple library (monkeypatch of huggingface tokenizers) that tries to have consistent universal tokenization of text.
## What does it do?
For any token of text you will get the same* id number, even if you change the vocab size or add new special tokens
## Why would you want that?
Better reuse - if you train LLM model weights, you can change the vocab size, and you can train from the old weights and quickly get to the same performance.
Anyone can train a model with with this tokanizer with any parameters and you can use that model without a specialized tokenizer
## what are the limitations?
Inefficient use of IDs space - if your vocab == your hashmod, about 27% of the id space will be unused
Colisions during hashing - tokens can get mapped to the same id
In practice, this is not really that big of a deal, as the model knows what token you mean from context (like homographs), and the id space is a small price to pay for not retraining from scratch for every model iteration!


# sample code
from hashtokenizers import Tokenizer
from tokenizers import models, trainers, pre_tokenizers, normalizers, decoders
import tokenizers

text = "Neurons! fjdsklf this [PAD] you zz! you this is great you [PAD] become what what This is so great, but you are greater! what is your name?"
vocab_size = 1024
hash_mod_size = vocab_size*2

## Use like huggingface Tokenizer - but set hash mod (number of unique ids, must match input of embedding in LLM)
token = Tokenizer(models.BPE())
token.set_hash_mod(hash_mod_size)

## Now use exactly like huggingface Tokenizer
token.normalizer = normalizers.Lowercase()
token.pre_tokenizer = pre_tokenizers.ByteLevel()
token.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=vocab_size, show_progress=False, special_tokens=["[PAD]", "[UNK]", "[NO]"])
token.train_from_iterator([text], trainer)

## Original BPE IDs
original_ids = token.encode("what").ids
print(original_ids)
## Hashed IDs
hashed_ids = token.encode_with_hashed_ids("what").ids
print(hashed_ids)

original_token = token.decode(original_ids)
print(original_token)
hashed_token = token.decode_with_hashed_ids(hashed_ids)
print(hashed_token)