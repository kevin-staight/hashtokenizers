from tokenizers import models, trainers, pre_tokenizers, normalizers, decoders
from project.hashtokenizers import Tokenizer
import hashlib
import tokenizers

text = "Neurons! fjdsklf this [PAD] you zz! you this is great you [PAD] become what what This is so great, but you are greater! what is your namev?"
vocab_size = 1024

token = Tokenizer(models.BPE())
token.set_hash_mod(1002)
token.normalizer = normalizers.Lowercase()
token.pre_tokenizer = pre_tokenizers.ByteLevel()
token.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=vocab_size, show_progress=False, special_tokens=["[PAD]", "[UNK]", "[NO]"])
token.train_from_iterator([text], trainer)

# Original BPE IDs
original_ids = token.encode("what").ids
print(original_ids)
# Hashed IDs
hashed_ids = token.encode_with_hashed_ids("what").ids
print(hashed_ids)

original_token = token.decode(original_ids)
print(original_token)
hashed_token = token.decode_with_hashed_ids(hashed_ids)
print(hashed_token)