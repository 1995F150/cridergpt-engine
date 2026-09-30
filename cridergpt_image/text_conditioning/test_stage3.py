import torch

from tokenizer import CriderImageTokenizer
from encoder import CriderImageTextEncoder


def test_tokenizer_deterministic():
    captions = ["A red chicken in grass.", "A blue server rack."]
    a = CriderImageTokenizer.train(captions, vocab_size=64)
    b = CriderImageTokenizer.train(captions, vocab_size=64)
    assert a.vocab == b.vocab
    assert a.encode("A red chicken.") == b.encode("A red chicken.")


def test_special_tokens_and_shape():
    tok = CriderImageTokenizer.train(["A red chicken in grass."], vocab_size=64)
    ids, mask = tok.encode("A red chicken.")
    assert len(ids) == 64
    assert len(mask) == 64
    assert ids[0] == tok.bos_id
    assert tok.eos_id in ids

    model = CriderImageTextEncoder(
        vocab_size=len(tok.vocab), max_length=64, embedding_dim=256,
        layers=4, heads=4, feedforward_dim=1024, dropout=0.0,
        pad_id=tok.pad_id,
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor([ids]), torch.tensor([mask]))
    assert tuple(out.shape) == (1, 64, 256)


def test_unknown_token():
    tok = CriderImageTokenizer.train(["known words"], vocab_size=32)
    ids, _ = tok.encode("completely_unseen_token", pad_to_max=False)
    assert tok.unk_id in ids
