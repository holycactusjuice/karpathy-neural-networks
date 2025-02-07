import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300  # check loss every 300 iters
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32 # embedding dimension
# -----------------

torch.manual_seed(1337)

with open("minishakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# get the chars
chars = sorted(list(set(text)))
vocab_size = len(chars)

# mappings from char <--> int
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s): return [stoi[i] for i in s]
def decode(l): return "".join([itos[i] for i in l])


# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading


def get_batch(split):
    # return inputs and targets
    data = train_data if split == "train" else val_data
    # we need <block_size> consecutive tokens for each sequence
    # so we can't start after len(data) - block_size
    ix = torch.randint(len(data) - block_size, (block_size,))
    # get all the block_size chars sequencse
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])  # get all the chars immediately after
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    # this function estimates the loss by taking many batches
    # from both train and val splits
    # and calculating the average loss over them
    out = {}
    # set model to eval mode while estimation
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # set model back to train mode after done estimation
    model.train()
    return out


# simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx)  # (B, T, C), C = n_embd
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T)
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            # get all the last elements
            logits = logits[:, -1, :]
            # convert to probs
            probs = F.softmax(logits, dim=1)
            # sample
            idx_next = torch.multinomial(probs, num_samples=1)
            # append new char to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx  # (B, T)


model = BigramLanguageModel()
m = model.to(device)  # assigns all the parameters in model to device

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# optimization loop
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}")

    # sample batch of data
    xb, yb = get_batch("train")

    # evaluate loss and step
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# generate returns (B, T) so when B = 1 we need to retrieve the first element
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
