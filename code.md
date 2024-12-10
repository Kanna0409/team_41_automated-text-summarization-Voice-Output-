## **Installation**
Before running the code, install the required dependencies:
```bash
pip install torch transformers sentencepiece spacy gtts pygame pydub
```

---

## **Code Walkthrough**

### **1. Importing Libraries**
```python
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
```
- **Purpose**: Import necessary libraries for processing data, training the model, and managing datasets.

---

### **2. Dataset Loading**
```python
path = '/path/to/train.csv'
df = pd.read_csv(path)
df.head()
```
- **Input**: A `.csv` file containing the following columns:
  - `id`
  - `dialogue`
  - `summary`
  - `topic`
- **Output**: A DataFrame preview displaying the first few rows of the dataset.

---

### **3. Text Preprocessing**
```python
import spacy
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text)
    cleaned_tokens = [
        token.text for token in doc if not token.is_punct and not token.is_space
    ]
    return " ".join(cleaned_tokens)
```
- **Functionality**:
  - Removes punctuation and spaces from the text.
  - Returns cleaned text.

---

### **4. Sequence Length Analysis**
```python
print(np.percentile([len(x) for x in df['dialogue']], 97))  # 97% dialogues ≤ 1584 chars
print(np.percentile([len(x) for x in df['summary']], 97))  # 97% summaries ≤ 283 chars
```
- **Purpose**: Identify maximum lengths for dialogues and summaries to ensure efficient processing.

---

### **5. Filtering Valid Data**
```python
def is_valid_length(paragraph, max_seq_length):
    return len(paragraph) < max_seq_length

valid_textsummaries_idx = [
    i for i in range(len(df['summary']))
    if is_valid_length(dialogue_list[i], 1584) and is_valid_length(summary_list[i], 290)
]
```
- Filters dialogues and summaries exceeding length constraints.

---

### **6. Tokenization**
```python
tokenizer = T5Tokenizer.from_pretrained("t5-base")
tokenized_dialogues = tokenizer(dialogue_list, padding='max_length', max_length=450, truncation=True)
tokenized_summaries = tokenizer(summary_list, padding='max_length', max_length=450, truncation=True)
```
- **Purpose**:
  - Tokenizes texts into integer IDs.
  - Pads and truncates sequences to fixed lengths.

---

### **7. DataLoader Setup**
```python
class TextDataset(Dataset):
    def __init__(self, dialogues, summaries):
        self.dialogues = dialogues
        self.summaries = summaries

    def __len__(self):
        return len(self.summaries)

    def __getitem__(self, idx):
        return self.dialogues[idx], self.summaries[idx]

data = TextDataset(dialogue_list, summary_list)
train_loader = DataLoader(data, batch_size=2, shuffle=True)
```
- **Purpose**: Create batches of data for training.

---

### **8. Transformer Model Components**
#### **Multi-Head Attention**
```python
class MultiheadAttention(nn.Module):
    def __init__(self, weights_dim, n_heads):
        super().__init__()
        self.qkv_matrices = nn.Linear(weights_dim, 3 * weights_dim)
        self.linear_layer = nn.Linear(weights_dim, weights_dim)

    def forward(self, x, mask=None):
        q, k, v = self.qkv_matrices(x).chunk(3, dim=-1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        if mask is not None:
            scores += mask
        attention = torch.softmax(scores, dim=-1) @ v
        return self.linear_layer(attention)
```
- Computes scaled dot-product attention for the inputs.

---

#### **Feedforward Layer**
```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.layers(x)
```
- Applies a sequence of linear transformations and activations.

---

### **9. Positional Encoding**
```python
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.encoding = torch.zeros(max_len, embed_dim)
        positions = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        self.encoding[:, 0::2] = torch.sin(positions * div_term)
        self.encoding[:, 1::2] = torch.cos(positions * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :]
```
- Encodes positional information to retain word order.

---

### **10. Transformer Model Assembly**
```python
class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = Encoder(embed_dim, num_heads, num_layers)
        self.decoder = Decoder(embed_dim, num_heads, num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt):
        enc_out = self.encoder(src)
        dec_out = self.decoder(tgt, enc_out)
        return self.fc_out(dec_out)
```

---

### **11. Training**
```python
optimizer = torch.optim.Adam(transformer.parameters())
criterion = nn.CrossEntropyLoss()

for batch in train_loader:
    optimizer.zero_grad()
    input_ids, target_ids = batch
    outputs = transformer(input_ids, target_ids)
    loss = criterion(outputs.view(-1, vocab_size), target_ids.view(-1))
    loss.backward()
    optimizer.step()
```
- Trains the Transformer by minimizing cross-entropy loss.

---

### **12. Text-to-Speech Conversion**
```python
from gtts import gTTS
tts = gTTS(text="Generated summary here.", lang='en')
tts.save("output.mp3")
```
- Converts generated summaries into speech using Google TTS.
