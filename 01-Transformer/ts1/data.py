import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

# 示例数据集
class SimpleDateset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = 128

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels':torch.tensor(label, dtype=torch.long)
        }
# 创建数据集
texts = ["I love this movie!", "This was a terrible film."]
labels = [1, 0]
dataset = SimpleDateset(texts, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
for batch in dataloader:
    print(batch)

