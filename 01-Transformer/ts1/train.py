
# 训练模型
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from data import dataloader
from model import SimpleTransformerModel, model

# 初始化模型
# model = SimpleTransformerModel(num_labels=2)

# 损失函数
criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# 训练循环
model.train()
for epoch in range(5):
    # 训练3个epoch
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, loss: {loss.item()}')


