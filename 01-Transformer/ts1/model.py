# 定义模型
# 我们使用预训练的BERT模型进行分类任务
from transformers import BertModel,BertConfig
import torch.nn as nn

class SimpleTransformerModel(nn.Module):
    def __init__(self, num_labels):
        super(SimpleTransformerModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1] # 取池化后的输出
        logits = self.classifier(pooled_output)
        return logits

# 初始化模型
model = SimpleTransformerModel(num_labels=2)
