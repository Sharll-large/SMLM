import os
import re

import jieba
import torch
import torch.nn as nn


class SMLM2(nn.Module):
    def __init__(self, max_vocabulary_size=5000, embedding_dim=64, hidden_dim=128, n_layers=2):
        super().__init__()

        # 初始化token表
        self.vocabulary_size = 0
        self.max_vocabulary_size = max_vocabulary_size
        self.voc_to_id = {}
        self.id_to_voc = {}
        self.special_tokens = {}
        self.init_tokens()

        self.embedding = nn.Embedding(max_vocabulary_size, embedding_dim)

        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, max_vocabulary_size)

        self.activation = nn.Tanh()

    def init_tokens(self):
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<start>": 2,
            "<end>": 3
        }
        for i, j in self.special_tokens.items():
            self.add_single_token(i, j)

    def add_single_token(self, vocabulary, idx):
        if vocabulary not in self.voc_to_id and self.vocabulary_size < self.max_vocabulary_size:
            self.voc_to_id[vocabulary] = idx
            self.id_to_voc[idx] = vocabulary
            self.vocabulary_size += 1

    def tokenize(self, texts: list[str], add_special_tokens=True):
        tokens = []
        if add_special_tokens:
            tokens.append(self.voc_to_id['<start>'])

        for word in texts:
            if word in self.voc_to_id:
                tokens.append(self.voc_to_id[word])
            else:
                tokens.append(self.voc_to_id['<unk>'])

        if add_special_tokens:
            tokens.append(self.voc_to_id['<end>'])

        return tokens

    def detokenize(self, tokens: list[int], skip_special_tokens=True):
        words = []
        for token in tokens:
            if token in self.id_to_voc:
                word = self.id_to_voc[token]
                if (word not in self.special_tokens) or (not skip_special_tokens):  # 跳过特殊标记
                    words.append(word)

        return words

    def forward(self, t1, t2):
        emb1 = self.embedding(t1)
        emb2 = self.embedding(t2)

        combined = torch.cat([emb1, emb2], dim=-1)

        hidden = self.activation(self.fc1(combined))

        output = self.fc2(hidden)

        return output

    def add_auto_tokens(self, texts: str):
        for word in texts:
            self.add_single_token(word, self.vocabulary_size)

    def prepare_training_data(self, texts):
        """准备训练数据：输入前两个词，预测第三个词"""
        all_inputs = []
        all_targets = []

        tokens = self.tokenize(texts)

        # 创建训练样本 (t1, t2) -> t3
        for i in range(len(tokens) - 2):
            t1, t2, t3 = tokens[i], tokens[i + 1], tokens[i + 2]

            # 跳过包含特殊标记的样本（除了开始标记）
            if (t1 not in [0, 1, 3] and  # 不是pad, unk, end
                    t2 not in [0, 1, 3] and
                    t3 not in [0, 1]):  # 目标不能是pad或unk

                all_inputs.append([t1, t2])
                all_targets.append(t3)

        inputs_tensor = torch.tensor(all_inputs)
        targets_tensor = torch.tensor(all_targets)

        return inputs_tensor, targets_tensor

    def train_model(self, texts: str, epochs=10, batch_size=32):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        inputs, targets = self.prepare_training_data(texts)
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i + batch_size]
                batch_targets = targets[i:i + batch_size]

                t1 = batch_inputs[:, 0]
                t2 = batch_inputs[:, 1]

                outputs = self.forward(t1, t2)
                loss = criterion(outputs, batch_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch}/{epochs}, Avg Loss: {avg_loss}")

    def auto_train(self, texts: str, epochs=10, batch_size=32):
        self.add_auto_tokens(texts)
        self.train_model(texts, epochs=epochs, batch_size=batch_size)

    def _predict_next_token(self, token1: int, token2: int, temperature=1.0):
        self.eval()  # 设置为评估模式

        with torch.no_grad():  # 不计算梯度
            # 将输入转换为tensor
            t1_tensor = torch.tensor([token1])
            t2_tensor = torch.tensor([token2])

            # 前向传播获取输出
            outputs = self.forward(t1_tensor, t2_tensor)

            # 应用温度调节
            outputs = outputs / temperature

            # 转换为概率分布
            probabilities = torch.softmax(outputs, dim=-1)

            return probabilities[0]  # 返回概率分布

    def _select_next_token(self, token1: int, token2: int, do_sample=True, temperature=1.0, top_k=None):
        if not do_sample:
            probabilities = self._predict_next_token(token1, token2, temperature=1.0)

            # 直接选择概率最高的token
            next_token = torch.argmax(probabilities).item()
            return next_token

        probabilities = self._predict_next_token(token1, token2, temperature)
        if top_k is not None:
            top_k_probs, top_k_indices = torch.topk(probabilities, top_k)
            probabilities = torch.zeros_like(probabilities)
            probabilities[top_k_indices] = top_k_probs
            probabilities = probabilities / probabilities.sum()  # 重新归一化

        return torch.multinomial(probabilities, 1).item()

    def generate(self, words, max_length=128, do_sample=True, temperature=1.0, top_k=None):
        starting_tokens = self.tokenize(words, add_special_tokens=False)

        t1, t2 = starting_tokens[0], starting_tokens[1]
        generated_tokens = starting_tokens.copy()

        self.eval()

        for _ in range(max_length):
            # 选择下一个token
            next_token = self._select_next_token(t1, t2, do_sample, temperature, top_k)

            # 如果生成了结束标记，停止生成
            if next_token == self.voc_to_id["<end>"]:
                break

            generated_tokens.append(next_token)

            # 更新前两个token
            t1, t2 = t2, next_token

        # 将token转换回文本
        generated_text = self.detokenize(generated_tokens)
        return generated_text


smlm = SMLM2()


def preprocess_text(text):
    """文本预处理：清洗和标准化"""
    # 转换为小写
    text = text.lower()

    # 在标点符号周围添加空格
    text = re.sub(r'([.,!?;:"()])', r' \1 ', text)

    # 合并多个空格
    text = re.sub(r'\s+', ' ', text)

    return text.strip().split()


for i in os.listdir("./stories_cn"):
    with open("./stories_cn/{}".format(i), "r+", encoding="utf-8") as f:
        smlm.auto_train(jieba.lcut(f.read()), epochs=40)

print(smlm.voc_to_id.keys())

while True:
    print("".join(smlm.generate([input("w1: "), input("w2: ")], temperature=float(input("Temp: ")))))
