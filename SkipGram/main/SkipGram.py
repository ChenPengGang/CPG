import torch.nn as nn

class SkipGram(nn.Module):
    def __init__(self, n_vocab, n_embed):
        '''
        定义网络模型
        :param n_vocab: 词向量的个数
        :param n_embed: embedding的维度
        '''
        super().__init__()

        self.embed = nn.Embedding(n_vocab, n_embed)
        self.output = nn.Linear(n_embed, n_vocab)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embed(x)
        scores = self.output(x)
        log_pos = self.log_softmax(scores)

        return log_pos