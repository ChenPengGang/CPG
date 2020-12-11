import torch
import torch.optim as optim
import numpy as np
from ReadFile import ReadFile
from CPGnetwork import CPG_Network
from collections import Counter
from SkipGramNeg import SkipGramNeg
from MyLoss import NegativeSamplingLoss
import networkx as nx
#from gensim.models import Word2Vec
from copy import copy

#参数设置
EMBEDDING_DIM = 64 #词向量维度
PRINT_EVERY = 1000 #可视化频率
EPOCHS = 5 #训练的轮数
BATCH_SIZE = 300 #每一批训练数据大小
N_SAMPLES = 5 #负样本大小
WINDOW_SIZE = 5 #周边词窗口大小
FREQ = 0 #词汇出现频率
LR=0.0025

def progress1(percent, width=50):
    '''进度打印功能'''
    if percent >= 100:
        percent = 100

    show_str = ('[%%-%ds]' % width) % (int(width * percent / 100) * "#")  # 字符串拼接的嵌套使用
    print('\r%s %d%%' % (show_str, percent), end='')

def get_target(words, idx, window_size):
    ''' Get a list of words in a window around an index. '''

    R = np.random.randint(1, window_size + 1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = words[start:idx] + words[idx + 1:stop + 1]

    return list(target_words)


#批次化数据
def get_batch(words, BATCH_SIZE, WINDOW_SIZE):
    n_batches = len(words)//BATCH_SIZE
    words = words[:n_batches*BATCH_SIZE]
    for idx in range(0, len(words), BATCH_SIZE):
        batch_x, batch_y = [],[]
        batch = words[idx:idx+BATCH_SIZE]
        for i in range(len(batch)):
                x = batch[i]
                y = get_target(batch, i, WINDOW_SIZE)
                batch_x.extend([x]*len(y))
                batch_y.extend(y)
        yield batch_x, batch_y

def writeFile(model,network, ouput_filename_network,pix):
    f = open(ouput_filename_network, 'a+')
    f.seek(0)  # 将游标指到起始位置
    vectors = model.state_dict()["in_embed.weight"]
    for k, v in network.int2vocab.items():
        if v.endswith('_anchor'):
            v.replace('_anchor',pix)
        if v.endswith(pix):
            f.write(v)  # 写入新数据
            f.write(" ")
            value = torch.Tensor.cpu(vectors[k])
            value = value.data.numpy()
            for val in value:
                f.write(str(val))
                f.write("|")
            f.write("\n")
    print(len(vectors))
    f.flush()  # 将缓存区的数据写入硬盘
    f.close()

def readData(file_name, pix, anchor, graph):
    # 读取自己的twitter文件
    with open(file_name, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            array_edge = line.split(" ", 1)
            array_edge[1] = array_edge[1].replace("\n", "")
            if array_edge[0] in anchor:
                array_edge[0] += '_anchor'
            else:
                array_edge[0] += pix
            if array_edge[1] in anchor:
                array_edge[1] += '_anchor'
            else:
                array_edge[1] += pix
            graph.add_node(array_edge[0])
            graph.add_node(array_edge[1])
            graph.add_edge(array_edge[0], array_edge[1], weight=1)

    del anchor
    f.close()


def getAnchors(network):
    answer_list = []
    file_name = "../AcrossNetworkEmbeddingData/twitter_foursquare_groundtruth/groundtruth.9.foldtrain.train.number"
    # 读取自己的twitter文件
    with open(file_name, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            array_edge = line
            array_edge = array_edge.replace("\n", "")
            answer_list.append(array_edge)
            array_edge = array_edge + '_anchor'
            network.add_node(array_edge)
    print(len(answer_list))
    return answer_list


def read_graph():
    '''
    Reads the input network in networkx.
    '''
    networkx_file = "../AcrossNetworkEmbeddingData/foursquare/following.number"
    networky_file = "../AcrossNetworkEmbeddingData/twitter/following.number"
    graph = nx.DiGraph()

    anchor_list = getAnchors(graph)
    readData(networkx_file, "_foursquare", anchor_list, graph)
    readData(networky_file, "_twitter", anchor_list, graph)
    return graph,anchor_list


def Train():
    ouput_filename_networkx = "../AcrossNetworkEmbeddingData/foursquare/embeddings/emb-15.number"
    ouput_filename_networky = "../AcrossNetworkEmbeddingData/twitter/embeddings/emb-15.number"
    nx_G,anchor_list = read_graph()
    network = CPG_Network(nx_G, True, 1, 1)
    network.preprocess_transition_probs()
    walks = network.simulate_walks(1, 400)
    #learn_embeddings(walks)



    device = 'cuda'
    # 模型、损失函数及优化器初始化
    model = SkipGramNeg(len(network.vocab2int), EMBEDDING_DIM).to(device)
    criterion = NegativeSamplingLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    steps = 0
    all_walks=[]
    for words in walks:
        all_walks+=words
    int_all_words = [network.vocab2int[w] for w in all_walks]
    # 计算单词频次
    int_word_counts = Counter(int_all_words)
    total_count = len(int_all_words)
    word_freqs = {w: c / total_count for w, c in int_word_counts.items()}

    # 单词分布
    word_freqs = np.array(list(word_freqs.values()))
    unigram_dist = word_freqs / word_freqs.sum()
    noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))



    i = 0
    print(len(all_walks))

    for e in range(EPOCHS):
        # 获取输入词以及目标词
        for input_words, target_words in get_batch(int_all_words, BATCH_SIZE, WINDOW_SIZE):
            steps += 1
            inputs, targets = torch.LongTensor(input_words).to(device), torch.LongTensor(target_words).to(device)
            # 输入、输出以及负样本向量
            input_vectors = model.forward_input(inputs)
            output_vectors = model.forward_output(targets)
            size, _ = input_vectors.shape
            noise_vectors = model.forward_noise(size, N_SAMPLES,device,noise_dist)
            # 计算损失
            loss = criterion(input_vectors, output_vectors, noise_vectors)
            # 打印损失
            if steps % PRINT_EVERY == 0:
                print("loss：", loss)
            # 梯度回传
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # LR = 0.003 * (1 - float(i) / len(walks))
        # if LR < 0.0001 * 0.003:
        #     LR = 0.0001 * 0.003
        # optimizer = optim.Adam(model.parameters(), lr=LR)

        print(e)
    print('/')

    writeFile(model,network,ouput_filename_networkx,'_foursquare')
    writeFile(model,network,ouput_filename_networky,'_twitter')

if __name__ == '__main__':
    Train()