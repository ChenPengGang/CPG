import torch
import random
from torch.autograd import Variable

def getpAtN(network_x,network_y):
    f_test = open("twitter_foursquare_groundtruth/groundtruth.9.foldtrain.test.number")
    # f_test = open("twitter_foursquare_groundtruth/groundtruth.9.foldtrain.train.number")
    pAtN_x_map=dict()

    print('-------------------------')
    line = f_test.readline()
    all = 0
    i = 0
    while line:
        target = 0
        array_edge = line
        array_edge = array_edge.replace("\n", "")
        y = array_edge + "_twitter"
        x = array_edge + "_foursquare"

        if x in network_x.keys() and y in network_y.keys():
            sam = torch.cosine_similarity(network_x[x], network_y[y], dim=0)
            for value in network_y.values():
                if (torch.cosine_similarity(network_x[x], value, dim=0).double() > sam.double()):
                    target += 1
        pAtN_x_map[array_edge]=target
        all += 1
        line = f_test.readline()
        i += 1
    f_test.close()
    return pAtN_x_map


def getpAtN_Revers(network_y, network_x):
    f_test = open("twitter_foursquare_groundtruth/groundtruth.9.foldtrain.test.number")
    # f_test = open("twitter_foursquare_groundtruth/groundtruth.9.foldtrain.train.number")
    pAtN_y_map = dict()

    print('-------------------------')
    line = f_test.readline()
    all = 0
    i = 0
    while line:
        target = 0
        array_edge = line
        array_edge = array_edge.replace("\n", "")
        y = array_edge + "_twitter"
        x = array_edge + "_foursquare"

        if x in network_x.keys() and y in network_y.keys():
            sam = torch.cosine_similarity(network_x[x], network_y[y],dim=0)
            for value in network_x.values():
                if (torch.cosine_similarity(network_y[y], value,dim=0).double() > sam.double()):
                    target += 1
        pAtN_y_map[array_edge] = target
        all += 1
        line = f_test.readline()
        i += 1
    f_test.close()
    return pAtN_y_map,all

def test():
    a=[0]*30
    b=[0]*30
    # 读取自己的anchor文件
    f_networkx = open("foursquare/embeddings/emb-19.number")
    f_networky = open("twitter/embeddings/emb-19.number")

    network_x=dict()
    network_y=dict()


    line=f_networkx.readline()
    while line:
        listx = []
        line=line.replace("|\n","")
        sp=line.split(" ",1)
        vector_array=sp[1].split("|",127)
        for x in vector_array:
            listx.append(x)
        listx=list(map(float,listx))
        vector=change2tensor(listx)
        network_x[sp[0]]=vector
        line=f_networkx.readline()
    f_networkx.close()

    line = f_networky.readline()
    while line:
        listy = []
        line = line.replace("|\n", "")
        sp = line.split(" ", 1)
        vector_array = sp[1].split("|", 127)
        for y in vector_array:
            listy.append(y)
        listy = list(map(float, listy))
        vector = change2tensor(listy)
        network_y[sp[0]] = vector
        line = f_networky.readline()
    f_networky.close()

    map_x=getpAtN(network_x,network_y)
    map_y,all=getpAtN_Revers(network_y,network_x)

    for i in range(30):
        for value in map_x.values():
            if value==i:
                a[i]+=1
    for i in range(30):
        for value in map_y.values():
            if value==i:
                b[i]+=1

    for i in range(30):
        a[i]/=all
    for i in range(30):
        b[i]/=all

    for i in range(1,30):
        a[i]+=a[i-1]
        b[i]+=b[i-1]
    for i in range(30):
        print(i,':',(a[i]+b[i])/2,end=', ')


def change2tensor(list):
    list = torch.Tensor(list)
    list = list.squeeze()
    list = Variable(list)
    return list

if __name__ == '__main__':
    test()