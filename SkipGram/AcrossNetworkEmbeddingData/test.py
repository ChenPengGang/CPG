import torch
import random
from torch.autograd import Variable

def test():
    # 读取自己的anchor文件
    f_networkx = open("foursquare/embeddings/emb.number")
    f_networky = open("twitter/embeddings/emb.number")
    f_test=open("twitter_foursquare_groundtruth/groundtruth.9.foldtrain.test.number")
    #f_test = open("twitter_foursquare_groundtruth/groundtruth.9.foldtrain.train.number")
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

    print('-------------------------')
    line = f_test.readline()
    sum1 = 0
    sum2=0
    all = 0
    num1 = 0
    num2=0
    i=0
    while line:
        array_edge = line
        array_edge = array_edge.replace("\n", "")
        y = array_edge + "_twitter"
        x = array_edge + "_foursquare"
        if x in network_x.keys() and y in network_y.keys():
            z = random.sample(network_y.keys(),1)[0]
            if z==y:
                z = random.sample(network_y.keys(),1)[0]
            # while (z in network_x.input_map[x]) or (z in network_x.output_map[x]):
            #     z = random.sample(network_y.nodeList, 1)[0]
            vector3=network_y[z]

            vector1 = network_x[x]
            vector2 = network_y[y]
            #print(vector1.shape(),vector2.shape())
            sum1 = torch.cosine_similarity(vector1, vector2, dim=0)
            sum2=torch.cosine_similarity(vector1,vector3,dim=0)
        #print(sum.double())
        print(sum1)
        if sum1.double()>=0.4:
            num1+=1
        if sum2.double()>=0.4:
            num2+=1
        all+=1
        line = f_test.readline()
        i+=1
    f_test.close()
    print("accurancy pos:",num1/all)
    print('random:-------------------')
    print("accurancy neg:", num2 / all)
def change2tensor(list):
    list = torch.Tensor(list)
    list = list.squeeze()
    list = Variable(list)
    return list

if __name__ == '__main__':
    test()