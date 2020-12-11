class ReadFile:
    def getAnchors(self):
        answer_map = dict()
        # 读取自己的twitter文件
        f = open("../AcrossNetworkEmbeddingData/twitter_foursquare_groundtruth/groundtruth.9.foldtrain.train.number")
        line = f.readline()
        i = 0
        while line:
            array_edge = line
            array_edge = array_edge.replace("\n", "")
            array_edge1 = array_edge
            array_edge2 = array_edge
            answer_map[array_edge1] = array_edge2
            line = f.readline()
            i += 1
        print(len(answer_map))
        f.close()
        return answer_map
