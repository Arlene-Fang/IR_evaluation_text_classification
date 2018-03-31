import re
import numpy as np
from collections import OrderedDict

class evaluation(object):
    def __init__(self, filename):
        self.query_class = []
        self.data = np.zeros([])
        self.get_results(filename)

    def get_results(self, filename):
        self.data = np.loadtxt(filename)
        for query_no in query_dict.keys():
            self.query_class.append(np.where(self.data[:, 0] == query_no)[0][0])
        self.query_class.append(self.data.shape[0])

    def p10(self):
        precision = np.array([])
        for b, q in zip(self.query_class, query_dict.keys()):
            rel_ret = [int(i) for i in query_dict[q].keys() if int(i) in list(self.data[b:b+10, 2])]
            precision = np.append(precision, len(rel_ret)/10)
        return precision

    def r50(self):
        recall = np.array([])
        for b, q in zip(self.query_class, query_dict.keys()):
            rel_ret = [int(i) for i in query_dict[q].keys() if int(i) in list(self.data[b:b+50, 2])]
            recall = np.append(recall, len(rel_ret)/len(query_dict[q].items()))
        return recall

    def r_precision(self):
        precision = np.array([])
        for b, q in zip(self.query_class, query_dict.keys()):
            r = len(query_dict[q].keys())
            rel_ret = [int(i) for i in query_dict[q].keys() if int(i) in list(self.data[b:b+r, 2])]
            precision = np.append(precision, len(rel_ret) / r)
        return precision

    def map(self):
        ap_list = []
        for b, q in enumerate(query_dict.keys()):
            ret = self.data[self.query_class[b]:self.query_class[b+1], 2].astype(int)
            rel = set(query_dict[q].keys())
            llist = []
            for r in rel:
                i = np.where(ret == int(r))
                if len(i[0]):
                    llist.append(self.query_class[b]+i[0][0])
            llist = sorted(llist)
            ap = 0
            for j, k in enumerate(llist):
                ap += (j+1)/(k-self.query_class[b]+1)
            ap = ap/len(query_dict[q].keys())
            ap_list.append(ap)
        return ap_list

    def ndcg(self, parm=10):

        dcg_arr = np.array([])
        for b, q in enumerate(query_dict.keys()):
            ret = self.data[self.query_class[b]:self.query_class[b+1], :].astype(int)
            rel = set(query_dict[q].keys())
            for r in rel:
                i = np.where(ret[:, 2] == int(r))
                if len(i[0]):
                    ret[i[0][0], 1] = int(query_dict[q][r])
            log = log_func[:np.shape(ret)[0]]
            grad = list(ret[:, 1])
            grads = list(map(lambda x, y : x*y, log, grad))
            discount = 0
            for grad in grads:
                discount += grad
                dcg_arr = np.append(dcg_arr, discount)
        ndcg_parm_arr = np.array([])
        for b, q in enumerate(query_dict.keys()):
            dcg_parm = dcg_arr[self.query_class[b]:self.query_class[b] + parm]
            if idcg_class[b] + parm <= idcg_class[b + 1]:
                idcg_parm = query_idcg_arr[idcg_class[b]:idcg_class[b] + parm]
            else:
                repeat = query_idcg_arr[idcg_class[b + 1] - 1]
                repeat = repeat * np.ones([parm - idcg_class[b + 1] + idcg_class[b], 1])
                idcg_parm = np.append(query_idcg_arr[idcg_class[b]:idcg_class[b + 1]], repeat)
            ndcg = list(map(np.divide, dcg_parm, idcg_parm))
            ndcg_parm = ndcg[parm - 1]
            ndcg_parm_arr = np.append(ndcg_parm_arr, ndcg_parm)
        return ndcg_parm_arr

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
log_func = [1]+[np.log(2)/np.log(xx) for xx in np.arange(500)+2]

with open('qrels.txt') as query:
    raw_query = query.read().splitlines()
    query_dict = OrderedDict()
    query_idcg_arr = np.array([])
    idcg_class = np.array([0])
    for q in raw_query:
        q = re.split(':', q)
        q_no = q[0]
        q_doc = q[1].strip()
        docs = re.findall('\d+,\d+', q_doc)
        subdicts = {}
        for doc in docs:
            item = doc.split(',')
            subdicts[item[0]] = int(item[1])
        query_dict[int(q_no)] = subdicts
        ig = sorted(subdicts.values(), reverse=True)
        log = log_func[:np.shape(ig)[0]]
        query_idg_arr = list(map(np.multiply, log, ig))
        idg = 0
        for i in query_idg_arr:
            idg += i
            query_idcg_arr = np.append(query_idcg_arr, idg)
        idcg_class = np.append(idcg_class, idcg_class[-1] + len(query_idg_arr))

num = 6
file_all = open("All.eval", 'w')
file_all.write("\tP@10\tR@50\tr-Precision\tAP\tnDCG@10\tnDCG@20 \n")
for s in np.arange(num):
    filename = 'S' + str(s+1) + ".results"
    with open(filename) as filename:
        ss = evaluation(filename)
        p10 = ss.p10()
        r50 = ss.r50()
        r_precision = ss.r_precision()
        MAP = ss.map()
        ndcg10 = ss.ndcg(10)
        ndcg20 = ss.ndcg(20)
        matrix = np.array([p10, r50, r_precision, MAP, ndcg10, ndcg20]).transpose()
        mean = matrix.mean(0)
        str_mean = ''
        for i in mean:
            str_mean += (str("%.3f" % i) + '\t')
        file = open('S' + str(s+1) + ".eval", 'w')
        file.write("\tP@10\tR@50\tr-Precision\tAP\tnDCG@10\tnDCG@20 \n")
        for i in query_dict.keys():
            str_matrix = ''
            for t in matrix[i - 1, :]:
                str_matrix += (str("%.3f" % t) + '\t')
            file.write(str(i).strip() + "\t" + str_matrix + "\n")
        file.write("mean\t" + str_mean)
        file_all.write("S"+str(s+1)+"\t" + str_mean + "\n")
        file.close()
        print("System",s+1,"has been done")
file_all.close()
