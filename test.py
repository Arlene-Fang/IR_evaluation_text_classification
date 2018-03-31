import re
import numpy as np

class evaluation(object):
    def __init__(self, testfile, outfile):
        self.test_class = []
        self.out_class = []
        self.get_test_info(testfile)
        self.get_out_info(outfile)
        self.pre = []
        self.recall = []
        self.F1 = []
    def get_test_info(self, testfile):
        with open(testfile) as test:
            raw_test = test.read().splitlines()
            for i in raw_test:
                i = re.split(' ', i)
                self.test_class.append(i[0])

    def get_out_info(self, outfile):
        with open(outfile) as out:
            raw_out = out.read().splitlines()
            for i in raw_out:
                i = re.split(' ', i)
                self.out_class.append(i[0])

    def accuracy(self):
        length = len(self.test_class)
        acc = 0
        for i, k in zip(self.test_class, self.out_class):
            if i == k:
                acc = acc + 1
        Accuracy = acc / length
        #print(Accuracy)
        return  Accuracy

    def p(self):
        out_set = sorted(set(self.out_class), key=int)
        out_set_count = []
        for item in out_set:
            out_set_count.append(self.out_class.count(item))
        acc = []
        for class_no in out_set:
            t = 0
            for (i1,data1) in enumerate(self.out_class):
                for (i2, data2) in enumerate(self.test_class):
                    if class_no == data1 and data1 == data2 and i1 ==i2:
                        t = t + 1
            acc.append(t)
        p = []
        for i, k in zip(acc, out_set_count):
            t = i/k
            q = 'P:' + str("%.4f" % t)
            p.append(q)
            self.pre.append("%.4f" % t)
        #print(p)
        #self.pre = p
        return p

    def r(self):
        test_set = sorted(set(self.test_class), key=int)
        test_set_count = []
        for item in test_set:
            test_set_count.append(self.test_class.count(item))
        acc = []
        for class_no in test_set:
            t = 0
            for (i1, data1) in enumerate(self.test_class):
                for (i2, data2) in enumerate(self.out_class):
                    if class_no == data1 and data1 == data2 and i1 == i2:
                        t = t + 1
            acc.append(t)
        r = []
        for i, k in zip(acc, test_set_count):
            t = i / k
            q = "R:" + str("%.4f" % t)
            r.append(q)
            self.recall.append("%.4f" % t)
        #print(r)
        #self.recall = r
        return r

    def f1(self):
        f1 = []
        for i, k in zip(self.pre, self.recall):
            f = 2 * float(i) * float(k) / (float(i) + float(k))
            q = 'F:' + str("%.4f" % f)
            self.F1.append("%.4f" % f)
            f1.append(q)
        #print(f1)
        #self.F1 = f1
        return f1

    def macro_f1(self):
        f1 = list(map(float, self.F1))
        macro = sum(f1)/len(f1)
        return "%.4f" % macro

    def set(self):
        out_set = sorted(set(self.out_class), key=int)
        list = []
        for i in out_set:
            t = str(i) + ':'
            list.append(t)
        return list

origin = evaluation('feats.test.improve2', 'pred.out.improve2')
set_origin = origin.set()
acc_origin = origin.accuracy()
p_origin = origin.p()
r_origin = origin.r()
f1_origin = origin.f1()
macro_f1_origin = origin.macro_f1()
matrix = np.array([set_origin, p_origin, r_origin, f1_origin]).transpose()
list = list(matrix)
#print(matrix)

file =  open('Eval2.txt', 'w')
file.write('Accuracy = '+str(acc_origin)+'\n')
file.write('Macro-F1 = '+str(macro_f1_origin)+'\n')
file.write('Results per class:'+'\n')
for i in list:
    out = str(i).replace('[', '').replace(']', '').replace('\'', '')
    file.write(out+'\n')
file.close()





