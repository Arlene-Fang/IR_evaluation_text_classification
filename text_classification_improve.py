import re
import pickle
from collections import Counter
from stemming.porter2 import stem

url_regex = r"\"?http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\"?"
at_regex = r'@[0-9a-zA-Z_]{1,13}'

dicFile = open('classIDs.txt', 'r')
classIDs_Dict = {}
while True:
    line = dicFile.readline()
    if line == '':
        break
    index = line.rfind('\t')
    key = line[:index].lower()
    value = line[index:]
    classIDs_Dict[key] = int(value)
#print(classIDs_Dict)
dicFile.close()

def extract_bow_feature(filename):
    file = open(filename, 'r')
    tweet_content = ''
    for line in file:
        if not line.split():
            continue
        arr = line.lower()
        arr = re.split(r'\t', arr)
        if arr[1].startswith('RT'):
            arr[1] = arr[1][2:-1]
        tweet = re.sub(url_regex, ',', arr[1])  # remove url
        tweet = re.sub(at_regex, ',', tweet)  # remove @
        tweet = re.sub(r'\d', ',', tweet)  # remove numbers
        tweet = re.sub(r'\W+', ',', tweet)  # remove tokenise
        #print(tweet)
        tweet_content = tweet_content+","+tweet
    # print(tweet_content)
    tweet_list_stem = list(set(tweet_content.split(",")))
    # print(tweet_list_stem)
    cache = {}
    tweet_list = []
    for t in tweet_list_stem:
        if t in cache:
            tweet_list.append(cache[t])
        else:
            cache[t] = stem(t)
            tweet_list.append(cache[t])
    # print(tweet_list)
    vocab_set = (lambda x:(x.sort(),x)[1])(tweet_list)
    vocab_set = sorted(vocab_set,key = str.lower)
    # print(vocab_set)
    feats_dic = {}
    i = 1
    for v in vocab_set[3:-1]:
        feats_dic[v] = int(i)
        i =i + 1
    print(feats_dic)
    output = open('feats.dic.improve', 'ab+')
    pickle.dump(feats_dic, output)
    output.close()
    file.close()
    return feats_dic

featsdic = extract_bow_feature('Tweets.14cat.train')

def set_features(filename):
    file = open(filename, 'r')
    classname = str(filename)[13:]
    wfile = open('feats.' + classname + '.improve', 'w')
    for line in file:
        if not line.split():
            continue
        arr = line.lower()
        arr = re.split(r'\t', arr)
        if arr[1].startswith('rt'):
            arr[1] = arr[1][2:-1]
        tweet = re.sub(url_regex, ',', arr[1])  # remove url
        tweet = re.sub(at_regex, ',', tweet)  # remove @
        tweet = re.sub(r'\d', ',', tweet)  # remove numbers
        tweet = re.sub(r'\W+', ',', tweet)  # remove tokenise
        tweet_list = list(tweet.split(","))
        #print(tweet_list)
        while '' in tweet_list:
            tweet_list.remove('')
        while 'â' in tweet_list:
            tweet_list.remove('â')
        #print(tweet_list)
        # listtemp = [(x.lower(), x) for x in tweet_list]
        # listtemp.sort()
        # tweet_list_stem = [x[1] for x in listtemp]

        #print(tweet_list)
        cache = {}
        tweet_list_stem = []
        for t in tweet_list:
            if t in cache:
                tweet_list_stem.append(cache[t])
            else:
                cache[t] = stem(t)
                tweet_list_stem.append(cache[t])
        vocab_set = (lambda x: (x.sort(), x)[1])(tweet_list_stem)
        tweet_list = sorted(vocab_set, key=str.lower)

        #print(tweet_list)

        class_id = []
        for key in classIDs_Dict:
            if key in arr[2]:
                class_id.append(classIDs_Dict[key])

        feature = {}
        tweet_counter = Counter(tweet_list)
        tweet_counter = dict(tweet_counter)
        for i in tweet_list:
            if i in featsdic:
                feature[featsdic[i]] = tweet_counter[i]
            # else:
            #     feature[i] = tweet_counter[i]

        print(class_id, str(feature))
        str_class_id = str(class_id).replace('[','').replace(']', ' ')
        str_feature = str(feature).replace('{', '').replace(', ', ' ').replace(': ',':').replace('}','\n')
        wfile.write(str_class_id)
        wfile.write(str_feature)
    wfile.close()
    file.close()
    return 0

filelist = ['train','test']
for i in filelist:
    filename = 'Tweets.14cat.'+ i
    t = set_features(filename)

