from pyspark import SparkContext
import sys
import json
import time
import copy
import itertools
import re
import math
from collections import Counter

# spark-submit task3predict.py train_review.json test_review.json task3item.model task3.res item_based

start_time = time.time()

train_file = sys.argv[1]  # train_review.json
test_file = sys.argv[2]  # train_review.json
model_file = sys.argv[3]  # task3item.model task3user.model
output_file = sys.argv[4]
cf_type = sys.argv[5]  # either item_based or user_based

sc = SparkContext('local[*]', 'task3')
sc.setLogLevel('ERROR')

train0 = sc.textFile(train_file)

test0 = sc.textFile(test_file)

model0 = sc.textFile(model_file)

train1 = train0.map(
    lambda line: (json.loads(line)['user_id'], json.loads(line)['business_id'], int(json.loads(line)['stars'])))

test1 = test0.map(lambda line: (json.loads(line)['user_id'], json.loads(line)['business_id']))

if cf_type == 'item_based':

    # hyper-parameter
    n = 20


    def merge_two_dicts(x, y):
        z = x.copy()  # start with x's keys and values
        z.update(y)  # modifies z with y's keys and values & returns None
        return z


    model1 = model0.map(lambda line: (json.loads(line)['b1'], json.loads(line)['b2'], json.loads(line)['sim'])) \
        .map(lambda x: [(x[0], {x[1]: x[2]}), (x[1], {x[0]: x[2]})]) \
        .flatMap(lambda x: x) \
        .reduceByKey(lambda a, b: merge_two_dicts(a, b)) \
        .collect()

    sim_dic = {item[0]: item[1] for item in model1}

    #
    # for item in model1:
    #     b1 = item[0][0]
    #     b2 = item[0][1]
    #     sim = item[1]
    #     if b1 not in sim_dic:
    #         sim_dic[b1] = {}
    #     if b2 not in sim_dic:
    #         sim_dic[b2] = {}
    #     sim_dic[b1][b2] = sim
    #     sim_dic[b2][b1] = sim

    train2 = train1.map(lambda x: (x[1], {x[0]: x[2]})) \
        .reduceByKey(lambda a, b: merge_two_dicts(a, b)).collect()

    train_dic = {item[0]: item[1] for item in train2}

    # test2 = test1.map(lambda x: (x[1], 1)).reduceByKey(lambda a, b: a)
    #
    # print(len(sim_dic), len(train_dic), test2.count())

    def predict(x):
        user = x[0]
        business = x[1]
        strs = -9999
        if business in sim_dic:
            b2s = sim_dic[business]  # {b2: star}
            nume = 0
            deno = 0
            new_b2s = {}
            for key, val in b2s.items():
                try:
                    trytry = train_dic[key][user]
                    new_b2s[key] = val
                except:
                    pass
            topn = dict(Counter(new_b2s).most_common(min(n, len(new_b2s))))
            for key, val in topn.items():
                star = train_dic[key][user]
                nume += star * val
                deno += abs(val)
            if deno == 0:
                strs = -999
            else:
                strs = nume / deno
        else:
            strs = -99
        return ((user, business), strs)


    test2 = test1.map(lambda x: predict(x)) \
        .filter(lambda x: x[1] >= 0) \
        .mapValues(lambda x: max(2, min(4.95, x))) \
        .collect()

    output = open(output_file, "w")

    for pair in test2:
        output.write('{"user_id": "' + str(pair[0][0]) + '", "business_id": "' + str(pair[0][1]) + '", "stars": ' + str(
            pair[1]) + '}')
        output.write("\n")

if cf_type == 'user_based':
    # hyper-parameter
    n = 20

    def merge_two_dicts(x, y):
        z = x.copy()  # start with x's keys and values
        z.update(y)  # modifies z with y's keys and values & returns None
        return z

    model1 = model0.map(lambda line: (json.loads(line)['u1'], json.loads(line)['u2'], json.loads(line)['sim'])) \
        .map(lambda x: [(x[0], {x[1]: x[2]}), (x[1], {x[0]: x[2]})]) \
        .flatMap(lambda x: x) \
        .reduceByKey(lambda a, b: merge_two_dicts(a, b)) \
        .collect()

    sim_dic = {item[0]: item[1] for item in model1}

    # for item in model1:
    #     u1 = item[0][0]
    #     u2 = item[0][1]
    #     sim = item[1]
    #     if u1 not in sim_dic:
    #         sim_dic[u1] = {}
    #     if u2 not in sim_dic:
    #         sim_dic[u2] = {}
    #     sim_dic[u1][u2] = sim
    #     sim_dic[u2][u1] = sim

    train2 = train1.map(lambda x: (x[0], {x[1]: x[2]})) \
        .reduceByKey(lambda a, b: merge_two_dicts(a, b)).collect()  # user, {business: star}

    train_dic = {item[0]: item[1] for item in train2}


    def predict(x):
        user = x[0]
        business = x[1]
        strs = -9999
        if user in sim_dic:
            u2s = sim_dic[user]  # {b2: sim}
            nume = 0
            deno = 0
            new_u2s = {}
            for key, val in u2s.items():
                try:
                    trytry = train_dic[key][business]
                    new_u2s[key] = val
                except:
                    pass
            topn = dict(Counter(new_u2s).most_common(min(n, len(new_u2s))))
            mean1 = sum(train_dic[user].values()) / len(train_dic[user].values())
            for key, val in topn.items():  # user2, sim
                star = train_dic[key][business]
                mean2 = (sum(train_dic[key].values()) - star) / (len(train_dic[key].values()) - 1)
                nume += (star - mean2) * val
                deno += abs(val)
            if deno == 0:
                strs = -999
            else:
                strs = mean1 + nume / deno
        else:
            strs = -99
        return ((user, business), strs)


    test2 = test1.map(lambda x: predict(x)) \
        .filter(lambda x: x[1] >= 0) \
        .mapValues(lambda x: max(1, min(5, x))) \
        .collect()

    output = open(output_file, "w")

    for pair in test2:
        output.write('{"user_id": "' + str(pair[0][0]) + '", "business_id": "' + str(pair[0][1]) + '", "stars": ' + str(
            pair[1]) + '}')
        output.write("\n")

print('Duration: ' + str(time.time() - start_time))
