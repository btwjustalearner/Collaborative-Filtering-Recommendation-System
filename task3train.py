from pyspark import SparkContext
import sys
import json
import time
import copy
import itertools
import re
import math
import os
import shutil
import gc

# spark-submit task3train.py train_review.json task3item.model item_based

start_time = time.time()

train_file = sys.argv[1]  # train_review.json
model_file = sys.argv[2]  # task3item.model task3user.model
cf_type = sys.argv[3]  # either item_based or user_based

sc = SparkContext('local[*]', 'task3')
sc.setLogLevel('ERROR')


RDD0 = sc.textFile(train_file)

RDD00 = RDD0.map(lambda line: (json.loads(line)['user_id'], json.loads(line)['business_id'], int(json.loads(line)['stars'])))

if cf_type == 'item_based':

    RDD1 = RDD00.map(lambda x: (x[0], [x[1]]))

    RDD2 = RDD1.reduceByKey(lambda a, b: a + b)\
        .mapValues(lambda v: set(v))  # user, {businesses}


    def merge_two_dicts(x, y):
        z = x.copy()  # start with x's keys and values
        z.update(y)  # modifies z with y's keys and values & returns None
        return z


    RDD3 = RDD00.map(lambda x: (x[1], {x[0]: x[2]}))\
        .reduceByKey(lambda a, b: merge_two_dicts(a, b)) \
        .sortBy(lambda x: x[0]).collect()  # business, {user: star}

    RDD4 = RDD2.flatMapValues(lambda x: itertools.combinations(x, 2))\
        .mapValues(lambda x: tuple(sorted(x))) \
        .map(lambda x: (x[1], [x[0]])) \
        .reduceByKey(lambda a, b: a + b) \
        .mapValues(lambda x: set(x)) \
        .filter(lambda x: len(x[1]) >= 3)  # (b1, b2), {users}

    b2u2s_dict = {}
    for item in RDD3:
        b2u2s_dict[item[0]] = item[1]

    def computePearson(x):
        b1 = x[0][0]
        b2 = x[0][1]
        users = x[1]
        b2u2s1 = b2u2s_dict[b1]  # {user: star}
        b2u2s2 = b2u2s_dict[b2]
        mean1 = sum(b2u2s1.values())/len(b2u2s1.values())
        mean2 = sum(b2u2s2.values()) / len(b2u2s2.values())
        nume = 0
        deno1 = 0
        deno2 = 0
        for user in users:
            r1 = b2u2s1[user]
            r2 = b2u2s2[user]
            p1 = r1 - mean1
            p2 = r2 - mean2
            nume += p1 * p2
            deno1 += p1 * p1
            deno2 += p2 * p2
        if (deno1 == 0) | (deno2 == 0):
            w12 = 0
        else:
            w12 = nume/(math.sqrt(deno1) * math.sqrt(deno2))
        return((b1, b2), w12)


    RDD5 = RDD4.map(lambda x: computePearson(x))\
        .filter(lambda x: x[1] > 0).collect()

    output = open(model_file, "w")

    for pair in RDD5:
        output.write('{"b1": "' + str(pair[0][0]) + '", "b2": "' + str(pair[0][1]) + '", "sim": ' + str(pair[1]) + '}')
        output.write("\n")

if cf_type == 'user_based':


    # minhash

    # hyper parameters
    HASH_LEN = 18
    BAND_LEN = 18
    ROW_LEN = int(HASH_LEN / BAND_LEN)

    RDD1 = RDD00.map(lambda x: (x[1], x[0]))  # business, user

    u_ids = RDD1.map(lambda x: x[1]).distinct().collect()
    b_ids = RDD1.map(lambda x: x[0]).distinct().collect()
    u_ids.sort()
    b_ids.sort()

    u_len = len(u_ids)
    b_len = len(b_ids)

    u_index_map = {}
    b_index_map = {}
    index_u_map = {}
    index_b_map = {}

    for index, user in enumerate(u_ids):
        u_index_map[user] = index
        index_u_map[index] = user

    for index, business in enumerate(b_ids):
        b_index_map[business] = index
        index_b_map[index] = business

    RDD2 = RDD1.map(lambda x: (u_index_map[x[1]], [b_index_map[x[0]]])).reduceByKey(lambda a, b: a + b) \
        .mapValues(lambda x: sorted(list(set(x)))).sortBy(lambda x: x[0])  # user, {businesses}

    characteristic_matrix = RDD2.collect()

    prime_numbers = [188189, 193261, 201947]

    # hash


    def hashFunc(r, h):
        return (74923 * r + 13441 * h) % 151507


    def sigFunc(x):
        sig_list = []
        for hash_id in range(HASH_LEN):
            sig = float("inf")
            for row_id in x[1]:
                sig = min(sig, hashFunc(row_id, hash_id))
            sig_list.append(sig)
        return sig_list


    signature_matrix = RDD2.map(lambda x: sigFunc(x)).collect()

    # LSH
    candidate_set = set()
    similar_set = set()


    def dotProductFunc(l1, l2):
        dot = sum([l1[i] * l2[i] for i in range(len(l2))])
        return dot


    def hashBandFunc(signatures, start, end):
        hashVal = dotProductFunc(prime_numbers[:ROW_LEN], signatures[start:(end + 1)])
        hashVal = hashVal % 202201
        return hashVal


    for band in range(BAND_LEN):
        start = band * ROW_LEN
        end = band * ROW_LEN + ROW_LEN - 1

        bandBucket = {}

        for user in range(u_len):
            bucket = hashBandFunc(signature_matrix[user], start, end)
            if bucket in bandBucket.keys():
                bandBucket[bucket].append(user)
            else:
                bandBucket[bucket] = []
                bandBucket[bucket].append(user)

        for key, val in bandBucket.items():
            cand_pairs = itertools.combinations(val, 2)

            for pair in cand_pairs:
                candidate_set.add(tuple(sorted(pair)))


    def JacSim(l1, l2):
        s1 = set(l1)
        s2 = set(l2)
        return len(s1.intersection(s2)) / len(s1.union(s2))


    for pair in candidate_set:
        jaccSimilarity = JacSim(characteristic_matrix[pair[0]][1], characteristic_matrix[pair[1]][1])
        if jaccSimilarity >= 0.01:
            similar_set.add((index_u_map[pair[0]], index_u_map[pair[1]]))


    del characteristic_matrix, signature_matrix
    gc.collect()

    jacsimpairs = list(similar_set)
    jacsimpairs.sort()


    #########
    # RDD11 = RDD00.map(lambda x: (x[1], [x[0]]))  # business, [user]
    #
    # RDD21 = RDD11.reduceByKey(lambda a, b: a + b)\
    #     .mapValues(lambda v: set(v))  # business, {users}


    def merge_two_dicts(x, y):
        z = x.copy()  # start with x's keys and values
        z.update(y)  # modifies z with y's keys and values & returns None
        return z


    RDD3 = RDD00.map(lambda x: (x[0], {x[1]: x[2]}))\
        .reduceByKey(lambda a, b: merge_two_dicts(a, b)) \
        .sortBy(lambda x: x[0]).collect()  # user, {business: star}


    u2b2s_dict = {}
    for item in RDD3:
        u2b2s_dict[item[0]] = item[1]


    gc.collect()


    def computePearson(x):
        u1 = x[0]
        u2 = x[1]

        u2b2s1 = u2b2s_dict[u1]  # {business: star}
        u2b2s2 = u2b2s_dict[u2]

        bus1 = list(u2b2s1.keys())
        bus2 = list(u2b2s2.keys())

        s1 = set(bus1)
        s2 = set(bus2)
        businesses = s1.intersection(s2)

        if len(businesses) >= 3:
            nume = 0
            deno1 = 0
            deno2 = 0
            b_list1 = []
            b_list2 = []
            for bu in businesses:
                b_list1.append(u2b2s1[bu])
                b_list2.append(u2b2s2[bu])
            mean1 = sum(b_list1) / len(b_list1)
            mean2 = sum(b_list2) / len(b_list2)
            for i in range(len(b_list1)):
                r1 = b_list1[i]
                r2 = b_list2[i]
                p1 = r1 - mean1
                p2 = r2 - mean2
                nume += p1 * p2
                deno1 += p1 * p1
                deno2 += p2 * p2
            if (deno1 == 0) | (deno2 == 0):
                w12 = 0
            else:
                w12 = nume/(math.sqrt(deno1) * math.sqrt(deno2))
        else:
            w12 = -999
        return((u1, u2), w12)


    numPartitions = 100

    results1 = sc.parallelize(jacsimpairs, numPartitions) \
        .map(lambda x: computePearson(x))\
        .filter(lambda x: x[1] > 0)\
        .map(lambda x: ((u_index_map[x[0][0]], u_index_map[x[0][1]]), x[1])).collect()

    output = open(model_file, "w")

    for result in results1:
        output.write('{"u1": "' + index_u_map[result[0][0]] + '", "u2": "' + index_u_map[result[0][1]] + '", "sim": ' + str(result[1]) + '}')
        output.write("\n")



print('Duration: ' + str(time.time() - start_time))
