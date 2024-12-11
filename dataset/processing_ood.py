import numpy as np
import pandas as pd
import gzip
import json

dataset_name = "Beauty"
dataset_file = "amazon_beauty/"

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


df = getDF(dataset_file + 'reviews_%s_5.json.gz' % dataset_name)
print(df.head(2))

metadata = []
with gzip.open(dataset_file + 'meta_%s.json.gz' % dataset_name) as f:
    for l in f:
        metadata.append(eval(l.strip()))

# total length of list, this number equals total number of products
print(len(metadata))

# first row of the list
print(metadata[0])

# convert list into pandas dataframe

df_meta = pd.DataFrame.from_dict(metadata)

print(len(df_meta))

df3 = df_meta.fillna('')
df4 = df3[df3.title.str.contains('getTime')]  # unformatted rows
df5 = df3[~df3.title.str.contains('getTime')]  # filter those unformatted rows
print(len(df4))
print(len(df5))

rating = df[['overall', 'reviewTime', 'reviewerID', 'asin', 'unixReviewTime']]
print(rating.head(2))

rating = rating[['asin', 'reviewerID', 'overall', 'unixReviewTime']]
rating.columns = ['asin', 'user', 'rating', 'timestamp']

meta_data = df5[['asin', 'categories', 'title', 'brand', 'price']].rename(columns={'categories': 'category'})
print(meta_data.head(2))

print(meta_data.shape)
meta_data = meta_data.drop_duplicates(subset=['asin'], keep='last')
print(meta_data.shape)
print(meta_data.head(2))

data = rating.merge(meta_data, on='asin', how='right')

data = data.dropna()

rating_ = data.copy()

rating_.columns = ['iid', 'uid', 'rating', 'timestamp', 'category', 'title', 'brand', 'price']

date_min = pd.to_datetime(rating_.timestamp, unit='s').min()
date_max = pd.to_datetime(rating_.timestamp, unit='s').max()
print(date_min, date_max)

date_gap = (date_max - date_min) // (19 * 2)

rating_['time'] = pd.to_datetime(rating_.timestamp, unit='s').map(lambda x: x.year)

np.sort(rating_.time.unique())

s_rating = rating_[rating_.time.isin(rating_.time.unique().tolist())].copy()
s_rating['time'] = pd.to_datetime(s_rating.timestamp, unit='s').map(lambda x: x.month)
s_rating = s_rating[s_rating.time.isin(range(1, 13))]
print(s_rating.shape)

item_info = s_rating.groupby('iid').agg({"rating": 'count'})
user_info = s_rating.groupby('uid').agg({"rating": 'count'})

active_item = item_info[item_info['rating'] > 5].index  # .sample(frac=10/20,random_state=2023).index
active_user = user_info[user_info['rating'] > 5].index  # .sample(frac=10/20,random_state=2023).index

s_rating = s_rating[s_rating['uid'].isin(active_user)]
s_rating = s_rating[s_rating['iid'].isin(active_item)]

item_info = s_rating.groupby('iid').agg({"rating": 'count'})
user_info = s_rating.groupby('uid').agg({"rating": 'count'})

s_rating = s_rating.reset_index()

# rating or CTR
s_rating['label'] = s_rating['rating'].apply(lambda x: 1 if x >= 5 else 0)
# s_rating['label'] = s_rating['rating']


users = s_rating.uid.unique()
items = s_rating.iid.unique()
users_map = dict(zip(users, (np.arange(users.shape[0]) + 1).tolist()))
items_map = dict(zip(items, (np.arange(items.shape[0]) + 1).tolist()))
user_dict = json.dumps(users_map)
item_dict = json.dumps(items_map)

with open(dataset_file + "user_dict.json", 'w') as json_file:
    json_file.write(user_dict)
with open(dataset_file + "item_dict.json", 'w') as json_file:
    json_file.write(item_dict)
s_rating['uid'] = s_rating['uid'].map(users_map)
s_rating['iid'] = s_rating['iid'].map(items_map)

rating_train = s_rating[s_rating.time.isin(range(1, 12))].copy()
rating_valid_test = s_rating[s_rating.time.isin([12])].copy()
rating_valid_test.sort_values(by="timestamp", inplace=True)
N_ = rating_valid_test.shape[0] // 2
rating_valid = rating_valid_test.iloc[:N_].copy()
rating_test = rating_valid_test.iloc[N_:].copy()

rating_valid_test.timestamp.values[0:5].argsort()

train_user = rating_train['uid'].unique()
train_item = rating_train['iid'].unique()
rating_valid['not_cold'] = rating_valid[['uid', 'iid']].apply(lambda x: x.uid in train_user and x.iid in train_item,
                                                              axis=1).astype("int")
rating_test['not_cold'] = rating_test[['uid', 'iid']].apply(lambda x: x.uid in train_user and x.iid in train_item,
                                                            axis=1).astype("int")

rating_valid_f = rating_valid
rating_test_f = rating_test


def filter_cold_start(train, valid, test):
    train_user = train.uid.unique()
    train_item = train.iid.unique()
    valid = valid[valid['uid'].isin(train_user)]
    test = test[test['uid'].isin(train_user)]
    valid = valid[valid['iid'].isin(train_item)]
    test = test[test['iid'].isin(train_item)]
    return valid, test


import copy


def deal_with_each_u(x, u):
    items = np.array(x.iid)
    labels = np.array(x.label)
    titles = np.array(x.title)
    timestamp = np.array(x.timestamp)
    flags = np.array(x.flag)
    his = [0]  # adding a '0' by default
    his_title = ['']
    results = []
    for i in range(items.shape[0]):
        results.append((u, items[i], timestamp[i], np.array(his), copy.copy(his_title), titles[i], labels[i], flags[i]))
        # training data
        if labels[i] > 0:  # positive
            his.append(items[i])
            his_title.append(titles[i])
    return results


rating_train = rating_train.copy()

rating_train['flag'] = pd.DataFrame(np.ones(rating_train.shape[0]) * -1, index=rating_train.index)
rating_valid_f['flag'] = pd.DataFrame(np.zeros(rating_valid_f.shape[0]), index=rating_valid_f.index)
rating_test_f['flag'] = pd.DataFrame(np.ones(rating_test_f.shape[0]), index=rating_test_f.index)
data_ = pd.concat([rating_train, rating_valid_f, rating_test_f], axis=0, ignore_index=True)
data_ = data_.sort_values(by=['uid', 'timestamp'])
u_inter_all = data_.groupby('uid').agg({'iid': list, 'label': list, 'title': list, 'timestamp': list, 'flag': list})

results = []
for u in u_inter_all.index:
    results.extend(deal_with_each_u(u_inter_all.loc[u], u))

u_, i_, time_, label_, his_, his_title, title_, flag_ = [], [], [], [], [], [], [], []
for re_ in results:
    u_.append(re_[0])
    i_.append(re_[1])
    time_.append(re_[2])
    his_.append(re_[3])
    his_title.append(re_[4])
    title_.append(re_[5])
    label_.append(re_[6])
    flag_.append(re_[7])

data_ = pd.DataFrame(
    {"uid": u_, 'iid': i_, 'label': label_, 'timestamp': time_, 'his': his_, 'his_title': his_title, 'title': title_,
     'flag': flag_})

train_ = data_[data_['flag'].isin([-1])].copy()
valid_ = data_[data_['flag'].isin([0])].copy()
test_ = data_[data_['flag'].isin([1])].copy()

train_user = train_['uid'].unique()
train_item = train_['iid'].unique()
valid_['not_cold'] = valid_[['uid', 'iid']].apply(lambda x: x.uid in train_user and x.iid in train_item, axis=1).astype(
    "int")
test_['not_cold'] = test_[['uid', 'iid']].apply(lambda x: x.uid in train_user and x.iid in train_item, axis=1).astype(
    "int")

train_['not_cold'] = pd.DataFrame(np.ones(train_.shape[0]), index=train_.index).astype("int")

train_.to_pickle(dataset_file + "train_ood2.pkl")
valid_.to_pickle(dataset_file + "valid_ood2.pkl")
test_.to_pickle(dataset_file + "test_ood2.pkl")

valid_small = valid_.sample(frac=0.25, random_state=2023)
valid_small.to_pickle(dataset_file + "valid_small_ood_rating.pkl")

print(train_.shape, valid_.shape, test_.shape)
