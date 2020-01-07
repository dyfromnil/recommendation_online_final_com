from surprise import SVD
from surprise import Dataset
from surprise import dump
import pandas as pd
import pymysql
from sklearn.utils import shuffle
from surprise import Reader
from datetime import datetime
import numpy as np
from sklearn.externals import joblib
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import os
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix
import pickle
import re

#每天启动时清空推荐表
db = pymysql.connect("host", "user", "password", "database")
cursor = db.cursor()
cursor.execute("delete from zjc_intrec_supplier_recommend")
cursor.execute("ALTER TABLE zjc_intrec_supplier_recommend AUTO_INCREMENT=1")
db.commit()
db.close()

'''

state for '是否截标'
is_new for '新标书是否已经计算完毕并插入了insert_speed'

'''

# read data from MySql,return pandas dataframe
def readDataFromMySql(sql_order):
    db = pymysql.connect("host", "user", "password", "database")
    cursor = db.cursor()
    try:
        cursor.execute(sql_order)
        data = cursor.fetchall()
        data = pd.DataFrame(list(data))
    except:
        data = pd.DataFrame()
    db.close()
    return data

# ------------clustering------------

bid = readDataFromMySql("select * from v_intrec_bid")
bid=bid[[1,2,5,6]]
bid=bid[bid[2]!=0]
# bid.drop_duplicates(inplace=True)
bid=np.array(bid)

'''
bid=pd.read_csv('bid.csv')
bid=bid[['btype','materialclass','longitude','latitude']]
bid=np.array(bid)
'''

# 保存聚类前标准化数据集的std
std = StandardScaler()
bid[:, [2, 3]] = std.fit_transform(bid[:, [2, 3]])
joblib.dump(std, 'interDump/stdkproto.pkl')

print("cluster,begin...")
a=datetime.now()
# training kprototypes model
flag=1
while flag:
    kproto = KPrototypes(n_clusters=35,n_init=5,init='Huang', verbose=False)
    try:
        clusters = kproto.fit_predict(bid, categorical=[0, 1])
        flag=0
    except:
        flag=1
b=datetime.now()
print('cluster total:',(b-a).seconds,'s')

# dump the kprototypes model for later usage
joblib.dump(kproto, 'interDump/kprototypes.pkl')


# ------------SVD------------
#查看
data_check = readDataFromMySql("select * from zjc_intrec_supplier_check_log")
data_check=data_check[[2,1]]
data_check.columns=[0,1]
data_check[2] = 1
data_check.drop_duplicates(inplace=True)
#留言
data_quest = readDataFromMySql("select * from zjc_intrec_supplier_msg_log")
data_quest=data_quest[[2,1]]
data_quest.columns=[0,1]
data_quest[2] = 2
data_quest.drop_duplicates(inplace=True)
#关注
data_collect = readDataFromMySql("select * from zjc_intrec_supplier_interest_log")
data_collect=data_collect[[2,1]]
data_collect.columns=[0,1]
data_collect[2] = 3
data_collect.drop_duplicates(inplace=True)
#投标
data_tender = readDataFromMySql("select * from zjc_intrec_supplier_tender_log")
data_tender=data_tender[[2,1]]
data_tender.columns=[0,1]
data_tender[2] = 5
data_tender.drop_duplicates(inplace=True)

#供应商和标书的最大id
maxRow=max(max(data_check[0]),max(data_quest[0]),max(data_collect[0]),max(data_tender[0]))+1
maxCol=max(max(data_check[1]),max(data_quest[1]),max(data_collect[1]),max(data_tender[1]))+1



print("combing data,begin...")
a = datetime.now()

#将3类数据转化为稀疏矩阵
#is_check
row=np.array(data_check[0])
col=np.array(data_check[1])
value=np.array(data_check[2])
data_check=coo_matrix((value,(row,col)),shape=(maxRow,maxCol))
dataCombed=data_check.todok()
del data_check
#is_quest
row=np.array(data_quest[0])
col=np.array(data_quest[1])
value=np.array(data_quest[2])
data_quest=coo_matrix((value,(row,col)))
#is_collect
row=np.array(data_collect[0])
col=np.array(data_collect[1])
value=np.array(data_collect[2])
data_collect=coo_matrix((value,(row,col)))
#is_tender
row=np.array(data_tender[0])
col=np.array(data_tender[1])
value=np.array(data_tender[2])
data_tender=coo_matrix((value,(row,col)))

#combe
for index in range(data_quest.nnz):
    dataCombed[data_quest.row[index],data_quest.col[index]]=2

for index in range(data_collect.nnz):
    dataCombed[data_collect.row[index],data_collect.col[index]]=3

for index in range(data_tender.nnz):
    dataCombed[data_tender.row[index],data_tender.col[index]]=5

#转回dataframe
dataCombed=dataCombed.tocoo()
data_dict={'sid':dataCombed.row,'bid':dataCombed.col,'rating':dataCombed.data}
dataCombed=pd.DataFrame(data_dict)
b = datetime.now()
print("共", (b - a).seconds, "秒")


# transfer dataCombed into surpriseLib-SVD-fitting style
data = shuffle(dataCombed)
del dataCombed
data.to_csv("dataForDump/trainingData.data",
                    sep='\t', header=False, index=False)
reader = Reader(line_format='user item rating', sep='\t')
file_path = os.path.expanduser('dataForDump/trainingData.data')
dataForTraining=Dataset.load_from_file(file_path,reader=reader)
dataForTraining = dataForTraining.build_full_trainset()

# fitting...
algo = SVD(n_factors=30, n_epochs=30, lr_all=0.009, reg_all=0.08)
algo.fit(dataForTraining)

# Dump the SVD predictions for later usage
dump.dump("interDump/svd-predictions", predictions=None, algo=algo, verbose=False)
# prediction,algor = dump.load("svd-predictions")

# ----------------Insert supplierId-bidId-score into database----------------

# get ids of all suppliers who have at lease one operation
supplierId = data[['sid']].copy()
supplierId.drop_duplicates(inplace=True)

a=datetime.now()
#supplierData_dict供应商主营辅营物资转字典
supplierData_dict = {}
supplierData = readDataFromMySql('select supplier_id, mian_materialclass, sub_materialclass from zjc_intrec_supplier')

for index, row in supplierData.iterrows():
    '''
    supplier_Id:供应商id
    materialclass:辅营物资
    supplygoods:主营物资
    tot_supply:所有经营物资
    '''
    supplier_Id = row[0]
    if row[1]=='[]' or row[1]=='':
        materialclass=[]
    else:
        materialclass = re.split('\D+',row[1])
        materialclass.pop();materialclass.pop(0)
    if row[2]=='[]' or row[2]=='':
        supplygoods=[]
    else:
        supplygoods = re.split('\D+',row[2])
        supplygoods.pop();supplygoods.pop(0)
    tot_supply = materialclass + supplygoods
    supplierData_dict[supplier_Id] = tot_supply

supplierData_dict_later={}
supplierId=np.array(supplierId).flatten()
#supplierData_dict_later与有历史操作行为的supplierId同步
for sid in supplierId:
    supplierData_dict_later[sid]=supplierData_dict.get(sid, [])
b=datetime.now()
print("转字典，共", (b - a).seconds, "秒")

with open('interDump/supplierData.pkl', 'wb') as write_file:
    pickle.dump(supplierData_dict_later, write_file)

#dump supplierId for later usage
# np.save("interDump/supplierId.npy",np.array(supplierId).flatten())

# get ids of all bids which still online
bidData = readDataFromMySql("select * from zjc_intrec_bid where state=3")
bidData = bidData[[1,4]]
bidData.columns=[0,1]
bidData.drop_duplicates(inplace=True)

print('calculate scores,begin...')
a = datetime.now()
con=create_engine("mysql+pymysql://username:password@host(:port)/database",encoding='utf-8')
for bids_index, bids_row in bidData.iterrows():
    df=[]
    for key, value in supplierData_dict_later.items():
        if  (str(int(bids_row[1])) in value) or (not value):
            df.append((key, bids_row[0], round(algo.predict(
                str(key), str(bids_row[0]), verbose=False).est,4)))
    df=pd.DataFrame(df)
    df.columns=['supplier_id','bid_id','score']
    df[['supplier_id','bid_id']]=df[['supplier_id','bid_id']].astype('int')
    
    df.to_sql(name="zjc_intrec_supplier_recommend",con=con,if_exists='append',index=False)
b = datetime.now()
print("共", (b - a).seconds, "秒")
