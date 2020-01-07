import pymysql
from sklearn.externals import joblib
from surprise import dump
from surprise import SVD
import pandas as pd
import numpy as np
from datetime import datetime
import time
from sqlalchemy import create_engine
import pickle

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


def get_nearest_points(bid_data):
    # 分类属性
    categorical = [0, 1]
    #标准化数值数据
    bid_data[:, [2, 3]] = std.transform(bid_data[:, [2, 3]])
    # 预测样本的类别，并找出该类别所有其他样本，将其他样本按与该预测样本的距离从小到大排序
    _ , near_points = kproto.get_nearest(bid_data, categorical=categorical)
    #根据near_points获取对应的标书id
    match_bid_id=readDataFromMySql("select bid_id from v_intrec_bid")
    match_bid_id=np.array(match_bid_id).flatten()
    near_points[0]=match_bid_id[near_points[0]]

    return near_points



def get_score(uid, near_points):
    score_list = []
    for j in range(len(near_points)):
        score = 0
        if len(near_points[j]) > 10:
            for i in range(10):
                score += algo.predict(str(uid),
                                      str(near_points[j][i]), verbose=False).est
            score /= 10
        else:
            for bid in near_points[j]:
                score += algo.predict(str(uid), str(bid)).est
            score /= len(near_points[j])
        score_list.append(score)

    return score_list

# load the kprototypes&std models , SVD model and supplierId
kproto = joblib.load('interDump/kprototypes.pkl')
std = joblib.load('interDump/stdkproto.pkl')
predictions, algo = dump.load("interDump/svd-predictions")
# supplierId=np.load("interDump/supplierId.npy")
with open('interDump/supplierData.pkl', 'rb') as read_file:
    supplierData_dict_later = pickle.load(read_file)


'''-----------评分转为list-tuple--->dataframe利用engine进行插入------------'''
# while(True):
    #find the new-added bids
bids = readDataFromMySql("select * from v_intrec_bid where is_new=0 and state=3")

if not (bids.empty):
    bids = bids[[0,1,2,4,5]]

    #update the state1 from 0 to 1
    db = pymysql.connect("host", "user", "password", "database")
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    for bids_index, bids_row in bids.iterrows():
        sql="UPDATE zjc_intrec_bid SET is_new=1 WHERE bid_id = {}".format(int(bids_row[0]))
        db.ping(reconnect=True)
        try:
            # 执行sql语句
            cursor.execute(sql)
            db.commit()
        except:
            # 如果发生错误则回滚
            # db.rollback()
            print("error! ", ' ', bids_row[0])
    # 关闭数据库连接
    db.close()

    a=datetime.now()

    con=create_engine("mysql+pymysql://username:password@host(:port)/database",encoding='utf-8')
    for bids_index, bids_row in bids.iterrows():
        df=[]
        near_points = get_nearest_points(np.array([[int(bids_row[1]),int(bids_row[2]),bids_row[4],bids_row[5]],]))
        for key,value in supplierData_dict_later.items():
            # pred_score = get_score(sId, near_points)[0]
            if (str(int(bids_row[2])) in value) or (not value):
                df.append((key,bids_row[0],round(get_score(key, near_points)[0],4)))
        df=pd.DataFrame(df)
        df.columns=['supplier_id','bid_id','score']
        df[['supplier_id','bid_id']]=df[['supplier_id','bid_id']].astype('int')
        
        df.to_sql(name="zjc_intrec_supplier_recommend",con=con,if_exists='append',index=False)

    b = datetime.now()
    print("共", (b - a).seconds, "秒")
