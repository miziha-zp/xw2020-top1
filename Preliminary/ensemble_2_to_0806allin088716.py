#!/usr/bin/env python
# coding: utf-8

# In[1]:


# tijiao
# 0.85861
# 0.88053


# In[2]:


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
# import seaborn as sns
# import matplotlib.pyplot as plt
from scipy.stats import entropy
import gc
import os
from tqdm import tqdm
# pd.set_option('display.max_columns', 300)
# pd.set_option('display.max_rows', 50)
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"


# In[3]:


def acc_combo(y, y_pred):
    # 数值ID与行为编码的对应关系
    mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3', 
        4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5', 
        8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6', 
        12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6', 
        16: 'C_2', 17: 'C_5', 18: 'C_6'}
    # 将行为ID转为编码
    code_y, code_y_pred = mapping[y], mapping[y_pred]
    if code_y == code_y_pred: #编码完全相同得分1.0
        return 1.0
    elif code_y.split("_")[0] == code_y_pred.split("_")[0]: #编码仅字母部分相同得分1.0/7
        return 1.0/7
    elif code_y.split("_")[1] == code_y_pred.split("_")[1]: #编码仅数字部分相同得分1.0/3
        return 1.0/3
    else:
        return 0.0


# In[4]:


num2detail_mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3', 
        4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5', 
        8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6', 
        12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6', 
        16: 'C_2', 17: 'C_5', 18: 'C_6'}


# In[5]:


def feature_test_with_cv(X, y, params=None, cate_feas='auto', nfold=3):
    """
    [For LightGBM ONLY]
    Use cross validation to test if the feature distribution is the same in both train and test sets.
    y: 'istest' column with valus of 0-1
    Example:
        df_fea_auc = get_feature_report(df, features=gcn_feas, all_cate_feas=[], params=None,nfold=3)
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'early_stopping_rounds': 25,
            'metric': 'auc',
            'n_jobs': -1,
            'num_leaves': 31,
            'seed': 2020
        }
    sfold = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2020)
    models = []
    val_auc = 0
    train_auc = 0
    oof = np.zeros(len(X))
    for _, (train_idx, val_idx) in enumerate(sfold.split(X, y)):
        train_set = lgb.Dataset(X.iloc[train_idx],
                                y.iloc[train_idx],
                                categorical_feature=cate_feas)
        val_set = lgb.Dataset(X.iloc[val_idx],
                              y.iloc[val_idx],
                              categorical_feature=cate_feas)
        model = lgb.train(params,
                          train_set,
                          valid_sets=[train_set, val_set],
                          verbose_eval=20)
        val_auc += model.best_score['valid_1']['auc'] / 3
        train_auc += model.best_score['training']['auc'] / 3
        oof[val_idx] = model.predict(X.iloc[val_idx])
        models.append(model)
    return train_auc, val_auc, models, oof

def get_feature_report_by_covariate_shift_test(df_raw,
                                               features=None,
                                               all_cate_feas=[],
                                               params=None,
                                               nfold=3,
                                               y2test='istrain',
                                               train_all_feas=False):
    """
    Use cross validation to test if the feature distribution is the same in both train and test sets.
    Args:
        y2test: target to test, 'istrain' or 'istest'
        train_all_feas: if True, the model will be trained with all features together
    Return:
        result_dict: dict
    """
    df = df_raw.copy()
    del df_raw
    gc.collect()
    if features is None:
#         logger.info(
#             "features is none, all cols will be used except 'istrain' or 'istest'!"
#         )
        features = [
            col for col in df.columns if col not in ['istrain', 'istest']
        ]
    if train_all_feas:
        train_auc, val_auc, models, oof = feature_test_with_cv(
            X=df[features],
            y=df[y2test],
            params=params,
            cate_feas=[col for col in all_cate_feas if col in features],
            nfold=nfold)
        df['pred'] = oof
        if y2test == 'istrain':
            weights = df[df['istrain'] == 1]['pred'].values
            weights = (1. / weights) - 1.
            weights /= np.mean(weights)
        elif y2test == 'istest':
            weights = df[df['istest'] == 0]['pred'].values
            weights = (1. / (1 - weights)) - 1.
            weights /= np.mean(weights)
        else:
            raise NotImplementedError(
                "y2test should be in ['istrain','istest'] !")
        result_dict = {
            'train_auc': train_auc,
            'val_auc': val_auc,
            'models': models,
            'weights': weights
        }
    else:
        score_lst = []
        fea_lst = []
        for fea in features:
            if fea in all_cate_feas:
                cate_feas = [fea]
            else:
                cate_feas = 'auto'
#             logger.info("=" * 30)
#             logger.info(f"Testing: <{fea}> ...")
#             logger.info("=" * 30)
            train_auc, val_auc, _, _ = feature_test_with_cv(
                X=df[[fea]],
                y=df[y2test],
                params=params,
                cate_feas=cate_feas,
                nfold=nfold)
            fea_lst.append(fea)
            score_lst.append((train_auc, val_auc))
        df_fea_auc = pd.DataFrame(score_lst, columns=['train_auc', 'val_auc'])
        df_fea_auc['feat'] = fea_lst
        df_fea_auc = df_fea_auc.sort_values(by='val_auc', ascending=False)
        result_dict = {'df_fea_auc': df_fea_auc}
    return result_dict


# In[6]:


data_path = 'data/'
data_train = pd.read_csv(data_path+'sensor_train.csv')
data_test = pd.read_csv(data_path+'sensor_test.csv')
data_test['fragment_id'] += 10000
label = 'behavior_id'
data = pd.concat([data_train, data_test], sort=False).sort_values(["fragment_id","time_point"])
data.head()
df = data.drop_duplicates(subset=['fragment_id']).reset_index(drop=True)[['fragment_id', 'behavior_id']]
df.head()


# In[7]:


import joblib
data_path = "PKL/"


now_filepath=data_path + "0730_generator_one_fourth_orig_mixup_087765"
oof_test_data = joblib.load(os.path.join(now_filepath,[x for x in os.listdir(now_filepath) if 'pkl' in x][0]))
oof = oof_test_data["oof"]
preds = oof_test_data["test"]
for i in oof_test_data:
    print(i)
stacknp = np.concatenate([oof,preds],axis=0)
print(stacknp.shape)
stackpd = pd.DataFrame(data=stacknp,columns=["0730_generator_one_fourth_orig_mixup_087765_class"+str(i)for i in range(19)])
stackpd["fragment_id"] = df["fragment_id"]
df = df.merge(stackpd,how='left',on='fragment_id')
C5 = ["0730_generator_one_fourth_orig_mixup_087765_class"+str(i)for i in range(19)]

####################################################################
now_filepath=data_path + "0730_generator_one_fifth_orig_mixup_087099"
oof_test_data = joblib.load(os.path.join(now_filepath,[x for x in os.listdir(now_filepath) if 'pkl' in x][0]))
oof = oof_test_data["oof"]
preds = oof_test_data["test"]
for i in oof_test_data:
    print(i)
stacknp = np.concatenate([oof,preds],axis=0)
print(stacknp.shape)
stackpd = pd.DataFrame(data=stacknp,columns=["0730_generator_one_fifth_orig_mixup_087099_class"+str(i)for i in range(19)])
stackpd["fragment_id"] = df["fragment_id"]
df = df.merge(stackpd,how='left',on='fragment_id')
C2 = ["0730_generator_one_fifth_orig_mixup_087099_class"+str(i)for i in range(19)]

####################################################################
now_filepath=data_path + "0729_generator_one_third_orig_mixup_086223"
oof_test_data = joblib.load(os.path.join(now_filepath,[x for x in os.listdir(now_filepath) if 'pkl' in x][0]))
oof = oof_test_data["oof"]
preds = oof_test_data["test"]
for i in oof_test_data:
    print(i)
stacknp = np.concatenate([oof,preds],axis=0)
print(stacknp.shape)
stackpd = pd.DataFrame(data=stacknp,columns=["0729_generator_one_third_orig_mixup_086223_class"+str(i)for i in range(19)])
stackpd["fragment_id"] = df["fragment_id"]
df = df.merge(stackpd,how='left',on='fragment_id')
C3 = ["0729_generator_one_third_orig_mixup_086223_class"+str(i)for i in range(19)]

####################################################################
now_filepath=data_path + "0729_generator_one_sixth_orig_mixup_086686"
oof_test_data = joblib.load(os.path.join(now_filepath,[x for x in os.listdir(now_filepath) if 'pkl' in x][0]))
oof = oof_test_data["oof"]
preds = oof_test_data["test"]
for i in oof_test_data:
    print(i)
stacknp = np.concatenate([oof,preds],axis=0)
print(stacknp.shape)
stackpd = pd.DataFrame(data=stacknp,columns=["0729_generator_one_sixth_orig_mixup_086686_class"+str(i)for i in range(19)])
stackpd["fragment_id"] = df["fragment_id"]
df = df.merge(stackpd,how='left',on='fragment_id')
C4 = ["0729_generator_one_sixth_orig_mixup_086686_class"+str(i)for i in range(19)]

####################################################################
now_filepath=data_path + "0728_08648_online792"
oof_test_data = joblib.load(os.path.join(now_filepath,[x for x in os.listdir(now_filepath) if 'pkl' in x][0]))
oof = oof_test_data["oof"]
preds = oof_test_data["test"]
stacknp = np.concatenate([oof,preds],axis=0)
print(stacknp.shape)
stackpd = pd.DataFrame(data=stacknp,columns=["0728_08648_online792_class"+str(i)for i in range(19)])
stackpd["fragment_id"] = df["fragment_id"]
df = df.merge(stackpd,how='left',on='fragment_id')
# # oof_test_data
C1 = ["0728_08648_online792_class"+str(i)for i in range(19)]

####################################################################
now_filepath=data_path + "0725_conv2_2_net_weight_comm_0.85568"
oof_test_data = joblib.load(os.path.join(now_filepath,[x for x in os.listdir(now_filepath) if 'pkl' in x][0]))
oof = oof_test_data["oof"]
preds = oof_test_data["test"]
stacknp = np.concatenate([oof,preds],axis=0)
print(stacknp.shape)
stackpd = pd.DataFrame(data=stacknp,columns=["0725_conv2_2_net_weight_comm_0_class"+str(i)for i in range(19)])
stackpd["fragment_id"] = df["fragment_id"]
df = df.merge(stackpd,how='left',on='fragment_id')
C6 = ["0725_conv2_2_net_weight_comm_0_class"+str(i)for i in range(19)]


now_filepath=data_path + "0721_conv2_2_net_oof_comm_nn0.84665"
oof_test_data = joblib.load(os.path.join(now_filepath,[x for x in os.listdir(now_filepath) if 'pkl' in x][0]))
oof = oof_test_data["oof"]
preds = oof_test_data["test"]
stacknp = np.concatenate([oof,preds],axis=0)
print(stacknp.shape)
stackpd = pd.DataFrame(data=stacknp,columns=["0721_conv2_2_net_oof_comm_nn0_class"+str(i)for i in range(19)])
stackpd["fragment_id"] = df["fragment_id"]
df = df.merge(stackpd,how='left',on='fragment_id')
C7 = ["0721_conv2_2_net_oof_comm_nn0_class"+str(i)for i in range(19)]


now_filepath=data_path + "spetron_cnn"
oof_test_data = joblib.load(os.path.join(now_filepath,[x for x in os.listdir(now_filepath) if 'pkl' in x][0]))
oof = oof_test_data["oof"]
preds = oof_test_data["test"]
stacknp = np.concatenate([oof,preds],axis=0)
print(stacknp.shape)
stackpd = pd.DataFrame(data=stacknp,columns=["spetron0728_conv2_2_net_multiloss_0_class"+str(i)for i in range(19)])
stackpd["fragment_id"] = df["fragment_id"]
df = df.merge(stackpd,how='left',on='fragment_id')

now_filepath=data_path + "multi_lstm"
oof_test_data = joblib.load(os.path.join(now_filepath,[x for x in os.listdir(now_filepath) if 'pkl' in x][0]))
oof = oof_test_data["oof"]
preds = oof_test_data["test"]
stacknp = np.concatenate([oof,preds],axis=0)
stackpd = pd.DataFrame(data=stacknp,columns=["lstm_mutiloss_4sub_bs32_class"+str(i)for i in range(19)])
stackpd["fragment_id"] = df["fragment_id"]
df = df.merge(stackpd,how='left',on='fragment_id')


# In[8]:


train_df = df[df[label].isna()==False].reset_index(drop=True)
test_df = df[df[label].isna()==True].reset_index(drop=True)

drop_feat = ["istrain"] 
used_feat = [f for f in train_df.columns if f not in (['fragment_id', label] + drop_feat)]
print(used_feat)
df["istrain"] = (df[label].isna()==False).astype(np.int8)
result_dict = get_feature_report_by_covariate_shift_test(df,
                                               features=used_feat,
                                               all_cate_feas=[],
                                               params=None,
                                               nfold=3,
                                               y2test='istrain',
                                               train_all_feas=False)

result_df = result_dict["df_fea_auc"]
result_df


# In[9]:


result_df = result_dict["df_fea_auc"]
drop_bylgb = list(result_df[result_df["val_auc"] > 0.7]["feat"])
result_df
print('len of drop',len(drop_bylgb))


# In[10]:




train_df = df[df[label].isna()==False].reset_index(drop=True)
test_df = df[df[label].isna()==True].reset_index(drop=True)

drop_feat = ['acc_median','acc_y_min','acc_y_max', 'acc_std','acc_y_mean','acc_min', 'acc_x_mean']# lgb_auc 筛选的
drop_feat = ["istrain"] + drop_bylgb
used_feat = [f for f in train_df.columns if f not in (['fragment_id', label] + drop_feat)and 'id'not in f]
print(len(used_feat))
print(used_feat)

train_x = train_df[used_feat]
train_y = train_df[label]
test_x = test_df[used_feat]


# In[11]:


scores = []
imp = pd.DataFrame()
imp['feat'] = used_feat
from sklearn.linear_model import RidgeClassifier,LogisticRegression
params = {
    'learning_rate': 0.03,
    'metric': 'multi_error',
    'objective': 'multiclass',
    'num_class': 19,
    'feature_fraction': 0.80,
    'bagging_fraction': 0.75,
    'bagging_freq': 2,
    'n_jobs': -1,
#     'max_depth': 6,
    'num_leaves': 64,
    'lambda_l1': 0.6,
    'lambda_l2': 0.6,
}


oof_train = np.zeros((len(train_x), 19))
preds = np.zeros((len(test_x), 19))
folds = 5
# seeds = [44, 2020, 527, 1527,404,721]
seeds = [1111, 1024, 1314, 6666, 9999, 6969]
for seed in seeds:
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (trn_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
        x_trn, y_trn, x_val, y_val = train_x.iloc[trn_idx], train_y.iloc[trn_idx], train_x.iloc[val_idx], train_y.iloc[val_idx]
        train_set = lgb.Dataset(x_trn, y_trn)
        val_set = lgb.Dataset(x_val, y_val)
        print(str(fold)*10)
#         
        model = lgb.train(params, train_set, num_boost_round=100,
                          valid_sets=(train_set, val_set), early_stopping_rounds=50,
                          verbose_eval=50)
        oof_train[val_idx] += model.predict(x_val) / len(seeds)
        preds += model.predict(test_x) / folds / len(seeds)
        scores.append(model.best_score['valid_1']['multi_error'])
        
        imp['gain' + str(fold + 1)] = model.feature_importance(importance_type='gain')
        imp['split' + str(fold + 1)] = model.feature_importance(importance_type='split')
        del x_trn, y_trn, x_val, y_val, model, train_set, val_set
        gc.collect()
imp['gain'] = imp[[f for f in imp.columns if 'gain' in f]].sum(axis=1)/folds
imp['split'] = imp[[f for f in imp.columns if 'split' in f]].sum(axis=1)
imp = imp.sort_values(by=['gain'], ascending=False)
# imp[['feat', 'gain', 'split']]
imp = imp.sort_values(by=['split'], ascending=False)
imp = imp.merge(result_df,on='feat',how='left')
imp[['feat', 'gain', 'split',"train_auc","val_auc"]]


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def acc_combo(y, y_pred):
    # 数值ID与行为编码的对应关系
    mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3', 
        4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5', 
        8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6', 
        12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6', 
        16: 'C_2', 17: 'C_5', 18: 'C_6'}
    # 将行为ID转为编码
    code_y, code_y_pred = mapping[y], mapping[y_pred]
    if code_y == code_y_pred: #编码完全相同得分1.0
        return 1.0
    elif code_y.split("_")[0] == code_y_pred.split("_")[0]: #编码仅字母部分相同得分1.0/7
        return 1.0/7
    elif code_y.split("_")[1] == code_y_pred.split("_")[1]: #编码仅数字部分相同得分1.0/3
        return 1.0/3
    else:
        return 0.0

labels = np.argmax(preds, axis=1)
oof_y = np.argmax(oof_train, axis=1)
print(round(accuracy_score(train_y, oof_y), 5))
scores = sum(acc_combo(y_true, y_pred) for y_true, y_pred in zip(train_y, oof_y)) / oof_y.shape[0]
print(round(scores, 5))
data_path2 = 'data/'
sub = pd.read_csv(data_path2+'提交结果示例.csv')
sub['behavior_id'] = labels

sub.to_csv('sub/0806allin%.5f.csv' % scores, index=False)
print('file has been saved!!!!!!!!!!!!!!!!!!!!!!!!!')
sub.info()


# In[ ]:




