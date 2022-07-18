#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn.pipeline import Pipeline


# In[2]:


df = pd.read_csv('loan_data_2007_2014.csv', index_col=None)
# del_cols = ['id',
#             'member_id',
#             'Unnamed: 0',
#             'annual_inc_joint',
#             'dti_joint',
#             'verification_status_joint',
#             'open_acc_6m',
#             'open_il_6m',
#             'open_il_12m','open_il_24m',
#             'mths_since_rcnt_il',
#             'total_bal_il',
#             'il_util',
#             'open_rv_12m',
#             'open_rv_24m',
#             'max_bal_bc',
#             'all_util',
#             'inq_fi',
#             'total_cu_tl',
#             'inq_last_12m']
# df = data.drop(columns=del_cols, axis=1, inplace=True)
pd.set_option('display.max_columns', None)
df.head()


# In[3]:


df.info()


# In[4]:


null_col = df.isnull().sum().sort_values(ascending = False)
null_col = null_col[null_col.values >(0.35*len(df))]


# In[5]:


null_col


# In[6]:


dropped = list(null_col.index.values) #Making list of column names having null values greater than 35%
df.drop(labels = dropped,axis=1,inplace = True) #Droping those columns
# drop unused columns
df.drop(columns = ['Unnamed: 0','id', 'member_id', 'sub_grade', 'emp_title', 'url', 'title', 'zip_code','policy_code',
                          'recoveries', 'collection_recovery_fee', 'total_rec_prncp', 'total_rec_late_fee'], inplace = True)


# In[7]:


df.info()


# # Cleaning

# In[8]:


df.describe()


# In[9]:


df['loan_status'].value_counts(normalize=True)


# In[10]:


# make the following value below is 0 (bad status), and the other are 1 (good)

df['target'] = np.where(df.loc[:, 'loan_status'].isin(['Charged Off',
                                                       'Default',
                                                       'Late (31-120 days)',
                                                       'Does not meet the credit policy. Status:Charged Off']),0, 1)
df.drop(columns = ['loan_status'], inplace = True)


# In[11]:


df


# # Splitting dataset into train and test

# In[12]:


X = df.drop('target', axis = 1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 42, stratify = y)


# In[13]:


X_train

# fill numerical columns nan value to mean

X_train_num = X_train.select_dtypes(include = 'number').copy()
X_train_num.fillna(X_train_num.mean(), inplace = True)X_train.select_dtypes(include = 'number').fillna(X_train.mean(), inplace = True)numeric_columns = X_train.select_dtypes(include='number').columns
X_train[numeric_columns] = X_train[numeric_columns].fillna(X_train.mean())
# # Handling categorical

# In[14]:


list(X_train.select_dtypes(include=['object']))


# ### term

# In[15]:


df.term.unique()


# In[16]:


def term_handler(df):
    df['term'] = pd.to_numeric(df['term'].str.replace(r'\D', ''))


# In[17]:


term_handler(X_train)


# In[18]:


X_train.term.unique()


# ### emp_length

# In[19]:


X_train.emp_length.unique()


# In[20]:


# get any numerical value, and fill nan to 0
def emp_length_handler(df):
    df['emp_length'] = df['emp_length'].str.replace(r'\D', '')
    df['emp_length'].fillna(value = 0, inplace = True)
    df['emp_length'] = pd.to_numeric(df['emp_length'])


# In[21]:


emp_length_handler(X_train)


# In[22]:


X_train.emp_length.unique()


# ### date columns handler

# In[23]:


# convert date columns to datetime format and create a new column as a difference between today and the respective date
def date_columnhandler(df, column):
    # store current month
    today_date = pd.to_datetime('2022-06-04')
    # convert to datetime format
    df[column] = pd.to_datetime(df[column], format = "%b-%y")
    # calculate the difference in months and add to a new column
    df['m_since_' + column] = round(pd.to_numeric((today_date - df[column]) / np.timedelta64(1, 'M')))
    # make any resulting -ve values to be equal to the max date
    df['m_since_' + column] = df['m_since_' + column].apply(lambda x: df['m_since_' + column].max() if x < 0 else x)
    # drop the original date column
    df.drop(columns = [column], inplace = True)

# apply to X_train
date_columnhandler(X_train, 'earliest_cr_line')
date_columnhandler(X_train, 'issue_d')
date_columnhandler(X_train, 'last_pymnt_d')
date_columnhandler(X_train, 'last_credit_pull_d')


# In[24]:


print(X_train['m_since_earliest_cr_line'].describe())
print(X_train['m_since_issue_d'].describe())
print(X_train['m_since_last_pymnt_d'].describe())
print(X_train['m_since_last_credit_pull_d'].describe())


# In[25]:


heatmap = X_train.corr()
plt.figure(figsize=(10,10))
sns.heatmap(heatmap)


# In[26]:


cat_cols = list(X_train.select_dtypes(include=['object']))


# In[27]:


for i in cat_cols:
    print(X_train[i].unique())


# In[28]:


for i in cat_cols:
    print(X_test[i].unique())


# In[29]:


X_train.home_ownership.value_counts()


# In[30]:


X_test.home_ownership.value_counts()


# In[31]:


# drop identifically unused feature
iden_drop = ['addr_state','application_type']

def col_to_drop(df, columns_list):
    df.drop(columns = columns_list, inplace = True)

# apply to X_train
col_to_drop(X_train, iden_drop)


# In[32]:


cat_cols

# function to create dummy variables
def dummy_creation(df, columns_list):
    df_dummies = []
    for col in columns_list:
        df_dummies.append(pd.get_dummies(df[col], prefix = col, prefix_sep = ':'))
    df_dummies = pd.concat(df_dummies, axis = 1)
    df = pd.concat([df, df_dummies], axis = 1)
    return df

# apply to our final four categorical variables
X_train = dummy_creation(X_train, ['grade', 'home_ownership', 'verification_status', 'purpose', 'pymnt_plan','initial_list_status'])
# In[33]:


X_train = pd.get_dummies(X_train, prefix=['grade',
                                          'home_ownership',
                                          'verification_status',
                                          'purpose',
                                          'pymnt_plan',
                                          'initial_list_status'], prefix_sep=':')


# ## update test set

# In[34]:


emp_length_handler(X_test)
term_handler(X_test)
date_columnhandler(X_test, 'earliest_cr_line')
date_columnhandler(X_test, 'issue_d')
date_columnhandler(X_test, 'last_pymnt_d')
date_columnhandler(X_test, 'last_credit_pull_d')
col_to_drop(X_test, iden_drop)
X_test = pd.get_dummies(X_test, prefix=['grade',
                                         'home_ownership',
                                         'verification_status',
                                         'purpose',
                                         'pymnt_plan',
                                         'initial_list_status'], prefix_sep=':')
# reindex the dummied test set variables to make sure all the feature columns in the train set are also available in the test set
X_test = X_test.reindex(labels=X_train.columns, axis=1, fill_value=0)


# In[35]:


X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)
# numeric_columns = X_train.select_dtypes(include='number').columns
# X_train[numeric_columns] = X_train[numeric_columns].fillna(X_train.mean())


# In[36]:


X_train.info()


# # Prediction

# In[37]:


logreg = LogisticRegression(max_iter=100, class_weight = 'balanced')
logreg.fit(X_train, y_train)


# In[38]:


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
scores = cross_val_score(logreg, X_train, y_train, scoring = 'roc_auc', cv = cv)


# In[51]:


# draw a PR curve
# calculate the no skill line as the proportion of the positive class
no_skill = len(y_test[y_test == 1]) / len(y)
# plot the no skill precision-recall curve
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

# calculate inputs for the PR curve
precision, recall, thresholds = precision_recall_curve(y_test, lr_probs)
# plot PR curve
plt.plot(recall, precision, marker='.', label='Logistic')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.title('PR curve')


# In[43]:


ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = logreg.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()


# In[47]:


from sklearn.metrics import accuracy_score
pred = logreg.predict(X_test)
accuracy_score(y_test ,pred)


# In[53]:


print(confusion_matrix(y_test,pred))


# In[49]:


roc_auc_score(y_test ,lr_probs)


# In[52]:


X_train.head()


# In[54]:


pred_prob = logreg.predict_proba(X_test)
pred_prob


# In[60]:


df_pred_prob = pd.DataFrame(pred_prob, columns = ['prob_0', 'prob_1'])
df_pred_target = pd.DataFrame(logreg.predict(X_test), columns = ['predicted_TARGET'])
df_test_dataset = pd.DataFrame(y_test.values, columns= ['Actual Outcome'])

dfx=pd.concat([df_test_dataset, df_pred_prob, df_pred_target], axis=1)
dfx


# In[59]:


y_test.values


# In[61]:


dfx.info()


# In[62]:


pd.set_option('display.float_format', lambda x: '%.5f' % x)


# In[63]:


dfx


# In[73]:


dfx['Decile_rank'] = pd.qcut(dfx['prob_0'], 10, labels = False)


# In[74]:


dfx

dfx.sort_values(by = 'prob_1', ascending=False, inplace=True)
# In[79]:


dfx.to_excel('dfx.xlsx')


# In[ ]:




