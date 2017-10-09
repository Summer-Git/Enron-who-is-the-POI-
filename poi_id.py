#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pprint
import matplotlib.pyplot as plt  
from sklearn.preprocessing import MinMaxScaler
import numpy
from sklearn.metrics import precision_score, recall_score, f1_score  
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
"""1. 特征选择"""
full_features_list = ['poi','salary','bonus',
                 'deferral_payments','deferred_income','director_fees',
                 'exercised_stock_options','expenses','from_messages','from_poi_to_this_person',
                 'from_this_person_to_poi','loan_advances','long_term_incentive',
                 'restricted_stock','restricted_stock_deferred',
                 'shared_receipt_with_poi','to_messages','total_payments','total_stock_value']

NaN_list = {'poi':0,'salary':0,'bonus':0,
                 'deferral_payments':0,'deferred_income':0,'director_fees':0,
                 'exercised_stock_options':0,'expenses':0,'from_messages':0,'from_poi_to_this_person':0,
                 'from_this_person_to_poi':0,'loan_advances':0,'long_term_incentive':0,
                 'restricted_stock':0,'restricted_stock_deferred':0,
                 'shared_receipt_with_poi':0,'to_messages':0,'total_payments':0,'total_stock_value':0}

features_deleted_high_NaN = ['poi','salary','bonus',
                 'exercised_stock_options','expenses','from_messages','from_poi_to_this_person',
                 'from_this_person_to_poi','restricted_stock',
                 'shared_receipt_with_poi','to_messages','total_payments','total_stock_value']

"""1.1 对于决策树，通过clf.feature_importances_查看特征的重要度，并且删除不重要的特征，最终得到以下特征"""
features_list = ['poi','bonus',
                 'exercised_stock_options','to_poi_ratio',
                 'shared_receipt_with_poi']
# You will need to use more features， 
"""1.2 对于逻辑回归，通过clf.coef删除系数为0的特征，最终得到以下特征"""
#features_list = ['poi','bonus',
#                 'exercised_stock_options','expenses','to_poi_ratio']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#pprint.pprint(data_dict['FREVERT MARK A'])

"""2. 数据DEA"""
"""2.1 查看数据点的总数和POI数量"""
print "数据总个数:",len(data_dict)
i = 0
for key in data_dict: 
    if data_dict[key]['poi'] == True:
        i += 1
print "POI个数:",i

"""2.2 查看full_feature的NaN数量，去掉NaN过多的特征"""
for key in data_dict:
    for f in full_features_list:
        if data_dict[key][f] == 'NaN':
            NaN_list[f] = NaN_list[f] + 1
#print sorted(NaN_list.items(),key = lambda item:item[1])
    """
    ('loan_advances', 142),('director_fees', 129),('restricted_stock_deferred', 128),('deferral_payments', 107),
    ('deferred_income', 97),('long_term_incentive', 80)
    这些Features缺省值太多，直接删除。
    """
### Task 2: Remove outliers
Out_check= []
for key in data_dict:
    if data_dict[key]["salary"] != 'NaN':
        Out_check.append(data_dict[key]["salary"])
plt.boxplot((Out_check))
plt.show()
"""2.3 查找异常点"""
"""通过箱线图发现salary有一个异常点26704229"""
for key in data_dict:
    if data_dict[key]["salary"] == 26704229:
        print "异常值",key
"""key为Total,检查后删除该点"""       
del data_dict["TOTAL"]
#del data_dict_amplified["TOTAL"]

"""同样方法检查其余Feature,认为异常值应与嫌疑人有关，所以不进行删除"""
### Task 3: Create new feature(s)

"""3. 塑造新的Feature"""
for key in data_dict:
    if data_dict[key]['to_messages'] != 'NaN' and data_dict[key]['from_poi_to_this_person'] != 'NaN':
        data_dict[key]["from_poi_ratio"] = data_dict[key]['from_poi_to_this_person']/float(data_dict[key]['to_messages'])
    else:
        data_dict[key]["from_poi_ratio"] = 'NaN'
for key in data_dict:
    if data_dict[key]['from_messages'] != 'NaN' and data_dict[key]['from_this_person_to_poi'] != 'NaN':
        data_dict[key]["to_poi_ratio"] = data_dict[key]['from_this_person_to_poi']/float(data_dict[key]['from_messages'])
    else:
        data_dict[key]["to_poi_ratio"] = 'NaN'
        
my_dataset = data_dict
#train_data = data_dict_amplified

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)


"""4. 部署特征缩放"""
scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)

labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

"""5. Modelling"""
"""5.1 采用逻辑回归"""
#from sklearn import linear_model
#clf = linear_model.LogisticRegression(penalty='l1')
"""发现线性模型在此处应该不合适，总是得到较高的Precesion值,和较低的Recall值"""

"""5.2 采用决策树"""
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=2)
"""发现使用决策树Precesion值和Recall值都达到了0.3 ,且经过调试min_samples_split=18时，效果较好"""

"""5.3 尝试随机森林"""
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators = 100)
"""效果不如决策树"""



        

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

"""##gridcv寻找最优参数"""
#param_grid = {
#         'min_samples_split': [2,3,4,5,6,
#                               7,8,9,10]
#          }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
#clf = GridSearchCV( tree.DecisionTreeClassifier(), param_grid)
#clf = clf.fit(features_train, labels_train)
#print clf
""""""
"""6.使用交叉验证"""
cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
for train_idx, test_idx in cv: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )
    
    ### fit the classifier using training set, and test on test set
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    for prediction, truth in zip(predictions, labels_test):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        elif prediction == 1 and truth == 1:
            true_positives += 1
        else:
            print "Warning: Found a predicted label not == 0 or 1."
            print "All predictions should take value 0 or 1."
            print "Evaluating performance for processed predictions:"
            break
try:
    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    accuracy = 1.0*(true_positives + true_negatives)/total_predictions
    precision = 1.0*true_positives/(true_positives+false_positives)
    recall = 1.0*true_positives/(true_positives+false_negatives)
    f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
    f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
    print clf
    print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
    print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
    print ""
except:
    print "Got a divide by zero when trying out:", clf
    print "Precision or recall may be undefined due to a lack of true positive predicitons."
    
"""7. 特性选择：逻辑回归系数与Feature重要度"""
#for c in clf.coef_[0]:
#    print c
print "特征重要度：",clf.feature_importances_

"""sklearn.metrics方法"""
#clf.fit(features_train,labels_train)
#predictions = clf.predict(features_test)
#p = precision_score(labels_test, predictions, average='binary')  
#print "模型Precision：",p
#r = recall_score(labels_test, predictions, average='binary')  
#print "模型Recall：",r
#f1score = f1_score(labels_test, predictions, average='binary')  
#print "模型F1：",f1score

"""可视化决策树"""
#import graphviz 
#dot_data = tree.export_graphviz(clf, out_file=None) 
#graph = graphviz.Source(dot_data) 
#graph.render("POI")

#for c in clf.coef_[0]:


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)






