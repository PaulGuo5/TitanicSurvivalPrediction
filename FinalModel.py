import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline

warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv', dtype={"Age": np.float64})
test = pd.read_csv('test.csv', dtype={"Age": np.float64})
PassengerId = test['PassengerId']
all_data = pd.concat([train, test], ignore_index=True)

all_data['Title'] = all_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
Title_Dict = {}

# Title_Dict.update(dict.fromkeys(['Mrs'], 'Mrs'))
# Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
# Title_Dict.update(dict.fromkeys(['Capt', 'Rev', 'Don', 'Dona', 'Jonkheer'], 'D'))
# Title_Dict.update(dict.fromkeys(['Miss', 'Master', 'Dr', 'Major', 'Col'], 'Maybe'))
# Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Lady', 'Sir', 'Mlle', 'the Countess'], 'Survived'))

Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
all_data['Title'] = all_data['Title'].map(Title_Dict)

# sns.barplot(x="Title", y="Survived", data=all_data, palette='Set3')
# plt.show()

# sns.barplot(x="SibSp", y="Survived", data=all_data, palette='Set3')
# plt.show()
# sns.barplot(x="Parch", y="Survived", data=all_data, palette='Set3')
# plt.show()

all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1


def Fam_label(s):
    """ 按生存率把FamilySize分为三类，构成FamilyLabel特征。"""
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif s > 7:
        return 0


all_data['FamilyLabel'] = all_data['FamilySize'].apply(Fam_label)

all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')
all_data['Deck'] = all_data['Cabin'].str.get(0)

Ticket_Count = dict(all_data['Ticket'].value_counts())
all_data['TicketGroup'] = all_data['Ticket'].apply(lambda x: Ticket_Count[x])


def Ticket_Label(s):
    """按生存率把TicketGroup分为三类。"""
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif s > 8:
        return 0


all_data['TicketGroup'] = all_data['TicketGroup'].apply(Ticket_Label)

# 数据清洗

# 1) 缺失值填充
age_df = all_data[['Age', 'Pclass', 'Title', 'Deck']]
age_df = pd.get_dummies(age_df)
known_age = age_df[age_df.Age.notnull()].as_matrix()
unknown_age = age_df[age_df.Age.isnull()].as_matrix()
y = known_age[:, 0]
X = known_age[:, 1:]
rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
rfr.fit(X, y)
predictedAges = rfr.predict(unknown_age[:, 1::])
all_data.loc[(all_data.Age.isnull()), 'Age'] = predictedAges

all_data['Embarked'] = all_data['Embarked'].fillna('C')

fare = all_data[(all_data['Embarked'] == "S") & (all_data['Pclass'] == 3)].Fare.median()
all_data['Fare'] = all_data['Fare'].fillna(fare)

# 3) 特征转换
# all_data = pd.concat([train, test])
all_data = all_data[
    ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilyLabel', 'Deck', 'TicketGroup']]
all_data = pd.get_dummies(all_data)
train = all_data[all_data['Survived'].notnull()]
test = all_data[all_data['Survived'].isnull()].drop('Survived', axis=1)
X = train.as_matrix()[:, 1:]
y = train.as_matrix()[:, 0]

# 建模和优化

# 1) 参数优化
"""
pipe = Pipeline([('select', SelectKBest(k=20)),
                 ('classify', RandomForestClassifier(random_state=10, max_features='sqrt'))])

param_test = {'classify__n_estimators': list(range(20, 60, 1)),
              'classify__max_depth': list(range(3, 30, 1))}
gsearch = GridSearchCV(estimator=pipe, param_grid=param_test, scoring='roc_auc', cv=10)
gsearch.fit(X, y)
print(gsearch.best_params_, gsearch.best_score_)
print(gsearch.best_params_, gsearch.best_estimator_)
# """

"""
bestSol = 0
for k in range(15, 25):
    for random_state in range(5, 25):
        for estimators in range(20, 100, 2):
            for depth in range(3, 30, 3):
                select = SelectKBest(k=k)
                clf = RandomForestClassifier(random_state=random_state,
                                             warm_start=True,
                                             n_estimators=estimators,
                                             max_depth=depth,
                                             max_features='sqrt')
                pipeline = make_pipeline(select, clf)
                pipeline.fit(X, y)
                cv_score = cross_val_score(pipeline, X, y, cv=10)
                # print(k, random_state, estimators, depth)
                # print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))
                if np.mean(cv_score) > bestSol:
                    bestSol = np.mean(cv_score)
                    bestK = k
                    bestRandom = random_state
                    bestEstimator = estimators
                    bestDepth = depth
                    bestScore = np.mean(cv_score)
                    print(bestK, bestRandom, bestEstimator, bestDepth, bestScore)

print("=" * 64)
print(bestK, bestRandom, bestEstimator, bestDepth, bestScore)
# """
# """
# 2) 训练模型
select = SelectKBest(k=20)
clf = RandomForestClassifier(random_state=10, warm_start=True,
                             n_estimators=26,
                             max_depth=6,
                             max_features='sqrt')
pipeline = make_pipeline(select, clf)
pipeline.fit(X, y)

# 3) 交叉验证
cv_score = cross_val_score(pipeline, X, y, cv=10)
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))

# 预测
predictions = pipeline.predict(test)
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
submission.to_csv("submission.csv", index=False)
# print(submission.head(n=20))
# """
