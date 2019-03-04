import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv', dtype={"Age": np.float64})
test = pd.read_csv('test.csv', dtype={"Age": np.float64})
PassengerId = test['PassengerId']
all_data = pd.concat([train, test], ignore_index=True)

# 数据可视化

train.info()
test.info()
# print("-" * 40)
# test.info()

train['Survived'].value_counts().plot.pie(autopct='%1.2f%%')
plt.show()

# 1) Sex Feature：女性幸存率远高于男性
print(train.groupby(['Sex', 'Survived'])['Survived'].count())
sns.barplot(x="Sex", y="Survived", data=train, palette='Set3')
print("Percentage of females who survived:%.2f" % (
        train["Survived"][train["Sex"] == 'female'].value_counts(normalize=True)[1] * 100))
print("Percentage of males who survived:%.2f" % (
        train["Survived"][train["Sex"] == 'male'].value_counts(normalize=True)[1] * 100))
plt.show()

# 2) Pclass Feature：乘客社会等级越高，幸存率越高
print(train.groupby(['Sex', 'Pclass', 'Survived'])['Survived'].count())
sns.barplot(x="Pclass", y="Survived", data=train, palette='Set3')
print("Percentage of Pclass = 1 who survived:%.2f" % (
        train["Survived"][train["Pclass"] == 1].value_counts(normalize=True)[1] * 100))
print("Percentage of Pclass = 2 who survived:%.2f" % (
        train["Survived"][train["Pclass"] == 2].value_counts(normalize=True)[1] * 100))
print("Percentage of Pclass = 3 who survived:%.2f" % (
        train["Survived"][train["Pclass"] == 3].value_counts(normalize=True)[1] * 100))
plt.show()

sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train, palette="Set2")
plt.show()

# 3) SibSp Feature：配偶及兄弟姐妹数适中的乘客幸存率更高
sns.barplot(x="SibSp", y="Survived", data=train, palette='Set3')
plt.show()

# 4) Parch Feature：父母与子女数适中的乘客幸存率更高
sns.barplot(x="Parch", y="Survived", data=train, palette='Set3')

# 5) Age Feature：未成年人幸存率高于成年人
facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(18, 8))
sns.violinplot("Pclass", "Age", hue="Survived", data=train, palette='Set3', split=True, ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0, 110, 10))

sns.violinplot("Sex", "Age", hue="Survived", data=train, palette='Set3', split=True, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0, 110, 10))

plt.show()

# 6) Fare Feature：支出船票费越高幸存率越高
facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Fare', shade=True)
facet.set(xlim=(0, 200))
facet.add_legend()
plt.show()

# 7) Title Feature(New)：不同称呼的乘客幸存率不同
all_data['Title'] = all_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

# plt.hist(x=[all_data[all_data['Survived'] == 1]['Title'], all_data[all_data['Survived'] == 0]['Title']],
#          stacked=True, color=['g', 'r'], label=['Survived', 'Dead'])
# plt.title('Title Histogram by Survival')
# plt.xlabel('Title')
# plt.ylabel('# of Passengers')
# plt.legend()
# plt.show()

Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
# all_data['Title'] = all_data['Title'].map(Title_Dict)
sns.barplot(x="Title", y="Survived", data=all_data, palette='Set3')
plt.show()

# 8) FamilyLabel Feature(New)：家庭人数为2到4的乘客幸存率较高
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
sns.barplot(x="FamilySize", y="Survived", data=all_data, palette='Set3')
plt.show()


def Fam_label(s):
    """ 按生存率把FamilySize分为三类，构成FamilyLabel特征。"""
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif s > 7:
        return 0


all_data['FamilyLabel'] = all_data['FamilySize'].apply(Fam_label)
sns.barplot(x="FamilyLabel", y="Survived", data=all_data, palette='Set3')
plt.show()

# 9) Deck Feature(New)：不同甲板的乘客幸存率不同
all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')
all_data['Deck'] = all_data['Cabin'].str.get(0)
sns.barplot(x="Deck", y="Survived", data=all_data, palette='Set3')
plt.show()

# 10) TicketGroup Feature(New)：与2至4人共票号的乘客幸存率较高
Ticket_Count = dict(all_data['Ticket'].value_counts())
all_data['TicketGroup'] = all_data['Ticket'].apply(lambda x: Ticket_Count[x])
sns.barplot(x='TicketGroup', y='Survived', data=all_data, palette='Set3')
plt.show()


def Ticket_Label(s):
    """按生存率把TicketGroup分为三类。"""
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif s > 8:
        return 0


all_data['TicketGroup'] = all_data['TicketGroup'].apply(Ticket_Label)
sns.barplot(x='TicketGroup', y='Survived', data=all_data, palette='Set3')
plt.show()

# 11) 港口和存活与否的关系 Embarked
sns.barplot(x="Embarked", y="Survived", data=train, palette='Set3')
plt.show()
# sns.countplot('Embarked', hue='Survived', data=train, palette='Set2')
# plt.title('Embarked and Survived')
# plt.show()
# sns.factorplot('Embarked', 'Survived', data=train, size=3, aspect=2)
# plt.title('Embarked and Survived rate')
# plt.show()

# print(len(all_data[all_data['Fare'].isnull()]))
all_data = all_data[
    ['Survived', 'Pclass', 'Sex', 'Age', "SibSp", "Parch", 'Fare', 'Embarked', 'Title', 'FamilyLabel', 'Deck',
     'TicketGroup']]
all_data = pd.get_dummies(all_data)
print(np.array(all_data.drop(['Survived'], 1).columns))
sns.heatmap(all_data[["Age", "Pclass", 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Ms',
                      'Sex_female', 'Sex_male', "SibSp", "Parch"]].corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm")
plt.show()
# """
sns.heatmap(all_data[["Age", 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E',
                      'Deck_F', 'Deck_G', 'Deck_T',
                      'Deck_U']].corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm")
plt.show()
# """
"""
# 数据清洗

# 1) 缺失值填充
age_df = all_data[['Age', 'Pclass', 'Sex', 'Title']]
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

# 2) 同组识别
all_data['Surname'] = all_data['Name'].apply(lambda x: x.split(',')[0].strip())
Surname_Count = dict(all_data['Surname'].value_counts())
all_data['FamilyGroup'] = all_data['Surname'].apply(lambda x: Surname_Count[x])
Female_Child_Group = all_data.loc[
    (all_data['FamilyGroup'] >= 2) & ((all_data['Age'] <= 12) | (all_data['Sex'] == 'female'))]
Male_Adult_Group = all_data.loc[(all_data['FamilyGroup'] >= 2) & (all_data['Age'] > 12) & (all_data['Sex'] == 'male')]

Female_Child = pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())
Female_Child.columns = ['GroupCount']
print(Female_Child)
sns.barplot(x=Female_Child.index, y=Female_Child["GroupCount"], palette='Set3').set_xlabel('AverageSurvived')
plt.show()

Male_Adult = pd.DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())
Male_Adult.columns = ['GroupCount']
print(Male_Adult)
sns.barplot(x=Male_Adult.index, y=Male_Adult['GroupCount'], palette='Set3').set_xlabel('AverageSurvived')
plt.show()

# 反常惩罚
Female_Child_Group = Female_Child_Group.groupby('Surname')['Survived'].mean()
Dead_List = set(Female_Child_Group[Female_Child_Group.apply(lambda x: x == 0)].index)
print(Dead_List)
Male_Adult_List = Male_Adult_Group.groupby('Surname')['Survived'].mean()
Survived_List = set(Male_Adult_List[Male_Adult_List.apply(lambda x: x == 1)].index)
print(Survived_List)
train = all_data.loc[all_data['Survived'].notnull()]
test = all_data.loc[all_data['Survived'].isnull()]
test.loc[(test['Surname'].apply(lambda x: x in Dead_List)), 'Sex'] = 'male'
test.loc[(test['Surname'].apply(lambda x: x in Dead_List)), 'Age'] = 60
test.loc[(test['Surname'].apply(lambda x: x in Dead_List)), 'Title'] = 'Mr'
test.loc[(test['Surname'].apply(lambda x: x in Survived_List)), 'Sex'] = 'female'
test.loc[(test['Surname'].apply(lambda x: x in Survived_List)), 'Age'] = 5
test.loc[(test['Surname'].apply(lambda x: x in Survived_List)), 'Title'] = 'Miss'

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
pipe = Pipeline([('select', SelectKBest(k=20)),
                 ('classify', RandomForestClassifier(random_state=10, max_features='sqrt'))])

param_test = {'classify__n_estimators': list(range(20, 50, 2)),
              'classify__max_depth': list(range(3, 60, 3))}
gsearch = GridSearchCV(estimator=pipe, param_grid=param_test, scoring='roc_auc', cv=10)
gsearch.fit(X, y)
print(gsearch.best_params_, gsearch.best_score_)

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
"""
