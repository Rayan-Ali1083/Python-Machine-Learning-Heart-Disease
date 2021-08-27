# Classification Report
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
import seaborn as sns

# reading data
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

X = df.iloc[:, :12]
y = df.iloc[:, -1]

# Splitting data for train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# visualization

# listing ages

age_list = (df.iloc[:, 0]).tolist()
under_50 = len(df.loc[df.age <= 50])
bet_50_60 = len(df.loc[(df.age > 50) & (df.age <= 60)])
bet_60_70 = len(df.loc[(df.age > 60) & (df.age <= 70)])
bet_70_80 = len(df.loc[(df.age > 70) & (df.age <= 80)])
bet_80_90 = len(df.loc[(df.age > 80) & (df.age <= 90)])
above_90 = len(df.loc[df.age >= 90])

ages = [under_50, bet_50_60, bet_60_70, bet_70_80, bet_80_90, above_90]
labels = ['<50', "50<AGE<=60", "60<AGE<=70", "70<AGE<=80", "80<AGE<=90", "AGE>90"]

# pie chart for age groups

plt.title('Age Group Visualization')
plt.pie(x=ages, labels=labels, autopct='%.2f%%', colors=['orange', 'yellow', 'purple', 'green', 'red', 'pink'])
plt.show()

# Age group deaths

under_50 = len(df.loc[(df.age <= 50) & (df.DEATH_EVENT == 1)])
bet_50_60 = len(df.loc[(df.age > 50) & (df.age <= 60) & (df.DEATH_EVENT == 1)])
bet_60_70 = len(df.loc[(df.age > 60) & (df.age <= 70) & (df.DEATH_EVENT == 1)])
bet_70_80 = len(df.loc[(df.age > 70) & (df.age <= 80) & (df.DEATH_EVENT == 1)])
bet_80_90 = len(df.loc[(df.age > 80) & (df.age <= 90) & (df.DEATH_EVENT == 1)])
above_90 = len(df.loc[(df.age >= 90) & (df.DEATH_EVENT == 1)])
ages = [under_50, bet_50_60, bet_60_70, bet_70_80, bet_80_90, above_90]

plt.title('Ages and Death status')
plt.pie(x=ages, labels=labels, autopct='%.2f%%', colors=['orange', 'yellow', 'purple', 'green', 'red', 'pink'])
plt.show()

# Pair plot using Seaborn
X_sea = X_train[['age', 'creatinine_phosphokinase', 'ejection_fraction', 'sex']]

sns.pairplot(X_sea, hue="age", diag_kind="hist")
plt.show()

# Data organizing
y = df['DEATH_EVENT']
x = df.drop(['DEATH_EVENT', 'creatinine_phosphokinase', 'ejection_fraction', 'high_blood_pressure'], axis = 'columns')

# Splitting data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# Using Decision Tree Classifier

from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=3).fit(x_train, y_train)
score = model.score(x_test, y_test)

# Plotting tree for Decision Tree Classifier

plot_tree(model, filled=True, rounded=True)

# Applying Cross Validation with 10 folds

val_score = cross_val_score(model, x, y, cv=10)

# Printing model and Cross Validation Score

print('Model score:\n' + str(score))
print('Cross val score:\n' + str(val_score))

# Applying Sanity Check

dummy = DummyClassifier(strategy='most_frequent').fit(x_train, y_train)
dummy_score = dummy.score(x_test, y_test)
print('Dummy Classifier score (strategy: most_frequent)\n' + str(dummy_score))

plt.show()
