import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv('student-mat.csv', sep=';')
data['sex'] = data['sex'].map({'F': 0, 'M': 1})
data = data[['age', 'sex', 'Fedu', 'Medu', 'failures', 'absences', 'G1', 'G2', 'G3']]


X = data[['age', 'sex', 'Fedu', 'Medu', 'failures', 'absences', 'G1', 'G2']]
Y = data['G3']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)


# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
'''best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    model = LinearRegression()

    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print("Accuracy: " + str(score))

    if score > best:
        best = score
        # Saving best accuracy model
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(model, f)'''


# Loading the model

pickle_in = open("studentmodel.pickle", "rb")
model = pickle.load(pickle_in)

# print("Model Score ", best)

predictions = model.predict(X_test)

# score = model.score(X_test, y_test)

# print("Model score:", score)


# predictions = model.predict(X_test)

print("Co: \n", model.coef_)
print("Intercept: \n", model.intercept_)

for x in range(len(predictions)):
    print(predictions[x], X_test.iloc[x], y_test.iloc[x])


p = 'absences'
style.use("ggplot")
plt.scatter(data[p], data['G3'])
plt.xlabel(p)
plt.ylabel('Final Grade')
plt.show()

