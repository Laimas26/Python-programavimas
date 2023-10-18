import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib import style



def readingFiles():
    return pd.read_csv('car.data', sep=',')

def variableFixes(data):
    # Define a dictionary for mapping values in multiple columns
    mapping_dict = {
        'buying': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
        'maint': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
        'door': {'5more': 5},
        'persons': {'more': 5},
        'lug_boot': {'small': 0, 'med': 1, 'big': 2},
        'safety': {'low': 0, 'med': 1, 'high': 2},
        'class': {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
    }
    # Apply the mapping to each specified column
    for column, mapping in mapping_dict.items():
        data[column] = data[column].map(mapping).fillna(data[column]).astype(int)


def variableFixes2(data):
    #  Kitas budas pervadinti labels ----------------------------------------------------
    
    le = preprocessing.LabelEncoder()
    buying = le.fit_transform(list(data['buying']))
    maint = le.fit_transform(list(data['maint']))
    door = le.fit_transform(list(data['door']))
    persons = le.fit_transform(list(data['persons']))
    lug_boot = le.fit_transform(list(data['lug_boot']))
    safety = le.fit_transform(list(data['safety']))
    cls = le.fit_transform(list(data['class']))
    
    return buying, maint, door, persons, lug_boot, safety, cls

def dataSplitting(data):
    X = data[['buying', 'maint', 'persons', 'safety']]
    Y = data['class']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    return x_train, x_test, y_train, y_test

def dataSplitting2(buying, maint, door, persons, lug_boot, safety, cls):
    X = list(zip(buying, maint, door, persons, lug_boot, safety))
    Y = list(cls)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    return x_train, x_test, y_train, y_test

def modelTraining(x_train, y_train):
    k = 3
    model = KNeighborsClassifier(n_neighbors=k)
    
    return model.fit(x_train,y_train)

def visualizePlot(data):
    p = 'buying'
    style.use("ggplot")
    plt.scatter(data[p], data['class'])
    plt.xlabel(p)
    plt.ylabel('class')
    plt.show()


def main():
    data = readingFiles()
    # variableFixes(data)
    buying, maint, door, persons, lug_boot, safety, cls = variableFixes2(data)
    # x_train, x_test, y_train, y_test = dataSplitting(data)
    x_train, x_test, y_train, y_test = dataSplitting2(buying, maint, door, persons, lug_boot, safety, cls)
    model = modelTraining(x_train, y_train)
    score = model.score(x_test, y_test)
    # visualizePlot(data)
    
    print(f"Accuracy is: {score}")
    
    names = ['unacc', 'acc', 'good', 'vgood']
    
    # Predictions based on x_test data
    predictions = model.predict(x_test)
    
    
    for x in range(len(predictions)):
        print("Predicted:", names[predictions[x]], "  Data:", x_test[x], "  Actual:", names[y_test[x]], "\n")
        n = model.kneighbors([x_test[x]], 9, True)
        print("N: ", n)
        # print("Predicted: ", names[predictions[x]])
        # print("Data: ", x_test[x])
        # print("Actual: ", names[y_test[x]])
    
    

    # for x in range(len(predictions)):
    #     print(x_test.iloc[x], y_test.iloc[x])
    
main()
