import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



def loadData():
    cancer = datasets.load_breast_cancer()
    # print(cancer.feature_names)
    # print(cancer.target_names)
    
    return cancer

def setData(cancer):
    x = cancer.data
    y = cancer.target
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    return x, y, x_train, x_test, y_train, y_test


def main():
    cancer = loadData()
    x, y, x_train, x_test, y_train, y_test = setData(cancer)
    # print(x_train, y_train)
    classes = ['malignant' 'benign']
    
    clf = svm.SVC(kernel='linear', C=2)
    clf.fit(x_train, y_train)
    
    # Palyginimui su KNN klasifikavimu
    # clf = KNeighborsClassifier(n_neighbors=1)
    # clf.fit(x_train, y_train)
    
    
    y_pred = clf.predict(x_test)
    
    acc = metrics.accuracy_score(y_test, y_pred)
    
    print('Accuracy: ', acc)

main()