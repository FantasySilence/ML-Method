import os
import sys
import warnings
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)
warnings.filterwarnings("ignore")

from src.algorithm.SVM.svm_smo_classifier import SVMClassifier
from src.common.utilsAll.multiclass import MultiClassifierWrapper


iris = load_iris()
X, y = iris.data, iris.target
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, shuffle=True
)

svm = SVMClassifier(C=10.0, kernel="linear")
modes = ["ovo", "ovr"]
for mode in modes:
    mcw = MultiClassifierWrapper(base_classifier=svm, mode=mode)
    mcw.fit(X_train, y_train)
    y_test_pred = mcw.predict(X_test)
    print(classification_report(y_test, y_test_pred))
    print("=" * 50)