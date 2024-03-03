import os
import sys
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

from src.common.utilsAll.multiclass import MultiClassifierWrapper
from src.algorithm.logistic_regression.singleLogisticRegression\
import LogisticRegressor


digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

lg_lr = LogisticRegressor()

mcw = MultiClassifierWrapper(lg_lr, mode="ovo")
mcw.fit(X_train, y_train)
y_test_hat = mcw.predict(X_test)
print(classification_report(y_test, y_test_hat))
