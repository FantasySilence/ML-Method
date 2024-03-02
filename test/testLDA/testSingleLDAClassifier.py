import os
import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

from src.algorithm.LDA.singleLDAClassify import SingleLDAClassifier

iris = load_iris()
X, y = iris.data[:100, :], iris.target[:100]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

lda = SingleLDAClassifier()
lda.fit(X_train, y_train)
y_test_pred = lda.predict(X_test)
print(classification_report(y_test, y_test_pred))