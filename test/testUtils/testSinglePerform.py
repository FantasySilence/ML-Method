import os 
import sys
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer,load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

matplotlib.rcParams['font.sans-serif'] = ['STsong']
matplotlib.rcParams['axes.unicode_minus'] = False
current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

from src.common.utilsAll.PerformanceMetrics import ModelPerformanceMetrics


print("="*60)
bc = load_breast_cancer()
X, y = bc.data, bc.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True, stratify=y)

models = ["LogisticRegression", "BernoulliNB", "LinearDiscriminantAnalysis"]
plt.figure(figsize=(8,6))
for model in models:
    model_obj = eval(model)()
    model_obj.fit(X_train, y_train)
    y_test_prob = model_obj.predict_proba(X_test)
    y_test_lab = model_obj.predict(X_test)
    model1 = ModelPerformanceMetrics(y_test, y_test_prob)
    # # PR曲线
    # pr_ = model1.precision_recall_curve()
    # model1.plt_PRcurve(pr_, label=model, is_show=False)
    # ROC曲线
    roc_ = model1.roc_metrics_curve()
    model1.plt_ROCcurve(roc_, label=model, is_show=False)
plt.show()