import os 
import sys
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

from src.common.utilsAll.PerformanceMetrics import ModelPerformanceMetrics
from src.algorithm.logistic_regression.MultiLogisticRegression\
import MultiLogisticRegressor



iris = load_iris()
X, y = iris.data, iris.target
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=0,stratify=y)

model = MultiLogisticRegressor(alpha=0.5, l1_ratio=0.05, l2_ratio=0.05, en_rou=0.6, 
                            normalized=False, batch_size=10, max_epoches=1000, eps=1e-15)
model.fit(X_train, y_train, X_test, y_test)
print("L1正则化模型系数如下：")
theta = model.get_params()
fn = iris.feature_names
for i, w in enumerate(theta[0]):
    print(fn[i], ":", w)
print("theta0:", theta[1])
print("="*70)
y_test_prob = model.predict_prob(X_test)

pm = ModelPerformanceMetrics(y_test, y_test_prob)
print(pm.cal_classification_report())

plt.figure(figsize=(16,12))
plt.subplot(221)
model.plt_loss_curve(lab="L1", is_show=False)   
pr_values = pm.precision_recall_curve()
plt.subplot(222)
pm.plt_PRcurve(pr_values, is_show=False)
roc_values = pm.roc_metrics_curve()
plt.subplot(223)
pm.plt_ROCcurve(roc_values, is_show=False)
plt.subplot(224)
cm = pm.cal_confusion_matrix()
model.plt_confusion_matrix(cm, is_show=False)
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()