import os 
import sys
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

current_path = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.dirname(current_path)
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

from src.common.utilsAll.PerformanceMetrics import ModelPerformanceMetrics
from src.algorithm.logistic_regression.singleLogisticRegression\
import LogisticRegressor



bc_data = load_breast_cancer()
X, y = bc_data.data, bc_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

model = LogisticRegressor(alpha=0.5, l1_ratio=0.5, batch_size=20, 
                          max_epoches=1000, eps=1e-15)
model.fit(X_train, y_train, X_test, y_test)
print("L1正则化模型参数如下：\n", model.get_param())
print("="*70)
y_test_prob = model.predict_proba(X_test)
pm = ModelPerformanceMetrics(y_test, y_test_prob)
print(pm.cal_classification_report())


plt.figure(figsize=(18, 12))
plt.subplot(221)
model.plt_loss_curve(lab='L1', is_show=False)
plt.subplot(222)
pr_values = pm.precision_recall_curve()
pm.plt_PRcurve(pr_values, is_show=False)
plt.subplot(223)
roc_values = pm.roc_metrics_curve()
pm.plt_ROCcurve(roc_values, is_show=False)
plt.subplot(224)
cm = pm.cal_confusion_matrix()
model.plt_confusion_matrix(cm, label_name=["malignant","benign"], is_show=False)
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()