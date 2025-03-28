import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Giả định y_true là nhãn thực tế, y_pred là nhãn dự đoán của mô hình
y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])  # 1: Real, 0: Fake
y_pred = np.array([1, 0, 1, 0, 0, 0, 1, 0, 1, 1])  # Dự đoán của mô hình

# Tạo confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Vẽ confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
            'Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Hiển thị báo cáo phân loại
print(classification_report(y_true, y_pred, target_names=['Fake', 'Real']))
