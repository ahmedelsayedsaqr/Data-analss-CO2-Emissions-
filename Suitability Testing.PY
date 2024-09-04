import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# إنشاء بيانات غير مستقرة
np.random.seed(0)
t = np.arange(1, 101)
data_nonstationary = np.cumsum(np.random.normal(0, 1, 100))  # سلسلة غير مستقرة
data_stationary = np.diff(data_nonstationary)  # جعلها مستقرة عبر التفريق

# رسم الرسم البياني للاختبار ديكي فولر


def plot_adf(data, title, ax):
    ax.plot(data, label="Original Data")
    ax.plot(np.diff(data), label="Differenced Data (Stationary)")
    ax.set_title(title)
    ax.legend()

# اختبار Johansen (نحتاج إلى مصفوفة بيانات)


def plot_johansen(data, ax):
    # تقليم البيانات غير المستقرة لتتوافق مع طول البيانات المستقرة
    data_trimmed = data[1:]  # إزالة العنصر الأول لمطابقة الأحجام

    # دمج البيانات بعد تعديل الحجم
    combined_data = np.column_stack([data_trimmed, data_stationary])

    result = coint_johansen(combined_data, det_order=0, k_ar_diff=1)
    ax.plot(result.eig, label="Eigenvalues")
    ax.set_title("Johansen Test")
    ax.legend()

# رسم اختبار KPSS


def plot_kpss(data, ax):
    ax.plot(data, label="Original Data (Non-Stationary)")
    ax.set_title("KPSS Test")
    ax.legend()


# إنشاء الرسوم البيانية
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

plot_adf(data_nonstationary, "Dickey-Fuller Test", axs[0, 0])
plot_johansen(data_nonstationary, axs[0, 1])
plot_adf(data_nonstationary, "Augmented Dickey-Fuller Test", axs[1, 0])
plot_kpss(data_nonstationary, axs[1, 1])

plt.tight_layout()
plt.show()
