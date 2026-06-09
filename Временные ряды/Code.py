import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
df = pd.read_csv(
    "AirQualityUCI.csv",
    sep=";",
    decimal=","
)
# удаляем пустые столбцы
df = df.drop(columns=["Unnamed: 15", "Unnamed: 16"])
# удаляем полностью пустые строки
df = df.dropna(how="all")
# объединяем дату и время
df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H.%M.%S"
)
df = df.set_index("Datetime")
# удаляем исходные столбцы
df = df.drop(columns=["Date", "Time"])
# заменяем служебные пропуски
df = df.replace(-200, np.nan)
print(df.head())
print("\nРазмерность:")
print(df.shape)
print("\nТипы данных:")
print(df.dtypes)
print("\nИнформация:")
print(df.info())
time_diff = df.index.to_series().diff()
print("Наиболее частый интервал:")
print(time_diff.mode()[0])
plt.figure(figsize=(14,5))
plt.plot(df.index, df["CO(GT)"])
plt.title("Концентрация CO(GT)")
plt.xlabel("Дата")
plt.ylabel("CO")
plt.grid(True)
plt.show()
features = [
    "CO(GT)",
    "NMHC(GT)",
    "NOx(GT)",
    "T",
    "RH",
    "AH"
]
fig, axes = plt.subplots(
    len(features),
    1,
    figsize=(14,12),
    sharex=True
)
for i, col in enumerate(features):
    axes[i].plot(df.index, df[col])
    axes[i].set_title(col)
    axes[i].grid(True)
plt.tight_layout()
plt.show()
stats = df.describe().T
stats = stats[
    ["count","mean","std","min","25%","50%","75%","max"]
]
print(stats)
for col in df.columns:
    mean = df[col].mean()
    median = df[col].median()

    print(
        f"{col}: mean={mean:.2f}, median={median:.2f}"
    )
missing_percent = (
    df.isna().sum()
    / len(df)
    * 100
)
missing_percent = missing_percent.sort_values(
    ascending=False
)
print(missing_percent)
plt.figure(figsize=(14,6))
sns.heatmap(
    df.isna(),
    cbar=False
)
plt.title("Карта пропусков")
plt.show()
co = df["CO(GT)"]
mean = co.mean()
std = co.std()
upper = mean + 3 * std
lower = mean - 3 * std
outliers = co[
    (co > upper) |
    (co < lower)
]
print("Нижняя граница:", lower)
print("Верхняя граница:", upper)
print("Количество выбросов:")
print(len(outliers))
print(
    "Процент выбросов:",
    len(outliers)/len(co.dropna())*100
)
plt.figure(figsize=(8,4))
sns.boxplot(x=df["CO(GT)"])
plt.title("Диаграмма размаха CO(GT)")
plt.show()
plt.figure(figsize=(14,6))
sns.boxplot(data=df)
plt.xticks(
    rotation=90
)
plt.title(
    "Диапазоны значений признаков"
)
plt.show()
corr = df.corr()
print(corr)
plt.figure(figsize=(12,10))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title(
    "Корреляционная матрица"
)
plt.show()
corr_pairs = corr.unstack()
corr_pairs = corr_pairs.sort_values(
    ascending=False
)
print(corr_pairs.head(30))
# Декомпозиция временного ряда

series = df["CO(GT)"]

# Заполняем пропуски интерполяцией
series = series.interpolate()

# Суточная сезонность (24 часа)
decomposition = seasonal_decompose(
    series,
    model="additive",
    period=24
)

# Извлекаем компоненты
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Построение графиков
fig, axes = plt.subplots(
    4,
    1,
    figsize=(14,10),
    sharex=True
)

axes[0].plot(series)
axes[0].set_title("Исходный временной ряд CO(GT)")
axes[0].grid(True)

axes[1].plot(trend)
axes[1].set_title("Тренд")
axes[1].grid(True)

axes[2].plot(seasonal)
axes[2].set_title("Сезонная компонента")
axes[2].grid(True)

axes[3].plot(residual)
axes[3].set_title("Остатки (шум)")
axes[3].grid(True)

plt.tight_layout()
plt.show()
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
signal = trend + seasonal
result = pd.DataFrame({
    "signal": signal,
    "noise": residual
})
result = result.dropna()
signal_var = np.var(
    result["signal"]
)
noise_var = np.var(
    result["noise"]
)
snr = 10 * np.log10(
    signal_var / noise_var
)
print("Дисперсия сигнала:", signal_var)
print("Дисперсия шума:", noise_var)
print("SNR:", snr, "дБ")
plt.figure(figsize=(8,5))

plt.hist(
    result["noise"],
    bins=50
)
plt.title(
    "Распределение остатков"
)
plt.xlabel("Остатки")
plt.ylabel("Частота")
plt.show()