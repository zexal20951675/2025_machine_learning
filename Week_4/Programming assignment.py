from google.colab import files
uploaded = files.upload()   # 會跳出上傳視窗，選擇 O-A0038-003.xml
print("已上傳檔案：", list(uploaded.keys()))
from lxml import etree

# === 修改成你自己的路徑 ===
path = "/content/O-A0038-003.xml"

# 1️⃣ 解析 XML
tree = etree.parse(path)
root = tree.getroot()

# 2️⃣ 取出第一個 Content 節點的文字內容
content_node = root.find('.//{*}Content')
if content_node is None:
    raise RuntimeError("找不到 <Content> 節點，請確認檔案結構")
content_text = content_node.text.strip()

# 3️⃣ 印出前 300 個字元預覽
print("Content 前 300 字元：\n", content_text[:300])

# 4️⃣ 用逗號分隔並轉成浮點數
values = []
for s in content_text.replace('\n', ',').split(','):
    s = s.strip()
    if s == "":
        continue
    try:
        values.append(float(s))
    except ValueError:
        pass

print(f"\n成功解析出 {len(values)} 個數值（預期應為 67 × 120 = 8040）")

# 5️⃣ 檢查是否有 -999 (missing)
import numpy as np
arr = np.array(values)
missing_count = np.sum(arr == -999.0)
print(f"其中無效值 (-999.) 數量：{missing_count}")


import numpy as np
import matplotlib.pyplot as plt

# 你剛剛已經有 values，這裡假設我們直接用它
lon_count = 67
lat_count = 120
grid = np.array(values).reshape((lat_count, lon_count))

print("格點矩陣形狀：", grid.shape)
print("部分內容預覽（前 3 行 5 列）：\n", grid[:3, :5])

# 可視化看看整張格點溫度分布（用 missing value 遮蔽）
masked_grid = np.ma.masked_where(grid == -999.0, grid)
plt.figure(figsize=(6, 5))
plt.imshow(masked_grid, origin='lower', cmap='turbo')
plt.title("Temperature grid preview (masked -999)")
plt.colorbar(label="Temperature (°C)")
plt.show()

# === 基本設定 ===
lon0 = 120.00
lat0 = 21.88
dres = 0.03  # 解析度
lon_count = 67
lat_count = 120

# 生成座標軸
lons = lon0 + np.arange(lon_count) * dres
lats = lat0 + np.arange(lat_count) * dres

# 建立格點座標矩陣
lon_grid, lat_grid = np.meshgrid(lons, lats)

print("lon_grid 形狀：", lon_grid.shape)
print("lat_grid 形狀：", lat_grid.shape)
print("\n左下角：", lon_grid[0, 0], lat_grid[0, 0])
print("右上角：", lon_grid[-1, -1], lat_grid[-1, -1])

# 用 matplotlib 確認經緯度對應溫度的合理分布
plt.figure(figsize=(6, 5))
plt.pcolormesh(lon_grid, lat_grid, masked_grid, shading='auto', cmap='turbo')
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
plt.title("Temperature (°C) by lon/lat grid")
plt.colorbar(label="Temperature (°C)")
plt.show()


import pandas as pd

# 攤平成一維向量
lon_flat = lon_grid.flatten()
lat_flat = lat_grid.flatten()
temp_flat = grid.flatten()

# 建立 DataFrame
df = pd.DataFrame({
    "lon": lon_flat,
    "lat": lat_flat,
    "temperature": temp_flat
})

# 檢查筆數
print("資料筆數：", len(df))
print("其中無效值 (-999.) 筆數：", (df["temperature"] == -999.0).sum())


# === 1️⃣ 分類資料集 ===
df_class = df.copy()
df_class["label"] = (df_class["temperature"] != -999.0).astype(int)

# === 2️⃣ 回歸資料集 ===
df_reg = df[df["temperature"] != -999.0].copy()  # 僅取有效值
df_reg = df_reg.rename(columns={"temperature": "label"})  # 統一標籤欄位名稱

print("分類資料集：", df_class.shape, "  (包含所有格點)")
print("回歸資料集：", df_reg.shape, "  (僅包含有效格點)")
print("\n分類標籤分佈：")
print(df_class["label"].value_counts())

# 預覽
print("\n分類資料集前五筆：")
display(df_class.head())
print("\n回歸資料集前五筆：")
display(df_reg.head())

import numpy as np
import pandas as pd

# 我們使用之前的 grid（120x67）
window_size = 3
pad = window_size // 2

# 將 -999 填入 pad 以便邊界處理
padded = np.pad(grid, pad_width=pad, mode='constant', constant_values=-999.0)

rows, cols = grid.shape
features = []
labels = []

for i in range(rows):
    for j in range(cols):
        window = padded[i:i+window_size, j:j+window_size].flatten()
        center_value = grid[i, j]
        # 對應分類任務的標籤
        label = 0 if center_value == -999.0 else 1
        features.append(window)
        labels.append(label)

X = np.array(features)
y = np.array(labels)

print("特徵矩陣形狀：", X.shape)
print("標籤向量形狀：", y.shape)

# 將經緯度攤平成一維
lon_flat = lon_grid.flatten()
lat_flat = lat_grid.flatten()

# 建立欄位名稱 f0~f8 對應 3x3 window
col_names = [f"f{i}" for i in range(9)]

# 建立 DataFrame
df_features = pd.DataFrame(X, columns=col_names)
df_features["lon"] = lon_flat
df_features["lat"] = lat_flat
df_features["label"] = y

print("完整資料形狀：", df_features.shape)
print("預覽前五筆：")
display(df_features.head())

# 輸出成 CSV（可下載）
output_path = "/content/classification_3x3.csv"
df_features.to_csv(output_path, index=False)
print("✅ 已輸出：", output_path)

# 從 grid 建立回歸特徵矩陣
window_size = 3
pad = window_size // 2
padded = np.pad(grid, pad_width=pad, mode='constant', constant_values=-999.0)

rows, cols = grid.shape
features_reg = []
labels_reg = []
lon_list = []
lat_list = []

for i in range(rows):
    for j in range(cols):
        center_value = grid[i, j]
        if center_value == -999.0:
            continue  # 只保留有效中心格點
        window = padded[i:i+window_size, j:j+window_size].flatten()
        features_reg.append(window)
        labels_reg.append(center_value)
        lon_list.append(lon_grid[i, j])
        lat_list.append(lat_grid[i, j])

X_reg = np.array(features_reg)
y_reg = np.array(labels_reg)

print("回歸樣本數：", X_reg.shape[0])
print("特徵維度：", X_reg.shape[1])
print("標籤筆數：", y_reg.shape[0])
print("前兩筆標籤：", y_reg[:2])

# 建立欄位名稱
col_names = [f"f{i}" for i in range(9)]

# 建立 DataFrame
df_regression = pd.DataFrame(X_reg, columns=col_names)
df_regression["lon"] = lon_list
df_regression["lat"] = lat_list
df_regression["label"] = y_reg

print("回歸資料集形狀：", df_regression.shape)
print("預覽前五筆：")
display(df_regression.head())

# 輸出成 CSV
output_path_reg = "/content/regression_3x3.csv"
df_regression.to_csv(output_path_reg, index=False)
print("✅ 已輸出：", output_path_reg)

# === "分類模型" ===
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

# === 載入分類資料集 ===
df_class = pd.read_csv("/content/classification_3x3.csv")

# === 特徵與標籤 ===
feature_cols = [f"f{i}" for i in range(9)] + ["lon", "lat"]
X = df_class[feature_cols].replace(-999.0, np.nan)  # 將 -999 視為缺值
X = X.fillna(X.mean())  # 用平均值補缺
y = df_class["label"]

# === 訓練 / 測試切分 ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 建立並訓練模型 ===
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# === 預測與評估 ===
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=== Logistic Regression (Classification) ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

# === "回歸模型" ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === 載入回歸資料集 ===
df_reg = pd.read_csv("/content/regression_3x3.csv")

# === 特徵與標籤 ===
feature_cols = [f"f{i}" for i in range(9)] + ["lon", "lat"]
X = df_reg[feature_cols].replace(-999.0, np.nan)
X = X.fillna(X.mean())  # 用平均值補缺
y = df_reg["label"]

# === 訓練 / 測試切分 ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 建立並訓練模型 ===
reg = LinearRegression()
reg.fit(X_train, y_train)

# === 預測與評估 ===
y_pred = reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=== Linear Regression (Temperature Prediction) ===")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# 預覽前 10 筆實際 vs 預測
compare = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": y_pred[:10]
})
display(compare)

# === "分類模型2" ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# === 讀取分類資料集 ===
df_class = pd.read_csv("/content/classification_3x3.csv")

# === 特徵與標籤 ===
feature_cols = [f"f{i}" for i in range(9)] + ["lon", "lat"]
X = df_class[feature_cols].replace(-999.0, np.nan)
X = X.fillna(X.median())  # 用中位數補值
y = df_class["label"]

# === 訓練 / 測試集切分 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 改進版 Logistic Regression ===
clf_balanced = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',   # ⭐ 平衡權重
    solver='lbfgs'
)
clf_balanced.fit(X_train, y_train)

# === 預測與評估 ===
y_pred = clf_balanced.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=== Improved Logistic Regression (Balanced) ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))


# === "回歸模型2" ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === 讀取回歸資料集 ===
df_reg = pd.read_csv("/content/regression_3x3.csv")

# === 移除中心格點特徵 (f4) ===
feature_cols = [f"f{i}" for i in range(9) if i != 4] + ["lon", "lat"]
X = df_reg[feature_cols].replace(-999.0, np.nan)
X = X.fillna(X.median())  # 用中位數補值
y = df_reg["label"]

# === 訓練 / 測試切分 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 建立並訓練模型 ===
reg = LinearRegression()
reg.fit(X_train, y_train)

# === 預測與評估 ===
y_pred = reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=== Improved Linear Regression (No Center Pixel) ===")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# 預覽前 10 筆實際 vs 預測
compare = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": y_pred[:10]
})
display(compare)



import os

for f in os.listdir("/content"):
    if f.endswith(".csv"):
        print(f)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score

# === 1. 讀取資料 ===
df_class = pd.read_csv("/content/classification_3x3.csv")
df_reg = pd.read_csv("/content/regression_3x3.csv")

# === 2. 準備分類資料 ===
feature_cols = [f"f{i}" for i in range(9)] + ["lon", "lat"]
X = df_class[feature_cols].replace(-999.0, np.nan).fillna(df_class.median())
y = df_class["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. Baseline Logistic Regression ===
clf_base = LogisticRegression(max_iter=1000)
clf_base.fit(X_train, y_train)
y_pred_base = clf_base.predict(X_test)

# === 4. Improved Logistic Regression (Balanced) ===
clf_bal = LogisticRegression(max_iter=1000, class_weight='balanced')
clf_bal.fit(X_train, y_train)
y_pred_bal = clf_bal.predict(X_test)

# === 5. Confusion Matrices ===
cm_base = confusion_matrix(y_test, y_pred_base)
cm_bal = confusion_matrix(y_test, y_pred_bal)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, cm, title in zip(axes, [cm_base, cm_bal], ["Baseline Logistic Regression", "Balanced Logistic Regression"]):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, cmap="Greys", colorbar=False)
    ax.set_title(title, fontsize=12)
    ax.grid(False)
plt.tight_layout()
plt.savefig("/content/Fig_CM_Baseline_vs_Balanced.png", dpi=300, bbox_inches="tight")
plt.show()

# === 6. 回歸資料 ===
feature_cols_reg = [f"f{i}" for i in range(9) if i != 4] + ["lon", "lat"]
Xr = df_reg[feature_cols_reg].replace(-999.0, np.nan)
Xr = Xr.fillna(Xr.median())
yr = df_reg["label"]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(Xr_train, yr_train)
yr_pred = reg.predict(Xr_test)

# === 7. Actual vs Predicted Scatter ===
plt.figure(figsize=(5, 5))
plt.scatter(yr_test, yr_pred, edgecolor="black", facecolor="none")
plt.plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], "r--")
plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")
plt.title("Regression: Actual vs Predicted")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.savefig("/content/Fig_Regression_Scatter.png", dpi=300, bbox_inches="tight")
plt.show()

# === 8. Residual Distribution ===
residuals = yr_test - yr_pred
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=30, edgecolor="black", color="gray")
plt.title("Residual Distribution (Regression)")
plt.xlabel("Residual (Actual - Predicted, °C)")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.savefig("/content/Fig_Residual_Distribution.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅ 所有圖已生成！")
print("你可以在左側檔案面板下載以下檔案：")
print(" - Fig_CM_Baseline_vs_Balanced.png")
print(" - Fig_Regression_Scatter.png")
print(" - Fig_Residual_Distribution.png")
