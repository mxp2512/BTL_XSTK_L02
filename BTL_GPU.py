import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import os
import dataframe_image as dfi
import scikit_posthocs as sp
from scipy.stats import kruskal
from scipy.stats import shapiro
from scipy.stats import norm
from scipy.stats import shapiro, levene
from statsmodels.formula.api import ols
from scipy.stats import f
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LassoCV
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.stattools import durbin_watson

# ==========================================================
#  BÀI TẬP LỚN XÁC SUẤT - THỐNG KÊ
#  ĐỀ TÀI: PHÂN TÍCH TẬP DỮ LIỆU GPU
# ==========================================================

# 1. Đọc dữ liệu
df = pd.read_csv("All_GPUs.csv")

# Hiển thị 6 dòng đầu tiên của dữ liệu đã đọc
pd.set_option('display.width', 170)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
#print(df.head())

# 2. Tiền xử lý dữ liệu

# Thay tất cả các ô trống ("") bằng NaN
df.replace("", np.nan, inplace=True)

# Thay các giá trị có định dạng giống "\n-" (hoặc chỉ chứa "-") bằng NaN
df.replace(r"^\n*-\s*$", np.nan, regex=True, inplace=True)

# Tạo DataFrame thống kê số lượng và tỷ lệ giá trị khuyết
na_summary = pd.DataFrame({
    "Column": df.columns,
    "NA_Count": df.isna().sum().values,
    "NA_Percentage": (df.isna().mean() * 100).values
})

# Làm đẹp hiển thị (sắp xếp giảm dần theo tỷ lệ thiếu)
na_summary = na_summary.sort_values(by="NA_Percentage", ascending=False)

# Hiển thị bảng thống kê
mid = len(na_summary) // 2
table_left = na_summary.iloc[:mid]
table_right = na_summary.iloc[mid:].reset_index(drop=True)

# Gộp hai bảng theo chiều ngang
#combined = pd.concat([table_left.reset_index(drop=True), table_right], axis=1)

# In ra gọn gàng
print(combined.to_string(index=False, formatters={
    "NA_Percentage": "{:.2f}%".format
}))


# --- Vẽ biểu đồ ---
plt.figure(figsize=(10, 6))
sns.barplot(data=na_summary, x="Column", y="NA_Percentage", color="steelblue")

# Thêm nhãn phần trăm trên cột
for i, row in enumerate(na_summary.itertuples()):
    plt.text(i, row.NA_Percentage + 0.5, f"{row.NA_Percentage:.1f}%", 
             ha='center', va='bottom', fontsize=8, color='black')

# Tiêu đề và nhãn
plt.title("Tỷ lệ dữ liệu khuyết ở các biến", fontsize=13)
plt.xlabel("Biến", fontsize=11)
plt.ylabel("Tỷ lệ dữ liệu khuyết (%)", fontsize=11)

# Xoay nhãn trục X cho dễ đọc
plt.xticks(rotation=90, fontsize=9)
plt.tight_layout()
plt.show()

# Lọc ra các cột có tỷ lệ NA dưới 10%
selected_columns = na_summary.loc[na_summary["NA_Percentage"] < 10, "Column"]

# Giữ lại các cột thỏa điều kiện trong df
new_GPU_data = df[selected_columns].copy()

# Xóa các hàng chứa dữ liệu khuyết
new_GPU_data = new_GPU_data.dropna()

# (Tùy chọn) Kiểm tra kích thước kết quả
print("Kích thước ban đầu:", df.shape)
print("Kích thước sau khi lọc:", new_GPU_data.shape)

columns_to_clean = ["L2_Cache", "Memory_Bandwidth", "Memory_Bus", "Memory_Speed"]

# Hàm xóa đơn vị (chỉ giữ lại số và dấu chấm thập phân)
def remove_units(column):
    # Dùng regex: xóa mọi ký tự không phải số hoặc dấu '.'
    cleaned = column.astype(str).str.replace(r"[^0-9.]", "", regex=True)
    # Chuyển về kiểu float
    cleaned = pd.to_numeric(cleaned, errors='coerce')
    return cleaned

# Áp dụng cho các cột được chọn
new_GPU_data[columns_to_clean] = new_GPU_data[columns_to_clean].apply(remove_units)

# Kiểm tra kết quả
#print(new_GPU_data[columns_to_clean].head())


# Lựa chọn các cột cần lấy
selected_columns = [
    "Memory_Bandwidth","L2_Cache","Memory_Speed","Memory_Bus","Memory_Type",
    "Shader","Dedicated","Manufacturer"
]

# Tạo dataframe mới chỉ chứa các cột trên
main_data = new_GPU_data[selected_columns].copy()

# Xem trước 5 dòng đầu
print(main_data.head())


# ==========================================================
# 3. THỐNG KÊ MÔ TẢ
# ==========================================================

# Lọc các cột biến định lượng
numeric_cols = ["Memory_Bandwidth", "L2_Cache", "Memory_Speed", "Memory_Bus"]
numeric_data = main_data[numeric_cols]

# Tính toán thống kê mô tả
summary_stats = pd.DataFrame({
    "Mean": numeric_data.mean(),
    "SD": numeric_data.std(),
    "Min": numeric_data.min(),
    "Q1": numeric_data.quantile(0.25),
    "Median": numeric_data.median(),
    "Q3": numeric_data.quantile(0.75),
    "Max": numeric_data.max()
})

# Hiển thị bảng
print(summary_stats)

# Thống kê số lượng cho các biến phân loại
categorical_cols = ["Manufacturer", "Shader", "Memory_Type", "Dedicated"]
for col in categorical_cols:
    #print(f"=== {col} ===")
    #print(main_data[col].value_counts())
    #print("\n")

# ==========================================================
# 4. VẼ BIỂU ĐỒ CƠ BẢN
# ==========================================================

# Histogram cho biến Memory_Bandwidth
plt.figure(figsize=(8, 5))

# Vẽ histogram
sns.histplot(main_data['Memory_Bandwidth'], 
             bins=50,                
             color='steelblue', 
             edgecolor='black', 
             alpha=0.7)
plt.title("Phân Phối của Memory_Bandwidth", fontsize=14)
plt.xlabel("Memory_Bandwidth (GB/s)", fontsize=12)
plt.ylabel("Số Lượng", fontsize=12)
plt.tight_layout()
plt.show()


# Vẽ boxplot cho Memory_Bandwidth theo các biến phân loại
categorical_features = ['Manufacturer', 'Dedicated', 'Memory_Type', 'Shader']

# Thiết lập style chung
sns.set_style("whitegrid")

# Vẽ từng boxplot
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=feature, y='Memory_Bandwidth', data=main_data,
                color='steelblue', fliersize=5, linewidth=1.2)
    plt.title(f"Boxplot của Memory_Bandwidth theo {feature}", fontsize=13)
    plt.xlabel(feature, fontsize=11)
    plt.ylabel("Memory_Bandwidth (GB/s)", fontsize=11)
    
    # Xoay nhãn trục X cho dễ đọc
    rotation = 45 if feature == 'Manufacturer' else 0
    ha = 'right' if feature == 'Manufacturer' else 'center'
    plt.xticks(rotation=rotation, ha=ha)

plt.tight_layout()
plt.show()


# Scatterplot của Memory_Bandwidth theo các biến kỹ thuật
features = ["Memory_Speed", "L2_Cache", "Memory_Bus"]
units = {
    "Memory_Speed": "(MHz)",
    "L2_Cache": "(KB)",
    "Memory_Bus": "(bits)"
}

# Tạo figure với 3 biểu đồ trên cùng hàng
plt.figure(figsize=(15, 4))

for i, feature in enumerate(features, 1):
    plt.subplot(1, 3, i)
    sns.scatterplot(data=main_data, x=feature, y="Memory_Bandwidth",
                    color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.4)
    plt.title(f"Scatterplot của Memory_Bandwidth theo {feature}", fontsize=12)
    plt.xlabel(f"{feature} {units[feature]}", fontsize=11)
    plt.ylabel("Memory_Bandwidth (GB/s)", fontsize=11)
    plt.grid(True)

plt.tight_layout()
plt.show()

# Chọn các biến số từ main_data
numeric_features = ["Memory_Bandwidth", "Memory_Speed", "L2_Cache", "Memory_Bus"]

# Tạo ma trận tương quan
corr_matrix = main_data[numeric_features].corr()

# Vẽ heatmap tương quan
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, 
            annot=True, 
            cmap="coolwarm", 
            fmt=".2f", 
            linewidths=0.5, 
            cbar_kws={"shrink": 0.8})

plt.title("Ma trận tương quan giữa các biến kỹ thuật", fontsize=14)
plt.tight_layout()
plt.show()

#=========================================================
# 5. KIỂM ĐỊNH TRUNG BÌNH HAI MẪU
#=========================================================

# Tách dữ liệu theo Dedicated = "No" và "Yes"
GPUnodedicated = main_data[main_data["Dedicated"] == "No"]
GPUdedicated = main_data[main_data["Dedicated"] == "Yes"]

# Tính các thống kê mô tả cho Memory_Bandwidth
n1 = len(GPUnodedicated["Memory_Bandwidth"])
xtb1 = GPUnodedicated["Memory_Bandwidth"].mean()
s1 = GPUnodedicated["Memory_Bandwidth"].std()

n2 = len(GPUdedicated["Memory_Bandwidth"])
xtb2 = GPUdedicated["Memory_Bandwidth"].mean()
s2 = GPUdedicated["Memory_Bandwidth"].std()

# Tạo bảng kết quả
#summary = pd.DataFrame({
#    "n1": [n1],
#    "xtb1": [xtb1],
#    "s1": [s1],
#    "n2": [n2],
#    "xtb2": [xtb2],
#    "s2": [s2]
#})

#print(summary)

# Kiểm định phân phối chuẩn
# Q-Q Plot
# Dữ liệu 2 nhóm
data_no = GPUnodedicated["Memory_Bandwidth"]
data_yes = GPUdedicated["Memory_Bandwidth"]

# Vẽ QQ Plot cho 2 nhóm
plt.figure(figsize=(10, 4))

# --- Nhóm Non-Dedicated ---
plt.subplot(1, 2, 1)
stats.probplot(data_no, dist="norm", plot=plt)
plt.title("Q-Q Plot for Memory Bandwidth (Non-Dedicated)")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")

# --- Nhóm Dedicated ---
plt.subplot(1, 2, 2)
stats.probplot(data_yes, dist="norm", plot=plt)
plt.title("Q-Q Plot for Memory Bandwidth (Dedicated)")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")

plt.tight_layout()
plt.show()

# Shapiro Test
# Kiểm định Shapiro cho nhóm Non-Dedicated
stat_no, p_no = shapiro(data_no)
print("Non-Dedicated:")
print(f"  Shapiro-Wilk statistic = {stat_no:.4f}, p-value = {p_no:.4f}")

# Kiểm định Shapiro cho nhóm Dedicated
stat_yes, p_yes = shapiro(data_yes)
print("Dedicated:")
print(f"  Shapiro-Wilk statistic = {stat_yes:.4f}, p-value = {p_yes:.4f}")

# Diễn giải kết quả
alpha = 0.05
for group, p in zip(["Non-Dedicated", "Dedicated"], [p_no, p_yes]):
    if p > alpha:
        print(f"{group}: Không bác bỏ giả thuyết chuẩn (dữ liệu có phân phối chuẩn)")
    else:
        print(f"{group}: Bác bỏ giả thuyết chuẩn (dữ liệu không có phân phối chuẩn)")

# Tính giá trị kiểm định thống kê
Zo = (xtb1 - xtb2) / np.sqrt((s1**2 / n1) + (s2**2 / n2))
print(f"Giá trị kiểm định Z = {Zo:.4f}")

# Miền bác bỏ
alpha = 0.05
Z_critical = norm.ppf(1 - alpha)
print(f"Z_critical (upper tail, alpha=0.05) = {Z_critical:.4f}")

#================================================================
# 6. ANOVA MỘT NHÂN TỐ
#================================================================
# Kiểm tra phân phối chuẩn (Shapiro-Wilk) ---
groups = main_data["Manufacturer"].unique()
for g in groups:
    data_g = main_data.loc[main_data["Manufacturer"] == g, "Memory_Bandwidth"]
    stat, p = shapiro(data_g)
    print(f"{g}: W = {stat:.4f}, p-value = {p:.4f}")

# Kết luận: Nếu p > 0.05 → không bác bỏ H0 → dữ liệu ~ chuẩn

# Kiểm tra đồng nhất phương sai (Levene Test) ---
group_data = [main_data.loc[main_data["Manufacturer"] == g, "Memory_Bandwidth"] for g in groups]
stat, p = levene(*group_data)
print(f"\nLevene test: W = {stat:.4f}, p-value = {p:.4f}")

# Kết luận: Nếu p > 0.05 → không bác bỏ H0 → phương sai bằng nhau

# 1. Lấy 4 mẫu theo Manufacturer (ignore case)
frequency_AMD    = main_data.loc[main_data['Manufacturer'].str.contains('AMD',   case=False, na=False), 'Memory_Bandwidth'].dropna().astype(float)
frequency_ATI    = main_data.loc[main_data['Manufacturer'].str.contains('ATI',   case=False, na=False), 'Memory_Bandwidth'].dropna().astype(float)
frequency_Intel  = main_data.loc[main_data['Manufacturer'].str.contains('Intel', case=False, na=False), 'Memory_Bandwidth'].dropna().astype(float)
frequency_Nvidia = main_data.loc[main_data['Manufacturer'].str.contains('Nvidia',case=False, na=False), 'Memory_Bandwidth'].dropna().astype(float)

# 2. Hàm loại bỏ điểm ngoại lai theo quy tắc IQR
def remove_outliers_iqr(series):
    if series.empty:
        return series
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series[(series >= lower) & (series <= upper)]

# Áp dụng cho từng nhóm
frequency_AMD_clean    = remove_outliers_iqr(frequency_AMD)
frequency_ATI_clean    = remove_outliers_iqr(frequency_ATI)
frequency_Intel_clean  = remove_outliers_iqr(frequency_Intel)
frequency_Nvidia_clean = remove_outliers_iqr(frequency_Nvidia)

# 3. Tính thống kê: mean, sd (sample std), n
stats_df = pd.DataFrame({
    "Frequency": ["frequency_AMD", "frequency_ATI", "frequency_Intel", "frequency_Nvidia"],
    "Mean": [
        frequency_AMD_clean.mean(),
        frequency_ATI_clean.mean(),
        frequency_Intel_clean.mean(),
        frequency_Nvidia_clean.mean()
    ],
    "Variance": [
        frequency_AMD_clean.std(ddof=1),  
        frequency_ATI_clean.std(ddof=1),
        frequency_Intel_clean.std(ddof=1),
        frequency_Nvidia_clean.std(ddof=1)
    ],
    "Size": [
        len(frequency_AMD_clean),
        len(frequency_ATI_clean),
        len(frequency_Intel_clean),
        len(frequency_Nvidia_clean)
    ]
})

# Hiển thị bảng thống kê
print(stats_df)

# Số nhóm = 4 (AMD, ATI, Intel, Nvidia)
k = 4

# Kích thước mẫu (sau khi loại ngoại lai)
n1 = len(frequency_AMD_clean)
n2 = len(frequency_ATI_clean)
n3 = len(frequency_Intel_clean)
n4 = len(frequency_Nvidia_clean)

# Bậc tự do
df1 = k - 1
df2 = (n1 + n2 + n3 + n4) - k

# Giá trị tới hạn F
F_critical = f.ppf(1 - alpha, df1, df2)

print(f"F-critical (alpha = 0.05): {F_critical:.4f}")
print(f"df1 = {df1}, df2 = {df2}")

# Gộp dữ liệu thành một DataFrame ---
df_anova = pd.DataFrame({
    "Gia_tri": (
        list(frequency_AMD_clean)
        + list(frequency_ATI_clean)
        + list(frequency_Intel_clean)
        + list(frequency_Nvidia_clean)
    ),
    "Frequency": (
        ["AMD"] * len(frequency_AMD_clean)
        + ["ATI"] * len(frequency_ATI_clean)
        + ["Intel"] * len(frequency_Intel_clean)
        + ["Nvidia"] * len(frequency_Nvidia_clean)
    ),
})

# Thực hiện kiểm định ANOVA ---
model = ols("Gia_tri ~ C(Frequency)", data=df_anova).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("=== Kết quả ANOVA ===")
print(anova_table)

# Phân tích hậu kiểm Tukey HSD ---
tukey = pairwise_tukeyhsd(endog=df_anova["Gia_tri"],
                          groups=df_anova["Frequency"],
                          alpha=0.05)

print("\n=== Kết quả kiểm định Tukey HSD ===")
print(tukey)


# ==========================================================
# 6. HỒI QUY TUYẾN TÍNH ĐA BIẾN
# ==========================================================
# Chia dữ liệu 8:2
train_data, test_data = train_test_split(main_data, test_size=0.2, random_state=123)

# --- Loại bỏ các biến không dùng ---
cols_to_drop = ['Manufacturer', 'Dedicated', 'Shader']
train_data = train_data.drop(columns=cols_to_drop, errors='ignore')
test_data = test_data.drop(columns=cols_to_drop, errors='ignore')

# Tạo biến giả cho các biến phân loại
train_data = pd.get_dummies(train_data, columns=['Memory_Type', 'Shader'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Memory_Type', 'Shader'], drop_first=True)
train_data = pd.get_dummies(train_data, columns=['Memory_Type'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Memory_Type'], drop_first=True)

# Đồng bộ cột giữa train và test
test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

# Biến độc lập và phụ thuộc
X_train = train_data.drop(columns=['Memory_Bandwidth'])
y_train = train_data['Memory_Bandwidth']

# Thêm hằng số
X_train = sm.add_constant(X_train)

# Ép tất cả dữ liệu sang kiểu float (tránh lỗi dtype object)
X_train = X_train.astype(float)
y_train = y_train.astype(float)

# Xây dựng mô hình hồi quy tuyến tính
model = sm.OLS(y_train, X_train).fit()

# In kết quả
print(model.summary())
print(f"R-squared: {model.rsquared:.4f}")
print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")

# Vẽ các đồ thị phần dư
# Lấy các giá trị fitted và residual
fitted_vals = model.fittedvalues
residuals = model.resid
plt.figure(figsize=(12, 10))

# 1. Residuals vs Fitted
plt.subplot(2, 2, 1)
plt.scatter(fitted_vals, residuals, edgecolor='k', alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("1. Residuals vs Fitted")

# 2. Normal Q-Q
plt.subplot(2, 2, 2)
sm.qqplot(residuals, line='45', fit=True, ax=plt.gca())
plt.title("2. Normal Q-Q")

# 3. Scale-Location
plt.subplot(2, 2, 3)
plt.scatter(fitted_vals, np.sqrt(np.abs(residuals)), edgecolor='k', alpha=0.7)
plt.xlabel("Fitted values")
plt.ylabel("√|Residuals|")
plt.title("3. Scale-Location")

# 4. Residuals vs Leverage (vẽ Cook’s Distance)
plt.subplot(2, 2, 4)
influence = model.get_influence()
leverage = influence.hat_matrix_diag
studentized_residuals = influence.resid_studentized_external
cooks_d = influence.cooks_distance[0]

n = model.nobs
k = model.df_model

# Vẽ điểm
sc = plt.scatter(leverage, studentized_residuals, s=1000 * cooks_d, alpha=0.6, edgecolors='k')

# Đường Cook’s Distance
def cooks_line(D, leverage, n, k):
    return np.sqrt((D * k * (1 - leverage)) / leverage)

x = np.linspace(0.001, max(leverage) + 0.01, 100)
y05 = cooks_line(0.5, x, n, k)
y1 = cooks_line(1, x, n, k)
plt.plot(x, y05, label="Cook's D = 0.5", ls='--', color='orange', linewidth=1.8)
plt.plot(x, -y05, ls='--', color='orange', linewidth=1.8)
plt.plot(x, y1, label="Cook's D = 1.0", ls='-.', color='red', linewidth=1.8)
plt.plot(x, -y1, ls='-.', color='red', linewidth=1.8)

# Ghi nhãn điểm ảnh hưởng cao
influential_points = np.where(cooks_d > 0.5)[0]
for i in influential_points:
    plt.text(leverage[i], studentized_residuals[i], str(i), fontsize=8, color='darkblue')

plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Leverage')
plt.ylabel('Studentized Residuals')
plt.title('4. Residuals vs Leverage')
plt.legend()
plt.tight_layout()
plt.show()

# (Tùy chọn) xem CooksD các điểm 
influence_df = pd.DataFrame({
    "CooksD": cooks_d
}).sort_values("CooksD", ascending=False)

print(influence_df.head(4))

# Durbin - Watson test 
dw_value = durbin_watson(model.resid)
print("Durbin–Watson:", dw_value)

# VIF
X = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(X.shape[1])]

print(vif_data)

# Chuẩn hóa kiểu dữ liệu ---
bool_cols = [
    'Memory_Type_DDR2', 'Memory_Type_DDR3', 'Memory_Type_DDR4',
    'Memory_Type_GDDR2', 'Memory_Type_GDDR3', 'Memory_Type_GDDR4',
    'Memory_Type_GDDR5', 'Memory_Type_GDDR5X',
    'Memory_Type_HBM-1', 'Memory_Type_HBM-2', 'Memory_Type_eDRAM'
]

for col in bool_cols:
    train_data[col] = train_data[col].astype(int)
    test_data[col] = test_data[col].astype(int)

# Chuẩn bị dữ liệu ---
X_train = train_data.drop(columns=['Memory_Bandwidth'])
y_train = train_data['Memory_Bandwidth']
X_test = test_data.drop(columns=['Memory_Bandwidth'], errors='ignore')

# Đảm bảo cột giống hệt
X_test = X_test[X_train.columns]
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Dự đoán ---
predictions = model.predict(X_test)

# Thêm Predicted_Memory_Bandwidth vào test_data
test_data['Predicted_Memory_Bandwidth'] = predictions

print(test_data.head())

# Tính RMSE và R² ---
rmse = np.sqrt(np.mean((test_data['Memory_Bandwidth'] - predictions) ** 2))
r2 = 1 - np.sum((test_data['Memory_Bandwidth'] - predictions) ** 2) / np.sum((test_data['Memory_Bandwidth'] - np.mean(test_data['Memory_Bandwidth'])) ** 2)

print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")

#print(train_data.dtypes)
#print(test_data.dtypes)

# ==========================================================
# 7. KIỂM ĐỊNH KRUSKAL - WALLIS
# ==========================================================
# Tách dữ liệu theo từng hãng
groups = main_data["Manufacturer"].unique()
group_data = [main_data.loc[main_data["Manufacturer"] == g, "Memory_Bandwidth"] for g in groups]

# Thực hiện kiểm định Kruskal–Wallis
stat, p = kruskal(*group_data)
print(f"Kruskal–Wallis H = {stat:.4f}, p-value = {p:.4f}")

# Kết luận
if p < 0.05:
    print("→ Có sự khác biệt có ý nghĩa thống kê giữa ít nhất hai hãng sản xuất (p < 0.05).")
else:
    print("→ Không có sự khác biệt có ý nghĩa giữa các hãng (p ≥ 0.05).")

# Thực hiện Dunn post-hoc test
posthoc = sp.posthoc_dunn(main_data, val_col='Memory_Bandwidth', group_col='Manufacturer', p_adjust='bonferroni')
print(posthoc)

# ==========================================================
# 7. MÔ HÌNH HỒI QUY RIDGE/LASSO
# ==========================================================
n_features = ['L2_Cache', 'Memory_Speed', 'Memory_Bus']
c_features = ['Memory_Type']

numeric_features = n_features
categorical_features = c_features

# Tạo X, y
X = main_data[n_features + c_features].copy()
y = main_data['Memory_Bandwidth'].copy()

# --- 1. Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 2. Preprocessor ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop=None, sparse_output=False), categorical_features)
    ],
    remainder='drop'
)
# 3A. Ridge + GridSearchCV
ridge_pipeline = Pipeline([
    ('pre', preprocessor),
    ('reg', Ridge())
])

param_grid_ridge = {
    'reg__alpha': np.logspace(-3, 3, 25)
}

gs_ridge = GridSearchCV(ridge_pipeline, param_grid_ridge,
                        cv=5, scoring='r2', n_jobs=-1)

gs_ridge.fit(X_train, y_train)
ridge_best = gs_ridge.best_estimator_

best_alpha_ridge = gs_ridge.best_params_['reg__alpha']

def adjusted_r2(r2, y_true, feature_count):
    n = len(y_true)
    p = feature_count
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

y_pred_ridge = ridge_best.predict(X_test)

p_ridge = len(ridge_best.named_steps['reg'].coef_)
r2_adj_ridge = adjusted_r2(r2_score(y_test, y_pred_ridge), y_test, p_ridge)

rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

# 3B. LassoCV
lasso_cv = Pipeline([
    ('pre', preprocessor),
    ('reg', LassoCV(cv=5, max_iter=5000))
])

lasso_cv.fit(X_train, y_train)

best_alpha_lasso = lasso_cv.named_steps['reg'].alpha_

y_pred_lasso = lasso_cv.predict(X_test)

r2_lasso = r2_score(y_test, y_pred_lasso)

nonzero = np.sum(np.abs(lasso_cv.named_steps['reg'].coef_) > 1e-8)
r2_adj_lasso = adjusted_r2(r2_lasso, y_test, nonzero)

rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))


# 4. Lấy tên biến sau OneHot
ridge_pre = ridge_best.named_steps['pre']
ohe = ridge_pre.named_transformers_['cat']
ohe_feature_names = ohe.get_feature_names_out(c_features).tolist()

all_feature_names = numeric_features + ohe_feature_names

# Ridge coefficients
ridge_coefs = ridge_best.named_steps['reg'].coef_

# Lasso coefficients
lasso_pre = lasso_cv.named_steps['pre']
lasso_coefs = lasso_cv.named_steps['reg'].coef_

# Kiểm tra biến Lasso giữ lại
lasso_kept = [coef != 0 for coef in lasso_coefs]

# Tạo DataFrame
coef_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Ridge Coef': ridge_coefs,
    'Lasso Coef': lasso_coefs,
    'Lasso Kept': lasso_kept
})

# In ra bảng
#print(coef_df)


# Bảng so sánh kết quả
results_dict = {
    'Model': ['Ridge', 'Lasso'],
    'Best Alpha': [best_alpha_ridge, best_alpha_lasso],
    'R2 Test': [r2_score(y_test, y_pred_ridge), r2_score(y_test, y_pred_lasso)],
    'Adjusted R2': [r2_adj_ridge, r2_adj_lasso],
    'RMSE Test': [rmse_ridge, rmse_lasso]
}

# Tạo DataFrame
results_df = pd.DataFrame(results_dict)

# In ra
#print(results_df)

# Dự đoán trên X_test đã chia trước
predictions = lasso_cv.predict(X_test)

# Tạo DataFrame chứa X_test + dự đoán + giá trị thực
test_results = X_test.copy()
test_results['Memory_Bandwidth'] = y_test.values
test_results['Predicted_Memory_Bandwidth'] = predictions

# Tính RMSE và R2
rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
r2 = r2_score(y_test, predictions)

print(test_results.head())
print(f"RMSE: {rmse:.3f}")
print(f"R-square: {r2:.3f}") 
