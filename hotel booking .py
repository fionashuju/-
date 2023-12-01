#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
# 加载数据
hotel_reservations = pd.read_csv("/Users/wufan/Desktop/Hotel Reservations.csv")

# 计算描述性统计
descriptive_stats = hotel_reservations.describe()
descriptive_stats = descriptive_stats.T
descriptive_stats.to_excel("/Users/wufan/Desktop/Hotel Reservations.xlsx")

# 包括分类变量的描述性统计
descriptive_stats_incl_objects = hotel_reservations.describe(include='object')
descriptive_stats_incl_objects = descriptive_stats_incl_objects.T
descriptive_stats_incl_objects.to_excel("/Users/wufan/Desktop/Hotel Reservations.xlsx")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
hotel_reservations = pd.read_csv("/Users/wufan/Desktop/Hotel Reservations.csv")

# 设置 matplotlib 的字体，以确保中文能正确显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 'SimHei' 字体（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置图形风格
# sns.set(style="whitegrid")

# 可视化预订取消状态的分布
plt.figure(figsize=(10, 6))
sns.countplot(x='booking_status', data=hotel_reservations)
plt.title('预订取消状态分布')
plt.savefig(r"D:\he\375-酒店预定预测\预订取消状态分布.png")
plt.show()

# 可视化每月预订数量
plt.figure(figsize=(14, 7))
sns.countplot(x='arrival_month', data=hotel_reservations)
plt.title('每月预订数量')
plt.savefig(r"D:\he\375-酒店预定预测\每月预订数量.png")
plt.show()

# 查看成人数量和预订取消状态之间的关系
plt.figure(figsize=(10, 6))
sns.boxplot(x='booking_status', y='no_of_adults', data=hotel_reservations)
plt.title('成人数量和预订取消状态的关系')
plt.savefig(r"D:\he\375-酒店预定预测\成人数量和预订取消状态的关系.png")
plt.show()

import pandas as pd

# 加载数据
hotel_reservations = pd.read_csv("/Users/wufan/Desktop/Hotel Reservations.csv")

# 查看每列的缺失值数量
missing_values_before = hotel_reservations.isnull().sum()
print("缺失值情况（处理前）:", missing_values_before)

# 删除含有缺失值的记录
hotel_reservations_cleaned = hotel_reservations.dropna()

# 再次检查缺失值
missing_values_after = hotel_reservations_cleaned.isnull().sum()
print("缺失值情况（处理后）:", missing_values_after)

# 定义一个函数来识别和处理异常值
def remove_outliers(df, column_list):
    for column in column_list:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 删除异常值
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df

# 应用函数移除指定列的异常值
numeric_columns = hotel_reservations_cleaned.select_dtypes(include=['int64', 'float64']).columns
hotel_reservations_cleaned = remove_outliers(hotel_reservations_cleaned, numeric_columns)

# 检查数据
print(hotel_reservations_cleaned.describe())


# 识别分类变量列，除了 'Booking_ID'
categorical_columns = hotel_reservations_cleaned.select_dtypes(include=['object']).columns
categorical_columns = categorical_columns.drop('Booking_ID')

print()


# 对分类变量进行独热编码
hotel_reservations_encoded = pd.get_dummies(hotel_reservations_cleaned, columns=categorical_columns)

# 输出最终数据类型
final_data_types = hotel_reservations_encoded.dtypes
print("最终数据类型:", final_data_types)

# 将处理后的数据保存到Excel文件
output_excel_path = r"D:\he\375-酒店预定预测\Cleaned_Hotel_Reservations.xlsx"
hotel_reservations_encoded.to_excel(output_excel_path, index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 加载数据
hotel_reservations = pd.read_excel(r"D:\he\375-酒店预定预测\Cleaned_Hotel_Reservations.xlsx")

# 最后一列是因变量
X = hotel_reservations.iloc[:, :-1]
y = hotel_reservations.iloc[:, -1]

# 划分数据集为训练集和测试集 (9:1 比例)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 初始化模型
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC()
}

# 训练并评估模型
best_model, best_acc = None, 0
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 打印每个模型的参数及准确率
    print(f"{name} 模型参数: {model.get_params()}")
    print(f"{name} 准确率: {acc}\n")

    # 更新最佳模型
    if acc > best_acc:
        best_acc = acc
        best_model = model
        
# 打印最优模型的参数及准确率
if best_model:
    print(f"最优模型: {best_model.__class__.__name__}")
    print(f"最优模型参数: {best_model.get_params()}")
    print(f"最优模型准确率: {best_acc}")
    
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 加载数据
hotel_reservations = pd.read_excel("/Users/wufan/Desktop/数据分析文件/Cleaned_Hotel_Reservations.xlsx")

# 最后一列是因变量
X = hotel_reservations.iloc[:, :-1]
y = hotel_reservations.iloc[:, -1]

# 划分数据集为训练集和测试集 (9:1 比例)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 初始化随机森林模型
random_forest_model = RandomForestClassifier()

# 训练随机森林模型
random_forest_model.fit(X_train, y_train)

# 预测测试集
y_pred = random_forest_model.predict(X_test)

# 计算准确率
acc = accuracy_score(y_test, y_pred)
print(f"RandomForest 准确率: {acc}\n")

# 获取随机森林模型的特征重要性
feature_importances = random_forest_model.feature_importances_

# 将特征重要性与特征名称结合，并按重要性排序
sorted_indices = feature_importances.argsort()[::-1]
sorted_feature_importances = feature_importances[sorted_indices]
sorted_features = X.columns[sorted_indices]

# 绘制特征重要性图
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(len(sorted_feature_importances)), sorted_feature_importances, align='center')
plt.xticks(range(len(sorted_features)), sorted_features, rotation=90)
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlap
plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
