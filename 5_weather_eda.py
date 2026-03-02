#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
#%%
### 분류모형에 사용할 데이터 EDA
df = pd.read_csv("./data/cls_train_data.csv", parse_dates=['일시'])
df.head()

num_cols = list(df.select_dtypes(include='number').columns) # 수치형 변수
num_cols.remove('target')
obj_cols = list(df.select_dtypes(include=['object', 'string']).columns) # 문자형 변수
#%%
# 기초통계량 확인하기 (수치형 변수에만 적용 가능)
print(df[num_cols].describe())
#%%
print("발생횟수:\n", df['target'].value_counts().sort_index())
print("발생비율:\n", df['target'].value_counts(normalize=True).sort_index())

value_counts = df['target'].value_counts().sort_index()
value_ratio = df['target'].value_counts(normalize=True).sort_index()
categories = value_counts.index.astype(str)
counts = value_counts.values
ratio = value_ratio.values

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True)
axes[0].bar(categories, counts)
axes[0].set_xlabel("발생여부", fontsize=14)
axes[0].set_ylabel("횟수", fontsize=14)
axes[0].grid(alpha=0.3)
axes[1].bar(categories, ratio)
axes[1].set_xlabel("발생여부", fontsize=14)
axes[1].set_ylabel("비율", fontsize=14)
axes[1].grid(alpha=0.3)
for ax in axes:
    ax.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.savefig("./fig/5_weather_target.png")
plt.show()
plt.close()
#%%
### target group별 기초 통계량
group_stats = df.groupby('target')[num_cols].agg(['mean', 'median', 'max'])
print(group_stats)
#%%
### 분포의 차이를 확인
col_min = '최저기온(°C)(lag3)'
col_max = '평균 풍속(m/s)(lag1)'

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df.boxplot(
    column=col_min,
    by='target',
    grid=True,
    
    boxprops=dict(linewidth=2, color='tab:blue'), # 박스
    whiskerprops=dict(linewidth=2, color='tab:blue'), # 수염
    capprops=dict(linewidth=2, color='black'), # 최상(하)단 가로선
    medianprops=dict(linewidth=2.5, color='red'), # 중위수
    ax=axes[0]
)
axes[0].set_xlabel("발생여부", fontsize=14)
axes[0].set_ylabel(f"{col_min}", fontsize=14)
axes[0].grid(alpha=0.3)
axes[0].set_title("")
df.boxplot(
    column=col_max,
    by='target',
    grid=True,
    
    boxprops=dict(linewidth=2, color='tab:blue'), # 박스
    whiskerprops=dict(linewidth=2, color='tab:blue'), # 수염
    capprops=dict(linewidth=2, color='black'), # 최상(하)단 가로선
    medianprops=dict(linewidth=2.5, color='red'), # 중위수
    ax=axes[1]
)
axes[1].set_xlabel("발생여부", fontsize=14)
axes[1].set_ylabel(f"{col_max}", fontsize=14)
axes[1].grid(alpha=0.3)
axes[1].set_title("")
plt.suptitle("")
plt.tight_layout()
plt.savefig("./fig/5_weather_boxplot.png")
plt.show()
plt.close()
#%%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(
    df[df['target'] == 0][col_min], 
    bins='auto', alpha=0.5, label="Class 0")
axes[0].hist(
    df[df['target'] == 1][col_min], 
    bins='auto', alpha=0.5, label="Class 1")
axes[0].set_xlabel("")
axes[0].set_ylabel(f"{col_min}", fontsize=14)
axes[0].grid(alpha=0.3)
axes[0].set_title(f"발생여부에 따른 {col_min}의 분포 - 히스토그램", fontsize=15)
axes[0].legend(fontsize=13)

axes[1].hist(
    df[df['target'] == 0][col_max], 
    bins='auto', alpha=0.5, label="Class 0")
axes[1].hist(
    df[df['target'] == 1][col_max], 
    bins='auto', alpha=0.5, label="Class 1")
axes[1].set_xlabel("")
axes[1].set_ylabel(f"{col_max}", fontsize=14)
axes[1].grid(alpha=0.3)
axes[1].set_title(f"발생여부에 따른 {col_max}의 분포 - 히스토그램", fontsize=15)
axes[1].legend(fontsize=13)

plt.tight_layout()
plt.savefig("./fig/5_weather_hist.png")
plt.show()
plt.close()
#%%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(
    df[df['target'] == 0][col_min], 
    bins='auto', alpha=0.5, label="Class 0", density=True)
axes[0].hist(
    df[df['target'] == 1][col_min], 
    bins='auto', alpha=0.5, label="Class 1", density=True)
axes[0].set_xlabel("")
axes[0].set_ylabel(f"{col_min}", fontsize=14)
axes[0].grid(alpha=0.3)
axes[0].set_title(f"발생여부에 따른 {col_min}의 분포 - 밀도함수", fontsize=15)
axes[0].legend(fontsize=13)

axes[1].hist(
    df[df['target'] == 0][col_max], 
    bins='auto', alpha=0.5, label="Class 0", density=True)
axes[1].hist(
    df[df['target'] == 1][col_max], 
    bins='auto', alpha=0.5, label="Class 1", density=True)
axes[1].set_xlabel("")
axes[1].set_ylabel(f"{col_max}", fontsize=14)
axes[1].grid(alpha=0.3)
axes[1].set_title(f"발생여부에 따른 {col_max}의 분포 - 밀도함수", fontsize=15)
axes[1].legend(fontsize=13)

plt.tight_layout()
plt.savefig("./fig/5_weather_density.png")
plt.show()
plt.close()
#%%
plt.figure(figsize=(8, 10))
df[num_cols].boxplot(
    vert=False,
    showfliers=False
)
plt.title("기상변수들의 분포 범위", fontsize=15)
plt.xlabel("값")
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("./fig/5_weather_weather_range.png")
plt.show()
plt.close()
#%%
### 변수들간의 강한 상관관계 구조 확인
corr = df[num_cols].corr()

plt.figure(figsize=(7, 6))
plt.imshow(corr, aspect="auto")
plt.title("상관계수 행렬", fontsize=16)
plt.colorbar()
plt.tight_layout()
plt.savefig("./fig/5_weather_corrmat.png")
plt.show()
plt.close()
#%%