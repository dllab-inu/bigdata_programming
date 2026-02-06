#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
#%%
### 회귀모형에 사용할 데이터 EDA
df = pd.read_csv("./data/reg_data.csv", parse_dates=['일시', '발생일시', '진화종료시간'])
df.head()

num_cols = df.select_dtypes(include='number').columns
obj_cols = df.select_dtypes(include=['object', 'string']).columns
#%%
# 기초통계량 확인하기 (수치형 변수에만 적용 가능)
df[num_cols].describe()
#%%
# df['피해면적_합계'].hist() # 가장 심플한 방법

x = df['피해면적_합계']

plt.figure(figsize=(6, 4))
plt.hist(x, bins=15, edgecolor='black', alpha=0.7)
plt.axvline(x.mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {x.mean():.2f}")
plt.axvline(x.median(), color='blue', linestyle='-', linewidth=2, label=f"Median: {x.median():.2f}")
plt.xlabel("피해면적(ha)", fontsize=15)
plt.ylabel("빈도", fontsize=15)
plt.xticks(np.linspace(0, 5, 11), fontsize=14)
plt.yticks(fontsize=14)
plt.title("산불 피해면적 분포 (히스토그램)", fontsize=16)
plt.legend(fontsize=15)
plt.grid(axis='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("./fig/4_sanbul_histogram.png")
plt.show()
plt.close()
#%%
x = np.log(df['피해면적_합계']) # log(x) 변환
# x = np.log1p(df['피해면적_합계']) # log(1+x) 변환 (0을 0으로 변환)

### 회귀분석에서 log(target)의 형태를 고려 가능!
plt.figure(figsize=(6, 4))
plt.hist(x, bins=15, edgecolor='black', alpha=0.7)
plt.axvline(x.mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {x.mean():.2f}")
plt.axvline(x.median(), color='blue', linestyle='-', linewidth=2, label=f"Median: {x.median():.2f}")
plt.xlabel("log(피해면적(ha))", fontsize=15)
plt.ylabel("빈도", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("산불 피해면적 분포 (로그변환 히스토그램)", fontsize=16)
plt.legend(fontsize=15)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("./fig/4_sanbul_log_histogram.png")
plt.show()
plt.close()
#%%
df.groupby('발생원인_구분')['피해면적_합계'] # [주의] 데이터가 아님!

for name, group in df.groupby('발생원인_구분'):
    print(f"\n발생원인_구분: {name}")
    print(group['피해면적_합계'].values)

df.groupby('발생원인_구분')['피해면적_합계'].count().sort_values(ascending=False)

df.groupby('발생원인_구분')['피해면적_합계'].sum().sort_values(ascending=False)

df.groupby('발생원인_구분')['피해면적_합계'].mean().sort_values(ascending=False)

df.groupby('발생원인_구분')['피해면적_합계'].median().sort_values(ascending=False)

df.groupby('발생원인_구분')['피해면적_합계'].max().sort_values(ascending=False)

cause_summary = (
    df.groupby('발생원인_구분')['피해면적_합계']
        .agg(['count', 'sum', 'mean', 'median', 'max'])
).reset_index()
print(cause_summary.round(3))
#%%
plt.figure(figsize=(6, 4))
df.boxplot(
    column='피해면적_합계',
    by='발생원인_구분',
    grid=True,
    
    boxprops=dict(linewidth=2, color='tab:blue'), # 박스
    whiskerprops=dict(linewidth=2, color='tab:blue'), # 수염
    capprops=dict(linewidth=2, color='black'), # 최상(하)단 가로선
    medianprops=dict(linewidth=2.5, color='red') # 중위수
)
plt.xticks(rotation=45, fontsize=14)
plt.title("발생원인별 피해면적 분포", fontsize=16)
plt.suptitle("")
plt.xlabel("발생원인", fontsize=15)
plt.ylabel("피해면적(ha)", fontsize=15)
plt.tight_layout()
plt.savefig("./fig/4_sanbul_cause_boxplot.png")
plt.show()
plt.close()
#%%
df['log_피해면적'] = np.log(df['피해면적_합계'])

plt.figure(figsize=(6, 4))
df.boxplot(
    column='log_피해면적',
    by='발생원인_구분',
    grid=True,

    boxprops=dict(linewidth=2, color='tab:blue'), # 박스
    whiskerprops=dict(linewidth=2, color='tab:blue'), # 수염
    capprops=dict(linewidth=2, color='black'), # 최상(하)단 가로선
    medianprops=dict(linewidth=2.5, color='red') # 중위수
)
plt.xticks(rotation=45, fontsize=14)
plt.title("발생원인별 log(피해면적) 분포", fontsize=16)
plt.suptitle("")
plt.xlabel("발생원인", fontsize=15)
plt.ylabel("log(피해면적(ha))", fontsize=15)
plt.tight_layout()
plt.savefig("./fig/4_sanbul_cause_log_boxplot.png")
plt.show()
plt.close()
#%%
df['월'] = df['일시'].dt.month

plt.figure(figsize=(7, 4))
df.boxplot(
    column='log_피해면적',
    by='월',
    grid=True,
    showfliers=True,

    boxprops=dict(linewidth=1.8),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    medianprops=dict(color='red', linewidth=2)
)
plt.title("월별 log(피해면적) 분포", fontsize=16)
plt.suptitle("")
plt.xlabel("월", fontsize=15)
plt.ylabel("log(피해면적(ha))", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("./fig/4_sanbul_month_boxplot.png")
plt.show()
plt.close()
#%%
### 진화소요시간 (파생변수)
# [주의] 설명변수로는 사용할 수 없는 변수
df['진화소요시간_분'] = (df['진화종료시간'] - df['발생일시']).dt.total_seconds() / 60
df['진화소요시간_분'].hist()
np.log(df['진화소요시간_분']).hist()
np.exp(4.5)
#%%
weather_cols = sorted(list(set([x.split('(lag')[0] for x in num_cols if 'lag' in x])))

# for col in weather_cols:
col = weather_cols[0] # 최고기온
lag = 2

fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True)
# 원본
ax[0].scatter(df[f'{col}(lag{lag})'], df['피해면적_합계'], alpha=0.6)
ax[0].set_title("원본", fontsize=16)
ax[0].set_xlabel(f"{col}(lag{lag})", fontsize=15)
ax[0].set_ylabel("피해면적(ha)", fontsize=15)
ax[0].grid(alpha=0.3)
# 로그변환
ax[1].scatter(df[f'{col}(lag{lag})'], df['log_피해면적'], alpha=0.6)
ax[1].set_title("로그변환", fontsize=16)
ax[1].set_xlabel(f"{col}(lag{lag})", fontsize=15)
ax[1].set_ylabel("log(피해면적(ha))", fontsize=15)
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("./fig/4_sanbul_scatter_plot.png")
plt.show()
plt.close()
#%%
weather_lag_cols = [x for x in num_cols if 'lag' in x]

color_map = {
    '평균기온(°C)': 'tab:blue',
    '최고기온(°C)': 'tab:blue',
    '최저기온(°C)': 'tab:blue',
    '평균 상대습도(%)': 'tab:orange',
    '합계 일조시간(hr)': 'tab:green',
    '평균 풍속(m/s)': 'tab:red'
}
corr_log = (
    df[weather_lag_cols + ['log_피해면적']].corr()['log_피해면적']
        .drop('log_피해면적')
        .sort_values(key=abs, ascending=False)
)
colors = [color_map.get(c.split('(lag')[0]) for c in corr_log.index]

plt.figure(figsize=(6, 12))
corr_log.plot(kind='barh', color=colors, fontsize=13)
plt.title("피해면적과의 상관계수 (절댓값 기준 정렬)", fontsize=15)
plt.grid(axis='x', alpha=0.5, linestyle='--')
plt.tight_layout()
plt.savefig("./fig/4_sanbul_corr.png")
plt.show()
plt.close()
#%%
col = weather_cols[0] # 최고기온
lag_effect = []
for i in range(1, 8):
    corr = df[[f'{col}(lag{i})', 'log_피해면적']].corr().iloc[0, 1]
    lag_effect.append(corr)
plt.figure(figsize=(6, 4))
plt.plot(range(1, 8), lag_effect, marker='o')
plt.gca().invert_xaxis() # x축 방향 전환
plt.xlabel("days ago", fontsize=15)
plt.ylabel("상관계수", fontsize=15)
plt.title(f"{col}", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig("./fig/4_sanbul_corr_lag.png")
plt.show()
plt.close()
#%%
plt.figure(figsize=(8, 10))
df[weather_lag_cols].boxplot(
    vert=False,
    showfliers=False
)
plt.title("기상변수들의 분포 범위", fontsize=15)
plt.xlabel("값")
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("./fig/4_sanbul_weather_range.png")
plt.show()
plt.close()
#%%