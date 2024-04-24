import numpy as np
import pandas as pd
from string import ascii_uppercase

print("Формирование исходного датасета")
date_range = pd.date_range(start='2023-01-01', end='2023-12-31', freq='s')

df = pd.DataFrame({'date': date_range})

df = pd.concat([df, df, df])

li = [i for i in ascii_uppercase]

df['abc'] = np.random.choice(li, size=len(df))
del li
df['abc'] = df['abc'].astype('string')

df['num'] = np.random.randint(1000, size=len(df))
print("Датасет сформирован")
print(df)
df
df.head()



print('''
-----------------------------
считывание и процессинг
''')
print("-удаляем пустые значения...")
df.dropna(inplace=True)

print("-удаляем дубликаты...")
df.drop_duplicates(inplace=True)

print("-удаляем записи в промежутке от 1 до 3 часов ночи")
df = df[(df['date'].dt.hour < 1) | (df['date'].dt.hour > 3)]

print(df)



print("""
------------------
Расчет метрик
""")
print("-Агрегация по времени, для каждого часа рассчитать")
abc_unique = df['abc'].nunique()
print(f"-Рколичество уникальных string = {abc_unique}")

print('''
sql - 
SELECT COUNT(DISTINCT abc) FROM table
''')

print("-среднее и медиану для numeric")
df['hour'] = df['date'].dt.hour
df_agg = df.groupby('hour').agg({'date': 'count', 'num': ['mean', 'median']})
df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
print(df_agg)

print('''
sql - 
SELECT 
    hour,
    COUNT(date),
    AVG(num),
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY num)
FROM df
GROUP BY hour;
''')



print("""
--------------------
мерж с метриками
""")
df_merged = pd.merge(df, df_agg, left_on='hour', right_index=True, how='left')
print("df_merged")

print("""
-------------------------
аналитические метрики
""")
import matplotlib.pyplot as plt
print(df['num'].hist())

print('''
----------------------------------------------
95% доверительный интервал, с комментарием как выбирали методику расчета
''')
lower_bound = df['num'].quantile(0.025)
upper_bound = df['num'].quantile(0.975)
print(f'lower_bound = {lower_bound}')
print(f'upper_bound = {upper_bound}')
print('''
Чтобы построить 95% доверительный интервал для столбца num, я использовал метод quantile() столбца для расчета 2,5-го и 97,5-го процентилей, что даст вам нижнюю и верхнюю границы интервала соответственно.
lower_bound = df['num'].quantile(0.025)
upper_bound = df['num'].quantile(0.975)
''')



print('''
-------------------
визуализация
''')
df['month'] = df['date'].dt.month
avg_num_by_month = df.groupby('month')['num'].mean()

print('-Отрисовать график среднего значения numeric колонки (y) по месяцам (x).')
plt.figure(figsize=(10, 6))
plt.plot(avg_num_by_month.index, avg_num_by_month.values, marker='o')
plt.xlabel('month')
plt.ylabel('Average num')
plt.xticks(range(1, 13))
plt.grid(True)
plt.show()


print('-Heatmap по частотности символов в колонке string')
import seaborn as sns
char_freq = df['abc'].value_counts().reset_index()
char_freq.columns = ['char', 'frequency']

plt.figure(figsize=(10, 6))
heatmap_data = pd.pivot_table(char_freq, values='frequency', index='char', aggfunc='sum')
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='g')
plt.xlabel('Month')
plt.ylabel('Character')
plt.show()



print("""
-------------------------------------
Случайно поделить датасет на 3 части - в одной 25% записей, во второй 25% и 50% в третьей.
""")

print(f"исходный датасет = {df.shape}")
np.random.seed(0)
n = len(df)
split_1 = int(0.25 * n)
split_2 = int(0.50 * n)
split_3 = int(0.75 * n)

df_1 = df.sample(frac=1, random_state=0).head(split_1)
df_2 = df.sample(frac=1, random_state=0).head(split_2)[split_1:split_2]
df_3 = df.sample(frac=1, random_state=0).tail(n - split_2)

print(f"df_1 = {df_1.shape}")
print(f"df_2 = {df_2.shape}")
print(f"df_3 = {df_3.shape}")