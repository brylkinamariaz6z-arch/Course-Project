import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
plt.style.use('default')
# загрузка данных
df = pd.read_csv('train.csv')
print("Размер датасета:")
print(df.shape)
print("\nПервые строки:")
print(df.head())
print("\nИнформация о датасете:")
print(df.info())
print("Количество строк:", df.shape[0])
print("Количество признаков:", df.shape[1])
print("\nТипы данных:")
print(df.dtypes)
print("\nКоличество пропусков:")
print(df.isnull().sum())
numeric_features = [
    'Age',
    'RoomService',
    'FoodCourt',
    'ShoppingMall',
    'Spa',
    'VRDeck'
]
categorical_features = [
    'HomePlanet',
    'CryoSleep',
    'Cabin',
    'Destination',
    'VIP'
]
target = 'Transported'
fig, axes = plt.subplots(
    2,
    3,
    figsize=(15,8)
)
for i, col in enumerate(numeric_features):
    row = i // 3
    col_id = i % 3
    axes[row, col_id].hist(
        df[col].dropna(),
        bins=30,
        edgecolor='black'
    )
    axes[row, col_id].set_title(col)
plt.tight_layout()
plt.show()
plt.figure(figsize=(6,4))
df['Transported'].value_counts().plot(
    kind='bar'
)
plt.title('Transported')
plt.show()
plt.figure(figsize=(8,6))
sns.boxplot(
    x='Transported',
    y='Age',
    data=df
)
plt.title('Age vs Transported')
plt.show()
plt.figure(figsize=(8,6))
sns.countplot(
    x='HomePlanet',
    hue='Transported',
    data=df
)
plt.title('HomePlanet vs Transported')
plt.show()
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x='Age',
    y='FoodCourt',
    hue='Transported',
    alpha=0.6
)
plt.title('Age vs FoodCourt')
plt.show()
fig = px.scatter(
    df,
    x='Age',
    y='RoomService',
    color='Transported',
    hover_data=['PassengerId'],
    title='Age vs RoomService'
)
fig.show()
fig = px.scatter(
    df,
    x='FoodCourt',
    y='Spa',
    color='Transported',
    hover_data=['PassengerId'],
    title='FoodCourt vs Spa'
)
fig.show()
missing_count = df.isnull().sum()
missing_percent = (
    missing_count / len(df)
) * 100
missing_table = pd.DataFrame({
    'Количество пропусков': missing_count,
    'Процент пропусков': missing_percent
})
missing_table = missing_table[
    missing_table['Количество пропусков'] > 0
].sort_values(
    by='Количество пропусков',
    ascending=False
)
print(missing_table)
df_clean = df.copy()
expense_cols = [
    'RoomService',
    'FoodCourt',
    'ShoppingMall',
    'Spa',
    'VRDeck'
]
cryo_mask = df_clean['CryoSleep'] == True
for col in expense_cols:
    df_clean.loc[
        cryo_mask,
        col
    ] = df_clean.loc[
        cryo_mask,
        col
    ].fillna(0)
for col in expense_cols:
    median_value = df_clean.loc[
        ~cryo_mask,
        col
    ].median()
    df_clean[col] = df_clean[col].fillna(
        median_value
    )
df_clean['Age'] = df_clean['Age'].fillna(
    df_clean['Age'].median()
)
cat_cols = [
    'HomePlanet',
    'Destination',
    'VIP',
    'CryoSleep'
]
for col in cat_cols:
    mode_value = df_clean[col].mode()[0]
    df_clean[col] = df_clean[col].fillna(
        mode_value
    )
cabin_split = df_clean[
    'Cabin'
].str.split(
    '/',
    expand=True
)
cabin_split.columns = [
    'Deck',
    'CabinNum',
    'Side'
]
df_clean['Deck'] = cabin_split['Deck']
df_clean['CabinNum'] = cabin_split['CabinNum']
df_clean['Side'] = cabin_split['Side']
df_clean['Deck'] = df_clean['Deck'].fillna(
    df_clean['Deck'].mode()[0]
)
df_clean['Side'] = df_clean['Side'].fillna(
    df_clean['Side'].mode()[0]
)
print(
    "Количество пропусков после обработки:"
)
print(
    df_clean.isnull().sum()
)
plt.figure(figsize=(12,6))
sns.heatmap(
    df.isnull(),
    yticklabels=False,
    cbar=True,
    cmap='viridis'
)
plt.title(
    'Тепловая карта пропусков'
)
plt.show()
corr_matrix = df_clean[
    numeric_features
].corr()
plt.figure(figsize=(8,6))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm'
)
plt.title(
    'Корреляционная матрица'
)
plt.show()
duplicates = df_clean.duplicated().sum()
print(
    "Количество дубликатов:",
    duplicates
)
if duplicates > 0:

    df_clean = df_clean.drop_duplicates()

    print(
        "Дубликаты удалены."
    )
else:
    print(
        "Дубликаты отсутствуют."
    )
fig, axes = plt.subplots(
    2,
    3,
    figsize=(15,8)
)
for i, col in enumerate(
    numeric_features
):
    row = i // 3
    col_id = i % 3
    sns.boxplot(
        y=df_clean[col],
        ax=axes[row, col_id]
    )
    axes[row, col_id].set_title(col)
plt.tight_layout()
plt.show()
def count_outliers_iqr(
    data,
    column
):
    q1 = data[column].quantile(
        0.25
    )
    q3 = data[column].quantile(
        0.75
    )
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = data[
        (data[column] < lower)
        |
        (data[column] > upper)
    ]
    return len(outliers)
for col in numeric_features:

    print(
        col,
        ":",
        count_outliers_iqr(
            df_clean,
            col
        )
    )
earth_passengers = df_clean[
    df_clean['HomePlanet']
    == 'Earth'
]
print(
    len(earth_passengers)
)
cryo_transported = df_clean[
    (df_clean['CryoSleep'] == True)
    &
    (df_clean['Transported'] == True)
]
print(
    len(cryo_transported)
)
high_spend = df_clean[
    (
        df_clean['RoomService']
        +
        df_clean['FoodCourt']
        +
        df_clean['ShoppingMall']
        +
        df_clean['Spa']
        +
        df_clean['VRDeck']
    ) > 10000
]
print(
    len(high_spend)
)
np.random.seed(42)
df_clean['Age_noisy'] = (
    df_clean['Age']
    +
    np.random.normal(
        0,
        3,
        len(df_clean)
    )
)
df_clean['FoodCourt_noisy'] = (
    df_clean['FoodCourt']
    +
    np.random.normal(
        0,
        5,
        len(df_clean)
    )
)
df_clean[
    'FoodCourt_noisy'
] = df_clean[
    'FoodCourt_noisy'
].clip(
    lower=0
)
df_clean[
    [
        'Age',
        'Age_noisy',
        'FoodCourt',
        'FoodCourt_noisy'
    ]
].head()
def age_group(age):
    if age <= 12:
        return 'Child'
    elif age <= 18:
        return 'Teen'
    elif age <= 40:
        return 'Adult'
    elif age <= 60:
        return 'Middle'
    else:
        return 'Senior'
df_clean['Age_group'] = df_clean[
    'Age'
].apply(age_group)
age_distribution = pd.DataFrame({
    'Количество':
        df_clean[
            'Age_group'
        ].value_counts(),
    'Процент':
        round(
            df_clean[
                'Age_group'
            ].value_counts(
                normalize=True
            ) * 100,
            2
        )
})
print(age_distribution)
plt.figure(figsize=(8,5))
sns.countplot(
    data=df_clean,
    x='Age_group',
    order=[
        'Child',
        'Teen',
        'Adult',
        'Middle',
        'Senior'
    ]
)
plt.title(
    'Возрастные группы'
)
plt.show()
binary_cols = [
    'CryoSleep',
    'VIP',
    'Transported'
]
for col in binary_cols:
    df_clean[col] = df_clean[
        col
    ].map({
        True: 1,
        False: 0

    })
df_clean['Side'] = df_clean[
    'Side'
].map({
    'P': 0,
    'S': 1
})
df_clean = pd.get_dummies(
    df_clean,
    columns=[
        'HomePlanet',
        'Destination',
        'Deck'
    ],
    prefix=[
        'HP',
        'Dest',
        'Deck'
    ]
)
print(
    df_clean.dtypes.value_counts()
)
filtered = df[
    (df['CryoSleep'] == True)
    &
    (df['Transported'] == True)
]
all_stats = df[
    'Age'
].describe()
filtered_stats = filtered[
    'Age'
].describe()
comparison = pd.DataFrame({
    'Все пассажиры':
        all_stats,
    'Криосон и перемещённые':
        filtered_stats
})
print(comparison)
plt.figure(figsize=(10,6))
sns.histplot(
    df['Age'],
    bins=30,
    alpha=0.5,
    label='Все'
)
sns.histplot(
    filtered['Age'],
    bins=30,
    alpha=0.5,
    label='Криосон + перемещённые'
)
plt.legend()
plt.title(
    'Сравнение распределений возраста'
)
plt.show()
group_stats = df.groupby(
    'HomePlanet'
).agg({
    'Age': 'mean',
    'RoomService': 'mean',
    'FoodCourt': 'mean'
}).round(2)
print(group_stats)
categorical_columns = [
    'HomePlanet',
    'Destination',
    'CryoSleep',
    'VIP'
]
for col in categorical_columns:
    print(
        f"\n{col}"
    )
    print(
        df[col].value_counts(
            dropna=False
        )
    )
plt.figure(figsize=(8,5))
sns.countplot(
    data=df,
    x='HomePlanet'
)
plt.title(
    'Распределение HomePlanet'
)
plt.show()
plt.figure(figsize=(8,5))
sns.countplot(
    data=df,
    x='Destination'
)
plt.title(
    'Распределение Destination'
)
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(
    data=df,
    x='CryoSleep'
)
plt.title(
    'Распределение CryoSleep'
)
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(
    data=df,
    x='VIP'
)
plt.title(
    'Распределение VIP'
)
plt.show()
deck_data = df[
    'Cabin'
].str.split(
    '/',
    expand=True
)
df['Deck'] = deck_data[0]
plt.figure(figsize=(10,5))
sns.countplot(
    data=df,
    x='Deck'
)
plt.title(
    'Распределение палуб'
)
plt.show()
encoded_columns = [
    col
    for col in df_clean.columns
    if col.startswith('HP_')
    or col.startswith('Dest_')
    or col.startswith('Deck_')
]
print(encoded_columns)
deck_counts = df[
    'Deck'
].value_counts()
deck_percent = round(
    deck_counts
    /
    len(df)
    *
    100,
    3
)
rare_categories = pd.DataFrame({
    'Количество':
        deck_counts,
    'Процент':
        deck_percent
})
print(rare_categories)
planet_stats = df.groupby(
    'HomePlanet'
).agg(
    Avg_Age=(
        'Age',
        'mean'
    ),
    Avg_FoodCourt=(
        'FoodCourt',
        'mean'
    ),
    Transported_percent=(
        'Transported',
        lambda x:
            (
                x == True
            ).mean()
            *
            100
    )
).round(2)
print(planet_stats)
df['TotalSpend'] = (
    df['RoomService'].fillna(0)
    +
    df['FoodCourt'].fillna(0)
    +
    df['ShoppingMall'].fillna(0)
    +
    df['Spa'].fillna(0)
    +
    df['VRDeck'].fillna(0)

)
def spending_category(x):
    if x == 0:
        return 'Low'
    elif x <= 5000:
        return 'Medium'
    else:
        return 'High'
df['Spending_activity'] = df[
    'TotalSpend'
].apply(
    spending_category
)
spending_stats = df.groupby(
    'Spending_activity'
).agg(
    Count=(
        'PassengerId',
        'count'
    ),
    Transported_percent=(
        'Transported',
        lambda x:
            (
                x == True
            ).mean()
            *
            100

    )
).round(2)
print(spending_stats)
plt.figure(figsize=(8,5))
sns.countplot(
    data=df,
    x='Spending_activity',
    order=[
        'Low',
        'Medium',
        'High'
    ]
)
plt.title(
    'Уровень потребительской активности'
)
plt.show()
df_clean.to_csv(
    'spaceship_titanic_processed.csv',
    index=False
)