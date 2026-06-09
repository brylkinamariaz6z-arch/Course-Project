import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from wordcloud import WordCloud
import pymorphy3
from sklearn.feature_extraction.text import TfidfVectorizer

# ========== 1. ЗАГРУЗКА И ПЕРВИЧНЫЙ ОСМОТР ==========
df = pd.read_csv('opensentimentcorpus-final-by-majority.csv')
print(f"Всего записей: {len(df)}")
print(f"Колонки: {df.columns.tolist()}")
print(f"Пропуски:\n{df.isnull().sum()}")
print("\nПервые 5 строк:")
print(df.head())

# ========== 2. РАСПРЕДЕЛЕНИЕ КЛАССОВ ==========
sentiment_counts = df['Sentiment'].value_counts()
print("\nРаспределение классов:")
print(sentiment_counts)
print("\nВ процентах:")
for label, count in sentiment_counts.items():
    percent = count / len(df) * 100
    print(f"  {label}: {percent:.1f}%")

# Гистограмма
plt.figure(figsize=(8,5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Распределение тональностей в датасете')
plt.ylabel('Количество примеров')
plt.xlabel('Sentiment')
for i, v in enumerate(sentiment_counts.values):
    plt.text(i, v+20, str(v), ha='center')
plt.savefig('sentiment_distribution.png', dpi=150)
plt.show()

# ========== 3. ПРИМЕРЫ ПО КЛАССАМ ==========
print("\n=== Примеры positive ===")
print(df[df['Sentiment'] == 'positive']['Sentence'].head(3).to_string(index=False))
print("\n=== Примеры negative ===")
print(df[df['Sentiment'] == 'negative']['Sentence'].head(3).to_string(index=False))
print("\n=== Примеры neutral ===")
print(df[df['Sentiment'] == 'neutral']['Sentence'].head(3).to_string(index=False))
print("\n=== Примеры mixed ===")
print(df[df['Sentiment'] == 'mixed']['Sentence'].head(3).to_string(index=False))

# ========== 4. ОЧИСТКА И ЛЕММАТИЗАЦИЯ ==========
morph = pymorphy3.MorphAnalyzer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)   # удаляем пунктуацию
    text = re.sub(r'\d+', ' ', text)       # удаляем цифры
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize(text):
    words = text.split()
    lemmatized = [morph.parse(word)[0].normal_form for word in words if len(word) > 2]
    return ' '.join(lemmatized)

print("\nПрименяем очистку и лемматизацию...")
df['clean_text'] = df['Sentence'].apply(clean_text)
df['lemm_text'] = df['clean_text'].apply(lemmatize)

# ========== 5. ЧАСТОТА СЛОВ (БЕЗ СТОП-СЛОВ) ==========
# Простые стоп-слова для русского языка
stop_words = {'и', 'в', 'на', 'не', 'с', 'а', 'по', 'к', 'у', 'о', 'из', 'за', 'от', 'до', 'во',
              'это', 'что', 'как', 'так', 'все', 'было', 'но', 'или', 'же', 'бы', 'для', 'без',
              'еще', 'уже', 'только', 'очень', 'даже', 'есть', 'быть', 'весь', 'такой', 'тот',
              'свой', 'этот', 'который', 'там', 'тут', 'тогда', 'потом', 'вот', 'ведь', 'вдруг'}

all_words = []
for text in df['lemm_text']:
    words = text.split()
    all_words.extend([w for w in words if w not in stop_words and len(w) > 2])

word_freq = Counter(all_words)
top15 = word_freq.most_common(15)
print("\nТоп-15 самых частотных слов:")
for word, freq in top15:
    print(f"{word}: {freq}")

# График топ-15
plt.figure(figsize=(10,6))
words, freqs = zip(*top15)
sns.barplot(x=list(words), y=list(freqs), palette='coolwarm')
plt.xticks(rotation=45)
plt.title('Топ-15 самых частотных слов (после лемматизации и удаления стоп-слов)')
plt.savefig('top15_words.png', dpi=150)
plt.show()

# ========== 6. ОБЛАКА СЛОВ ==========
def generate_wordcloud(text_series, title, filename):
    all_text = ' '.join(text_series)
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          colormap='viridis', max_words=100, stopwords=stop_words,
                          contour_width=1, contour_color='steelblue').generate(all_text)
    plt.figure(figsize=(10,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig(filename, dpi=150)
    plt.show()

generate_wordcloud(df['lemm_text'], 'Облако слов – весь корпус', 'wordcloud_all.png')
for sentiment in ['positive', 'negative', 'neutral', 'mixed']:
    subset = df[df['Sentiment'] == sentiment]['lemm_text']
    if len(subset) > 0:
        generate_wordcloud(subset, f'Облако слов – {sentiment}', f'wordcloud_{sentiment}.png')

# ========== 7. TF-IDF ПРЕОБРАЗОВАНИЕ (раздел 4.3.6) ==========
print("\nВычисляем TF-IDF матрицу...")
tfidf = TfidfVectorizer(max_features=5000, stop_words=list(stop_words))
tfidf_matrix = tfidf.fit_transform(df['lemm_text'])
print(f"Размер TF-IDF матрицы: {tfidf_matrix.shape}")
print(f"Количество признаков (уникальных термов): {len(tfidf.get_feature_names_out())}")
text_length = df["Sentence"].str.split().str.len()

print(text_length.describe())

plt.figure(figsize=(8,5))
plt.hist(text_length, bins=30)
plt.xlabel("Количество слов")
plt.ylabel("Частота")
plt.title("Распределение длины предложений")
plt.grid(True)
plt.show()
print("\nАнализ завершён. Все графики сохранены в текущую папку.")