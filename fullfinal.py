import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import nltk
import contextlib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# --- Suppress NLTK download messages ---
with open(os.devnull, 'w') as fnull:
    with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
        nltk.download('punkt')
        nltk.download('stopwords')

# --- Load dataset ---
file_path = "daily_tweets.csv"
if not os.path.exists(file_path):
    print(f"\n❌ Error: The file '{file_path}' was not found.")
    print("🔎 Make sure the file is in the same folder as this script.")
    exit()

df = pd.read_csv(file_path)

# --- Identify tweet text column ---
if 'Tweet' in df.columns:
    tweet_column = 'Tweet'
elif 'text' in df.columns:
    tweet_column = 'text'
elif 'content' in df.columns:
    tweet_column = 'content'
else:
    print("❌ Error: No recognizable tweet text column found.")
    exit()

df = df.dropna(subset=[tweet_column])

# --- 1. TextBlob Sentiment ---
def textblob_sentiment(tweet):
    analysis = TextBlob(str(tweet))
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df['TextBlob_Sentiment'] = df[tweet_column].apply(textblob_sentiment)

# --- Emoji Mapping ---
def sentiment_to_emoji(sentiment):
    if sentiment == "Positive":
        return "😊"
    elif sentiment == "Negative":
        return "😠"
    else:
        return "😐"

df['TextBlob_Sentiment'] = df['TextBlob_Sentiment'].str.strip().str.capitalize()
df['Emoji'] = df['TextBlob_Sentiment'].apply(sentiment_to_emoji)

print("📊 Emoji mapping of the first Five Tweets:")
print(df[[tweet_column, 'TextBlob_Sentiment', 'Emoji']].head())

# --- 2. Prepare for ML ---
df_ml = df[df['TextBlob_Sentiment'] != "Neutral"].copy()
df_ml['label'] = df_ml['TextBlob_Sentiment'].apply(lambda x: 1 if x == "Positive" else 0)

X_train, X_test, y_train, y_test = train_test_split(
    df_ml[tweet_column], df_ml['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- 3. Logistic Regression ---
lr_model = LogisticRegression()
lr_model.fit(X_train_vec, y_train)
lr_preds = lr_model.predict(X_test_vec)
print("\n🔍 Logistic Regression Report:")
print(classification_report(y_test, lr_preds, target_names=['Negative', 'Positive']))
print("✅ Accuracy (Logistic Regression):", accuracy_score(y_test, lr_preds))

# --- 4. Naive Bayes ---
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_preds = nb_model.predict(X_test_vec)
print("\n🧪 Naive Bayes Report:")
print(classification_report(y_test, nb_preds, target_names=['Negative', 'Positive']))
print("✅ Accuracy (Naive Bayes):", accuracy_score(y_test, nb_preds))

# --- 5. Visualization ---
df['TextBlob_Sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
plt.title("Sentiment Analysis of Tweets (TextBlob)")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.tight_layout()
plt.show()

# --- 6. Save Results ---
df.to_csv("daily_tweets_sentiment_all_models.csv", index=False)
print("\n💾 Results saved to 'daily_tweets_sentiment_all_models.csv'.")

### --- 7. XGBoost ---
##try:
##    from xgboost import XGBClassifier
##
##    xgb_model = XGBClassifier(eval_metric='logloss')
##    xgb_model.fit(X_train_vec, y_train)
##    xgb_preds = xgb_model.predict(X_test_vec)
##
##    print("\n🚀 XGBoost Classifier Report:")
##    print(classification_report(y_test, xgb_preds, target_names=['Negative', 'Positive']))
##    print("✅ Accuracy (XGBoost):", accuracy_score(y_test, xgb_preds))
##
##except ImportError:
##    print("\n❌ XGBoost is not installed. Run 'pip install xgboost' to use it.")
##except Exception as e:
##    print(f"\n⚠️ XGBoost failed due to: {e}")
##
