from xgboost import XGBClassifier
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Loading the dataset
try:
    df = pd.read_csv("daily_tweets.csv")
except FileNotFoundError:
    print("Error: 'daily_tweets.csv' not found.")
    exit()

# Identifying the tweet columns
if 'Tweet' in df.columns:
    tweet_column = 'Tweet'
elif 'text' in df.columns:
    tweet_column = 'text'
elif 'content' in df.columns:
    tweet_column = 'content'
else:
    print("Error: No recognizable tweet text column found.")
    exit()

df = df.dropna(subset=[tweet_column])

# ---------- 1. TEXTBLOB ---------- #
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

# Clean up sentiment labels in case of stray spaces/case
df['TextBlob_Sentiment'] = df['TextBlob_Sentiment'].str.strip().str.capitalize()

# Apply the emoji function
df['Emoji'] = df['TextBlob_Sentiment'].apply(sentiment_to_emoji)

# Preview output
print("Emoji mapping of the first Five Tweets from the Data set: ")
print("\n\n")
print(df[[tweet_column, 'TextBlob_Sentiment', 'Emoji']].head())


# ---------- 2. Prepare for ML ---------- #
df_ml = df[df['TextBlob_Sentiment'] != "Neutral"].copy()

df_ml['label'] = df_ml['TextBlob_Sentiment'].apply(lambda x: 1 if x == "Positive" else 0)

X_train, X_test, y_train, y_test = train_test_split(
    df_ml[tweet_column], df_ml['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

## ---------- 3. Logistic Regression ---------- #
##lr_model = LogisticRegression()
##lr_model.fit(X_train_vec, y_train)
##lr_preds = lr_model.predict(X_test_vec)
##print("\n\n")
##print("\n🔍 Logistic Regression Report:")
##print(classification_report(y_test, lr_preds, target_names=['Negative', 'Positive']))

## ---------- 4. Naive Bayes ---------- #
##nb_model = MultinomialNB()
##nb_model.fit(X_train_vec, y_train)
##nb_preds = nb_model.predict(X_test_vec)
##print("\n\n")
##print("\n🧪 Naive Bayes Report:")
##print(classification_report(y_test, nb_preds, target_names=['Negative', 'Positive']))

### ---------- 5. Visualization (TextBlob-based) ---------- #
##df['TextBlob_Sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
##plt.title("Sentiment Analysis of Tweets (TextBlob)")
##plt.xlabel("Sentiment")
##plt.ylabel("Number of Tweets")
##plt.tight_layout()
##plt.show()
##
### ---------- 6. Save results ---------- #
##df.to_csv("daily_tweets_sentiment_all_models.csv", index=False)
##print("\nResults saved to 'daily_tweets_sentiment_all_models.csv'.")
##
# ------------7. XGBoost --------------- #
from sklearn.metrics import classification_report, accuracy_score

try:
    from xgboost import XGBClassifier

    xgb_model = XGBClassifier(eval_metric='logloss')
    xgb_model.fit(X_train_vec, y_train)
    xgb_preds = xgb_model.predict(X_test_vec)

    print("\n\n")
    print("\n📊 XGBoost Classifier Report:")
    print(classification_report(y_test, xgb_preds, target_names=['Negative', 'Positive']))
    
    # Print accuracy
    accuracy = accuracy_score(y_test, xgb_preds)
    print(f"\n✅ XGBoost Accuracy: {accuracy:.2f}")

except ImportError:
    print("\n❌ XGBoost library is not installed. Please install it using 'pip install xgboost'")
except Exception as e:
    print(f"\n❌ XGBoost training failed due to error: {e}")
