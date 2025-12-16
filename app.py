import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Load data
data = pd.read_csv("spam_and_ham_classification.csv")

# Keep only text and label columns
data = data[['text', 'label']]
data.columns = ['Message', 'label']

# Clean labels
data['label'] = data['label'].str.strip().str.lower()
data['category'] = data['label'].map({'ham': 'Not Spam', 'spam': 'Spam'})

# Drop rows with missing category
data = data.dropna(subset=['category'])

# Final columns
data = data[['Message', 'category']]

# Features and labels
mess = data['Message']
cat = data['category']

# Train-test split
mess_train, mess_test, cat_train, cat_test = train_test_split(
    mess, cat, test_size=0.2
)

# Vectorization
cv = CountVectorizer(stop_words='english')
features_train = cv.fit_transform(mess_train)
features_test = cv.transform(mess_test)

# Logistic Regression model
model = LogisticRegression(
    solver='liblinear',
    max_iter=1000,
    n_jobs=2
)

model.fit(features_train, cat_train)

# Check accuracy 
print("Logistic Regression accuracy:", model.score(features_test, cat_test))

# Predict function
def predict(message):
    X = cv.transform([message])
    proba = model.predict_proba(X)[0]
    spam_p = proba[1]
    return 'Spam' if spam_p >= 0.4 else 'Not Spam'

# Streamlit UI
st.header('Spam Detection')
input_mess = st.text_input('Enter Message Here')

if st.button('Validate'):
    if input_mess.strip() == "":
        st.warning("Please enter a message.")
    else:
        output = predict(input_mess)
        st.markdown(output)

