
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


emails = [
    "Your bank account is blocked. Click this link to verify immediately.",
    "Congratulations you won a free prize. Claim it now.",
    "Urgent action required. Update your password using this link.",
    "Your account has been suspended. Verify your details now.",
    "You have won a lottery. Send your information to claim.",
    "Click here to avoid account closure.",
    "Your payment failed. Update your card details immediately.",
    "We detected suspicious activity. Login now to secure your account.",

    "Meeting scheduled for tomorrow at 10 AM.",
    "Please find the assignment attached to this email.",
    "Your order has been shipped and will arrive soon.",
    "Reminder: Your appointment is scheduled for Monday.",
    "Thank you for attending our online meeting.",
    "The project report is attached for your review.",
    "Your monthly newsletter is ready to read.",
    "Your college examination timetable has been published."
]

labels = [
    "Phishing", "Phishing", "Phishing", "Phishing",
    "Phishing", "Phishing", "Phishing", "Phishing",
    "Safe", "Safe", "Safe", "Safe",
    "Safe", "Safe", "Safe", "Safe"
]

# Extract simple URL-related features for display
def extract_features(email):
    urls = re.findall(r'https?://\S+|www\.\S+', email)
    suspicious_words = [
        "urgent", "verify", "password", "blocked",
        "suspended", "prize", "claim", "click"
    ]

    keyword_count = sum(
        word in email.lower() for word in suspicious_words
    )

    return {
        "URLs": urls,
        "Suspicious Keywords": keyword_count
    }


X_train, X_test, y_train, y_test = train_test_split(
    emails,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)


model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])


model.fit(X_train, y_train)

predictions = model.predict(X_test)


accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", round(accuracy * 100, 2), "%")


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))


print("\n--- Email Testing ---")
new_email = input("Enter an email message: ")

features = extract_features(new_email)
result = model.predict([new_email])[0]

print("\nExtracted Features:", features)
print("Email Classification:", result)
