
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ---------------- NLTK DOWNLOADS ----------------

def download_resources():

    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4")
    ]

    for path, package in resources:

        try:
            nltk.data.find(path)

        except LookupError:
            nltk.download(package, quiet=True)


download_resources()


# ---------------- INITIALIZATION ----------------

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()


# ---------------- TEXT CLEANING ----------------

def clean_text(text):

    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        "",
        text
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenization
    words = text.split()

    # Remove stopwords and apply lemmatization
    cleaned_words = []

    for word in words:

        if word not in stop_words:

            word = lemmatizer.lemmatize(word)

            cleaned_words.append(word)

    return " ".join(cleaned_words)


# ---------------- TESTING ----------------

if __name__ == "__main__":

    sample = """
    THIS PRODUCT IS AMAZING!!!
    I absolutely LOVE it.
    Visit https://example.com
    """

    print("Original text:")
    print(sample)

    print("\nCleaned text:")
    print(clean_text(sample))
