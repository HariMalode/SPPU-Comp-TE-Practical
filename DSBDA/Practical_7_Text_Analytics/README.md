# Text Analytics with Python

## Overview

This code demonstrates some common text analytics techniques in Python using the NLTK library. It includes:

- Tokenization (word and sentence)
- Part-of-speech (POS) tagging
- Stopword removal
- Stemming
- Lemmatization

## Running the Code

To run this code, you need to have Python and the NLTK library installed.

First install NLTK:

```
pip install nltk
```

Then download the NLTK data:

```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```

Now you can run the code:

python text_analytics.py

## Explanation

### Tokenization

Tokenization splits text into tokens (words, punctuation, etc.). NLTK provides functions for word and sentence tokenization. Tokenization is an important first step in natural language processing tasks as it breaks down text into smaller units (words and sentences) that can then be further processed and analyzed.

```
from nltk import word_tokenize, sent_tokenize

word_tokens = word_tokenize(text)
sentence_tokens = sent_tokenize(text)
```

### POS Tagging

Part-of-speech (POS) tagging assigns a POS tag (noun, verb, adjective etc.) to each word. NLTK's pos_tag function does this. POS tagging labels each word with its part of speech based on its context and definition. This allows us to understand the grammatical structure of the text.

```
pos_tags = pos_tag(word_tokens)
```

### Stopword Removal

Stopwords are common words like 'a', 'and', 'the' that don't add much meaning. We can remove them with a list of stopwords. Stopword removal gets rid of frequent words that usually don't contain useful information for analysis. Removing stopwords helps reduce noise in the data.

```
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
cleaned_tokens = [word for word in words if word not in stop_words]
```

### Stemming

Stemming reduces words to their root form. NLTK provides several stemmer implementations, we use PorterStemmer here. Stemming simplifies words to their base form by removing affixes like prefixes and suffixes. This helps group together related words.

```
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stems = [stemmer.stem(word) for word in words]
```

### Lemmatization

Lemmatization reduces words to their root form based on context and vocabulary (lemmatization dictionary). NLTK provides WordNetLemmatizer. Lemmatization looks at the meaning and definition of words to convert them to their base form. It is more advanced than stemming.

```
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(word) for word in words]
```

This covers the main text analytics techniques shown in the code. They provide different ways to normalize, clean and process text that can be useful for many downstream NLP tasks.


--- 
Sure, let's go through the code step by step:

1. **Assigning Sentences**:
   ```python
   sentence1 = "I will walk 500 miles and I would walk 500 more. Just to be the man who walks " + \
            "a thousand miles to fall down at your door!"
   sentence2 = "I played the play playfully as the players were playing in the play with playfullness"
   ```
   Here, you've defined two sentences.

2. **Tokenization**:
   ```python
   from nltk import word_tokenize, sent_tokenize

   print("---------Tokenized Words------------")
   print('Tokenized words:', word_tokenize(sentence1))
   print("\n")
   ```
   This part imports necessary functions from NLTK library (`word_tokenize` and `sent_tokenize`). Then it tokenizes `sentence1` into words using `word_tokenize` and prints the result.

3. **Tokenization of Sentences**:
   ```python
   print("---------Tokenized Sentences------------")
   print('Tokenized sentences:', sent_tokenize(sentence1))
   print("\n")
   ```
   Similarly, this code tokenizes `sentence1` into sentences using `sent_tokenize` and prints the result.

4. **POS Tagging**:
   ```python
   from nltk import pos_tag
   print("---------POS Tagging------------")
   token = word_tokenize(sentence1) + word_tokenize(sentence2)
   print('POS tagged:', pos_tag(token))
   print("\n")
   ```
   This code imports `pos_tag` from NLTK and performs Part-Of-Speech (POS) tagging on the words of `sentence1` and `sentence2`, and prints the tagged words.

5. **Stop-Words Removal**:
   ```python
   print("---------Stop-Words Removal------------")
   from nltk.corpus import stopwords
   stop_words = set(stopwords.words('english'))
   token = word_tokenize(sentence1)
   cleaned_token = []
   for word in token:
       if word not in stop_words:
           cleaned_token.append(word)
   print("Unclean version:", token)
   print("\n")
   print("Cleaned version:", cleaned_token)
   print("\n")
   ```
   This part removes stop words from `sentence1`. It imports stop words from NLTK, then tokenizes `sentence1` and removes the stop words. It prints the original tokenized version and the cleaned version without stop words.

6. **Stemming**:
   ```python
   print("---------Stemming------------")
   from nltk.stem import PorterStemmer
   stemmer = PorterStemmer()
   token = word_tokenize(sentence2)
   stemmed = [stemmer.stem(word) for word in token]  
   print("Stemmed words:", stemmed)
   print("\n")
   ```
   It performs stemming on `sentence2`, reducing words to their root forms using Porter Stemmer and prints the stemmed words.

7. **Lemmatization**:
   ```python
   print("---------Lemmatization------------")
   from nltk.stem import WordNetLemmatizer
   lemmatizer = WordNetLemmatizer()
   token = word_tokenize(sentence2)
   lemmatized = [lemmatizer.lemmatize(word) for word in token]
   print("Lemmatized words:", lemmatized)
   print("\n")
   ```
   Similar to stemming, but here it performs lemmatization on `sentence2` using WordNet Lemmatizer and prints the lemmatized words.

8. **TF-IDF Representation**:
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer

   vectorizer = TfidfVectorizer()

   matrix = vectorizer.fit(sentence2)
   matrix.vocabulary_

   tfid_matrix = vectorizer.transform(sentence2)
   print(tfid_matrix)

   print(vectorizer.get_feature_names_out())
   ```
   This part creates a TF-IDF representation of `sentence2`. It imports `TfidfVectorizer` from scikit-learn, creates an instance of it, fits the model to `sentence2`, transforms `sentence2` to TF-IDF matrix, and then prints the matrix and the feature names. 

That's the line-by-line explanation of your code.

---

Sure, here are some potential questions along with their answers:

**Question 1:** What is the purpose of tokenization in natural language processing?

**Answer:** Tokenization is the process of breaking text into smaller units, such as words or sentences. It helps in preparing text data for further analysis by splitting it into meaningful units.

**Question 2:** Explain the concept of POS tagging. Why is it useful?

**Answer:** POS tagging stands for Part-Of-Speech tagging. It involves labeling each word in a sentence with its corresponding part of speech, such as noun, verb, adjective, etc. It's useful because it provides insights into the grammatical structure of a sentence, which is essential for many NLP tasks like text analysis, information extraction, and sentiment analysis.

**Question 3:** What are stop words, and why are they removed during text preprocessing?

**Answer:** Stop words are common words like "and," "the," "is," etc., that are often filtered out during text preprocessing because they occur frequently in the language and usually don't carry much meaning for analysis tasks. Removing them helps reduce noise and improve the efficiency of algorithms by focusing on more meaningful words.

**Question 4:** What is stemming? Provide an example.

**Answer:** Stemming is the process of reducing words to their root forms. For example, the word "walking" would be stemmed to "walk." It helps in normalization and reducing the vocabulary size.

**Question 5:** What is lemmatization, and how does it differ from stemming?

**Answer:** Lemmatization is the process of reducing words to their base or dictionary form, called a lemma. Unlike stemming, lemmatization considers the context of the word and aims to return a valid word that exists in the language. For example, "walking" would be lemmatized to "walk." 

**Question 6:** Explain the concept of TF-IDF. How is it used to represent a document?

**Answer:** TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a numerical statistic that reflects the importance of a word in a document relative to a collection of documents. It is calculated by multiplying the term frequency (how often a term appears in a document) by the inverse document frequency (the logarithmically scaled fraction of the documents that contain the term). TF-IDF is used to represent documents in a vector space, where each dimension corresponds to a unique term and the value represents the importance of that term in the document.

**Question 7:** How does the TF-IDF vectorizer in scikit-learn work?

**Answer:** The TF-IDF vectorizer in scikit-learn converts a collection of raw documents into a matrix of TF-IDF features. It tokenizes the input text, counts the occurrences of each term in each document (term frequency), and then applies the IDF transformation to downscale the importance of terms that appear frequently across documents. Finally, it normalizes the TF-IDF vectors to unit length. The result is a matrix where each row represents a document, and each column represents a term, with the cell values representing the TF-IDF scores.

These questions should help you understand the concepts and be prepared for your exam.
