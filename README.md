# Introduction
Text Analysis involves various techniques such as text preprocessing, sentiment analysis, named entity recognition, topic modelling, and text classification. Text analysis plays a crucial role in understanding and making sense of large volumes of text data, which is prevalent in various domains, including news articles, social media, customer reviews, and more.

## Choose Text Analysis Techniques:
Select the appropriate text analysis techniques based on your objectives. Common techniques include:

### Sentiment Analysis:
Determine the sentiment (positive, negative, neutral) expressed in the text.

### Topic Modeling:
Identify topics or themes present in the text data.

### Named Entity Recognition (NER):
Extract entities such as names, locations, organizations, etc.

### Text Classification:
Categorize text into predefined classes or labels.

**Note: Here, I choose to work on all these techniques.**

## 1. Text Classification
Text classification is a natural language processing (NLP) task where the goal is to assign predefined categories or labels to a given piece of text. It's a fundamental task in NLP and has a wide range of applications, including spam detection, topic categorization, and more.

Here's a general overview of the text classification process:

#### i) Data Collection:

Gather a dataset with labeled examples of text. Each example should be associated with a category or label.

**Here, I used real and fake news classification dataset.**

#### ii) Data Preprocessing:

Clean and preprocess the text data. Common preprocessing steps include:
* **Lowercasing**: Convert all text to lowercase to ensure consistency.
* **Tokenization**: Break the text into individual words or tokens.
* **Removing Stop Words**: Exclude common words that do not contribute much to the meaning.
* **Lemmatization or Stemming**: Reduce words to their base or root form.

#### iii) Feature Extraction:

Convert the preprocessed text into a numerical format suitable for machine learning algorithms. Common techniques include:
* **Bag of Words (BoW)**: Represent each document as a vector of word frequencies.
* **TF-IDF (Term Frequency-Inverse Document Frequency)**: Weigh words based on their importance in a document relative to the entire corpus.
* **Word Embeddings**: Represent words as dense vectors in a continuous vector space.

**In this project, I used word embedding technique.**

#### iv) Model Training:

Choose a classification algorithm and train the model on the labeled data. Common algorithms for text classification include:
Naive Bayes
Support Vector Machines (SVM)
Logistic Regression
Deep Learning models (e.g., LSTM, GRU, or Transformer-based models)

**In this project, I used Naive Bayes Classifier and Gradient Boosting Classifier.**

#### v) Evaluation:

Assess the performance of the trained model using metrics such as accuracy, precision, recall, and F1-score. Split the dataset into training and testing sets to evaluate generalization performance.

#### vi) Fine-Tuning and Optimization (If required):

Adjust hyperparameters, try different algorithms, or perform feature engineering to improve the model's performance.
Inference:

**Use the trained model to classify new, unseen text into predefined categories.**

## 2. Sentiment Analysis
Sentiment analysis, also known as opinion mining, is a natural language processing (NLP) task that involves determining the sentiment or emotional tone expressed in a piece of text. The goal is to classify the sentiment as positive, negative, or neutral. Sentiment analysis has applications in customer feedback analysis, social media monitoring, product reviews, and more.

#### Sentiment Analysis using TextBlob
The sentiment.polarity method of TextBlob calculates a sentiment polarity score for each article, where positive values indicate positive sentiment, negative values indicate negative sentiment and values close to zero suggest a more neutral tone.

**Note: In this project, sentiment analysis is performed on the text column in the dataset to assess the overall sentiment or emotional tone of the news articles. The TextBlob library is used here to analyze the sentiment polarity, which quantifies whether the text expresses positive, negative, or neutral sentiment.**

## 3. Named Entity Recognition
Named Entity Recognition (NER) is a natural language processing (NLP) task that involves identifying and classifying named entities (such as persons, organizations, locations, dates, and more) in a given text. The goal is to extract structured information about these entities, making it easier to understand the content and relationships within the text.

**In this project, I have extracted named entities from the Text column in the dataset. The extracted entities are stored in a new column called NER in the dataset. Then, a visualization is created to present the top 5 most frequently occurring named entities and their respective counts, allowing for a quick understanding of the prominent entities mentioned in the text data.**

**Note: For Named Entity Recognition (NER), it's generally advisable to provide the unprocessed or minimally processed text to the NER model. The reason for this is that NER models often rely on the original context of words, including stop words and punctuation, to accurately identify named entities.**

## 4. Topic Modelling
Topic modeling is a natural language processing technique that is used to identify topics present in a text corpus. The goal is to discover hidden thematic structures in a collection of documents. It helps in understanding the main themes or subjects that the documents are about. Topic modeling is widely used in various applications such as document categorization, content recommendation, and information retrieval.

#### Topic Modelling using LDA:

#### Latent Dirichlet Allocation (LDA):
Latent Dirichlet Allocation (LDA) is a popular topic modeling technique introduced by David Blei, Andrew Ng, and Michael Jordan in 2003. LDA assumes that documents are mixtures of topics and that each word in a document is attributable to one of the document's topics.

Here's a simplified explanation of how LDA works:

##### Assumptions:
* Each document is a mixture of a small number of topics.
* Each word in the document is attributable to one of the document's topics.

##### Process:
* For each document:
     * Decide on the number of words it will have.
     * Choose a mixture of topics for the document.

* For each word in the document:
     * Choose a topic from the document's mixture.
     * Choose a word from the topic's distribution.

##### Mathematical Representation:
* LDA represents documents as mixtures of topics and topics as mixtures of words.
* Mathematically, it uses probability distributions to model these mixtures.
* Dirichlet distributions are employed to model the distribution of topics in a document and the distribution of words in a topic.

##### Inference:
Given a collection of documents, the goal of LDA is to infer:
* The topics that are prevalent across the entire corpus.
* The distribution of topics within each document.
* The distribution of words within each topic.

##### Output:
The output of LDA is a set of topics, where each topic is represented by a distribution of words. Additionally, each document is represented by a distribution of topics.

##### Use Cases:
* LDA is used for document classification, clustering, and summarization.
* It's commonly applied in areas such as information retrieval, content recommendation, and understanding the themes in large text corpora.
