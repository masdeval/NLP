# Twitter Data Analysis

**Christian Braz**  
George Washington University — Department of Data Science

---

## Abstract

Today's Internet content is largely made from unstructured data, mostly images and text. Part of the text content is in the form of comments made by users giving their impressions about virtually everything. Processing natural language text documents is still a challenging task due to its ambiguity and context dependency. Nevertheless, being able to extract useful information from this source is highly desirable due to its market value. Understanding people's feelings about a matter, customer needs, or product quality are just some of the possibilities by accomplishing this task. In this work, we evaluate the performance of machine learning and natural language processing techniques in the task of classifying sentiment in tweets.

**Keywords:** sentiment analysis, machine learning, natural language processing

---

## Introduction

Since its rise in the early 90s, the World Wide Web has been modifying the way people interact. Its distributed infrastructure, built upon the Internet's top layers, made it pervasive and an ideal tool to enable all kinds of communication services. On a business perspective, new jargons were created to categorize such virtual interactions, such as Business-to-Consumer, Business-to-Business, and Customer-to-Customer (Brzozowska, 2018). Social networks such as Facebook, Twitter, and Instagram are well-known interactive platforms that enable users to express their opinions.

Being able to process and extract useful information from this rich source became valuable in many different aspects (Kanhere, 2011). Microblogging platforms have become highly successful as opinion promoters, and Twitter is their main representative. Users can publish their opinions on a variety of topics, discuss current issues, complain, and express positive or negative sentiment. Therefore, Twitter is a rich source of data for opinion mining and sentiment analysis. The value of such trend analysis goes beyond marketing. For instance, Kavanaugh et al. (2012) explore the use of traditional social media content — Twitter, Facebook, Flickr, and YouTube — to detect in real time spikes in activity related to public safety issues. Analyzing information from multiple social media sources should make it possible to identify convergence situations, useful for handling crisis events such as traffic incidents or earthquakes. Tools like the **tag cloud** help visualize the "big picture" of social media activity by identifying the most frequent terms in a large collection.

The purpose of this work is to evaluate two machine learning algorithms — **Support Vector Machines (SVM)** and **Maximum Entropy (MaxEnt)** — for predicting two classes of sentiment (positive or negative) in tweets, using word embeddings as features. We try different settings for creating custom vectors and also use pre-trained ones. We report the overall test accuracy and F1-score of a 5-fold cross-validation training step and also evaluate these metrics on a held-out test set. The paper is organized as follows: **Related Work** reviews the main techniques used for opinion classification. **Methods** presents the underlying theory. **Results** describes the experiment and discusses the values obtained. **Conclusion** presents final considerations.

---

## Related Work

### Sentiment Analysis Methods

Manual analysis of the volume of content produced through social networks is difficult and time-consuming. Sentiment Analysis (SA) — also known as Opinion Mining (OM) — has been introduced as an effective way to automatically extract knowledge from comments (Hemmatian & Sohrabi, 2017; Medhat, Hassan, & Korashy, 2014). More specifically, sentiment analysis is the process of extracting sentiments expressed by users in unstructured subjective texts and distinguishing their polarities — whether a feeling is positive or negative (Hemmatian & Sohrabi, 2017).

Medhat et al. (2014), Liu and Zhang (2012), Hemmatian and Sohrabi (2017), and Poirier, Bothorel, Neef, and Boullé (2008) roughly divide sentiment analysis into two groups: **linguistic** and **machine learning** approaches.

#### Linguistic-Based Sentiment Classification

In the research literature, opinion words are used to express desired and undesired states. Along with individual words, there are also opinion phrases and idioms — collectively called **opinion lexicon** — which are instrumental for opinion mining (Liu & Zhang, 2012, p. 9). Each word has a polarity associated with the feeling assessment it brings to mind (Hemmatian & Sohrabi, 2017). By studying the occurrence frequency of words in annotated text corpora, polarity can be identified: if a word occurs more often among positive texts, its polarity is positive; if equally distributed, it is neutral; if associated with more negative texts, it is negative.

Two main strategies exist for finding opinion words:

- **Dictionary-based:** Begins with a small set of opinion words with known orientation (seed words), then expands the set iteratively by adding synonyms and antonyms. The process stops when no new words are found.
- **Corpus-based:** Addresses the problem that semantic orientation is domain-dependent. It relies on a compiled set of polar words and syntactic patterns that co-occur with seed opinion words to find additional opinion words in a large corpus. The technique begins with a seed list of opinion adjectives and uses linguistic constraints on connectives (AND, OR, BUT, ...) to identify further adjective opinion words and their orientations (Medhat et al., 2014).

Linguistic-based methods perform well across many scenarios but require an extensive and expensive pre-processing step. Learning-based methods can be employed in a more straightforward way.

#### Learning-Based Sentiment Classification

Learning-based approaches rely on conventional machine learning algorithms to solve text classification problems (Medhat et al., 2014). Machine learning techniques can be classified as supervised and unsupervised; in both cases, the aim is to build a statistical model from data that can explain its behavior. According to Liu and Zhang (2012), "any existing supervised learning methods can be applied to sentiment classification" (p. 9).

One of the most common approaches to feature engineering is the **TF-IDF**[^1] scheme, which "has been shown quite effective in sentiment classification" (Liu & Zhang, 2012, p. 10). A term can be a single word (unigram) or a longer combination (n-grams). Features extracted this way are known as **bag-of-words (BOW)** because they do not rely on word order, only frequency. Despite its simplicity, high accuracy has been reported with unigram and bigram features for sentiment classification.

The main drawbacks of BOW are the lack of context and the large feature space. More recent strategies learn representations in an unsupervised way, based on the hypothesis that words occurring in similar contexts tend to have similar meanings — leading to **word vector semantics** (Jurafsky & Martin, 2000). These word vectors embed meaningful semantics and can be used in any NLP application. This work explores the use of word vectors as features for sentiment classification.

[^1]: Term Frequency-Inverted Document Frequency

### Sentiment Classification of Twitter

Twitter is a social networking and microblogging service allowing users to post real-time messages called **tweets**. Initially restricted to 140 characters (later increased to 280), tweets make heavy use of acronyms, emoticons, and special characters. Key Twitter terminology (Agarwal et al., 2011):

- **Emoticons:** Facial expressions pictorially represented using punctuation and letters that express the user's mood.
- **Target (`@`):** The "@" symbol refers to other users, automatically alerting them.
- **Hashtags (`#`):** Used to mark topics and increase the visibility of tweets.

Pang, Lee, and Vaithyanathan (2002) provided a pioneering paper on applying machine learning methods — Naïve Bayes, SVM, and Maximum Entropy — to text classification, using BOW with unigram, bigram, and part-of-speech features. They concluded that unigrams perform well in both classifiers and that SVM had the best performance in all scenarios.

Go, Bhayani, and Liu (2009) produced one of the earliest results on Twitter sentiment analysis using **distant supervision** — relying on the polarity of the last emoticon to label tweets as positive or negative. They built models with Naïve Bayes, MaxEnt, and SVM, reporting that SVM with unigram features outperforms other classifiers with **71.35% accuracy**.

Agarwal et al. (2011) designed an extensive set of features based on a dictionary of emoticons and acronyms, comparing three models: unigram, 100 senti-features, and tree kernel model. Results reached 74% for the kernel and 75.4% mixing unigram + senti-features. Notably, senti-features performed nearly as well as the unigram baseline (71.27%) despite the latter having ~13,000 features. A more recent result by Arslan, Küçük, and Birturk (2018) used a pre-trained word embedding model over 400 million tweets to extract features and train a neural network, reporting roughly **85% accuracy** — though in a flawed train/test methodology where the same dataset was used twice.

---

## The Experiment

The objective is to compare Maximum Entropy and Support Vector Machine for predicting overall tweet sentiment using word vectors as features. The baseline is a Maximum Entropy model trained on unigram and bigram BOW features.

### Dataset

Since no single large high-quality labeled dataset was found, the final dataset is a concatenation of **five human-annotated datasets**. Two components were used:

- **Embedding corpus:** 1.6 million automatically labeled tweets from the **Stanford Twitter Sentiment (Sentiment140)** dataset (Go, Bhayani, & Huang, 2009). Used only for training the custom word embedding — not for classifier training or testing — since automatic sentiment annotation via emoticons has arguable accuracy.
- **Classifier training corpus:** Five manually annotated datasets, four from Saif, Fernández, He, and Alani (2013) and one from Tromp and Pechenizkiy (2011).

### Overview of Manually Labeled Datasets

| Dataset | #tweets | #positives | #neutral | #negative |
|---|---|---|---|---|
| HCR | 2,516 | 541 | 470 | 1,381 |
| Sanders | 5,513 | 570 | 2,503 | 654 |
| SemEval | 34,183 | 15,543 | 6,440 | 12,290 |
| OMD | 3,238 | 710 | — | 1,196 |
| Tromp | 11,778 | 3,458 | 4,706 | 3,614 |
| **Total** | **57,228** | **20,732** | **14,119** | **19,135** |

The final balanced dataset contains **40,000 examples** of binary human-labeled tweet sentiments.

### Feature Extraction

**Bag-of-Words (BOW):** Represents each document as a sparse vector counting word occurrences. Word order is ignored (n-gram model). Raw frequencies are weighted using the **TF-IDF** scheme, calculated as:

\[ \text{tf-idf}(t, d, D) = \text{tf}(t, d) \times \text{idf}(t) \]

\[ \text{idf}(t) = \log \frac{n_d}{1 + \text{df}(d, t)} \]

**Word Semantic Vectors (Word2Vec):** A supervised learning technique for learning distributed word representations (Mikolov et al., 2013). The **skip-gram with negative sampling** model works as follows:

- Given an input word, a two-layer neural network is trained to predict the probability of all other words in the vocabulary being in the vicinity of the input word. The input layer is a one-hot vector of vocabulary size; the hidden layer has 300 neurons with an identity transfer function; the output is a softmax layer of vocabulary size. For every sentence where the input word occurs, a randomly chosen nearby word becomes the target label — the network learns that input and output words are related.

- **Subsampling:** Very frequent words like "the" add noise without useful context information. Word2Vec implements a subsampling mechanism to prune such words from training, improving representation quality and reducing computational cost.

- **Negative sampling:** With 300 hidden neurons fully connected to a vocabulary-size output layer, updating all weights for billions of training examples is prohibitively expensive. Negative sampling addresses this by having each training sample modify only a small percentage of weights — the output word plus a small number of randomly selected "negative" words (for which the network should output 0). This drastically reduces training cost.

**Custom embedding:** Created using the **Gensim** package with vector size 200 and a window of 5 words (appropriate given tweets average ~50 words).

**Pre-trained embedding:** **GloVe** 200-dimensional word vectors trained on a Twitter corpus of 2 billion tweets, 27 billion tokens, and 1.2 million vocabulary words (https://nlp.stanford.edu/projects/glove/).

### Classifiers

**Support Vector Machines (SVM):** Finds a hyperplane that separates tweets according to the expressed opinion, maximizing the separation margin. The algorithm returns a decision function \(h(d)\), so that the probability of an opinion \(s\) given a comment \(d\) is:

\[ p(s \mid d) = \frac{1}{1 + e^{ah(d)+b}} \]

where \(a\) and \(b\) are estimated by minimizing the negative log-likelihood function in \(\mathcal{D}\).

**Maximum Entropy (MaxEnt):** One of the most widely used probabilistic models across NLP applications. MaxEnt computes the probability of output class \(c\) given input \(d\) as:

\[ P_{ME}(c \mid d, \lambda) = \frac{\exp \left[ \sum_i \lambda_i f_i(c, d) \right]}{\sum_{c'} \left[ \exp \sum_i \lambda_i f_i(c, d) \right]} \]

### Experimental Design

A **5-fold cross-validation** was used for training, reporting overall accuracy and F1-score. The experiment was structured as follows:

1. Train a **Logistic Regression baseline** using unigram and bigram BOW features.
2. Create a **custom 200-dimensional Word2Vec embedding** from 1.6 million tweets (window size 5, punctuation removed).
3. Classify using 5-fold cross-validation with **custom Word2Vec** vectors (punctuation removed, Gensim tokenizer).
4. Classify using 5-fold cross-validation with **GloVe Word2Vec** vectors (Stanford preprocessing script, punctuation kept).

---

## Results

### Main Results (5-fold Cross-Validation)

| Algorithm | Feature | Accuracy | F1 |
|---|---|---|---|
| Logistic Regression | BOW | 0.726 | 0.775 |
| Logistic Regression | Custom Word2Vec | 0.762 | 0.820 |
| SVM (C=1, linear) | Custom Word2Vec | 0.767 | 0.823 |
| Logistic Regression | GloVe Word2Vec | 0.783 | 0.834 |
| **SVM (C=1, linear)** | **GloVe Word2Vec** | **0.783** | **0.834** |

The advantage of semantic word vectors is consistent across all scenarios. GloVe embeddings were expected to perform better given their larger Twitter training corpus, but Logistic Regression and SVM achieved identical results on GloVe features. Results are comparable to those reported by Agarwal et al. (2011).

### SVM Kernel Comparison (Custom Word2Vec, C=1)

| Kernel | Accuracy |
|---|---|
| **Linear** | **0.767** |
| RBF | 0.726 |
| Polynomial | 0.651 |

The linear kernel achieves the best result and is the setting reported in the main results table.

### TF-IDF Weighting of Word Embeddings

Word embeddings were weighted by TF-IDF values before averaging to obtain tweet vectors. Interestingly, this did **not** improve accuracy:

| Algorithm | Feature | Accuracy | F1 |
|---|---|---|---|
| Logistic Regression | Custom Word2Vec + TF-IDF | 0.743 | 0.812 |
| SVM (C=1, linear) | Custom Word2Vec + TF-IDF | 0.747 | 0.813 |
| Logistic Regression | GloVe Word2Vec + TF-IDF | 0.757 | 0.821 |
| SVM (C=1, linear) | GloVe Word2Vec + TF-IDF | 0.758 | 0.822 |

TF-IDF weighting consistently degrades both accuracy and F1 relative to simple averaged embeddings.

---

## Conclusion

Two machine learning algorithms were evaluated for classifying sentiment in tweets using word vectors — an instantiation of the linguistic idea of distributional semantics — as features, compared against the traditional bag-of-words approach. Two sets of word vectors were used: custom vectors trained on the Sentiment140 corpus and pre-trained GloVe vectors. Classifier training used a carefully concatenated set of five manually labeled Twitter corpora from prior sentiment classification benchmarks.

Key findings:
- Word vectors consistently outperform BOW representations.
- GloVe and custom Word2Vec yield similar final accuracy when using the same classifier.
- TF-IDF weighting of word embeddings does not improve and slightly hurts performance.
- Linear SVM and Logistic Regression perform nearly identically on the same features.

For future experiments, the following directions are recommended:
- Higher-dimensional word vectors (beyond 200 dimensions)
- Different and ensemble classifiers for improved accuracy
- Neural sequence models, particularly **Recurrent Neural Networks (RNNs)**, which have been producing state-of-the-art results in many sequence-dependent tasks

---

## References

Agarwal, A., Xie, B., Vovsha, I., Rambow, O., & Passonneau, R. (2011). Sentiment analysis of twitter data. In *Proceedings of the Workshop on Languages in Social Media* (pp. 30–38). Association for Computational Linguistics. http://dl.acm.org/citation.cfm?id=2021109.2021114

Arslan, Y., Küçük, D., & Birturk, A. (2018). Twitter sentiment analysis experiments using word embeddings on datasets of various scales. In M. Silberztein et al. (Eds.), *Natural Language Processing and Information Systems* (pp. 40–47). Springer International Publishing.

Brzozowska, A. (2018). E-business as a new trend in the economy. *Procedia Computer Science, 65*, 1095–1104.

Go, A., Bhayani, R., & Liu, L. (2009). Twitter sentiment classification using distant supervision. *Technical Report, Stanford*. http://help.sentiment140.com

Hemmatian, F., & Sohrabi, M. K. (2017). A survey on classification techniques for opinion mining and sentiment analysis. *Artificial Intelligence Review*. https://doi.org/10.1007/s10462-017-9599-6

Jurafsky, D., & Martin, J. H. (2000). *Speech and language processing: An introduction to natural language processing, computational linguistics, and speech recognition* (1st ed.). Prentice Hall PTR.

Kanhere, S. (2011). Participatory sensing: Crowdsourcing data from mobile smartphones in urban spaces. In *ICMDM* (pp. 3–6).

Kavanaugh, A. L., Fox, E. A., Sheetz, S. D., Yang, S., Li, L. T., Shoemaker, D. J., … Xie, L. (2012). Social media use by government: From the routine to the critical. *Government Information Quarterly, 29*(4), 480–491.

Liu, B., & Zhang, L. (2012). A survey of opinion mining and sentiment analysis. In *Mining Text Data* (pp. 415–463).

Medhat, W., Hassan, A., & Korashy, H. (2014). Sentiment analysis algorithms and applications: A survey. *Ain Shams Engineering Journal, 5*(4), 1093–1113. https://doi.org/10.1016/j.asej.2014.04.011

Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In *Proceedings of the 26th International Conference on Neural Information Processing Systems — Volume 2* (pp. 3111–3119). Curran Associates Inc.

Pang, B., Lee, L., & Vaithyanathan, S. (2002). Thumbs up? Sentiment classification using machine learning techniques. *CoRR, cs.CL/0205070*.

Poirier, D., Bothorel, C., Neef, E. G. D., & Boullé, M. (2008). Automating opinion analysis in film reviews: The case of statistic versus linguistic approach. In *LREC Workshop on Sentiment Analysis: Emotion, Metaphor, Ontology and Terminology* (pp. 94–101).

Saif, H., Fernández, M., He, Y., & Alani, H. (2013). Evaluation datasets for twitter sentiment analysis: A survey and a new dataset, the STS-Gold. In *1st International Workshop on Emotion and Sentiment in Social and Expressive Media (ESSEM 2013)*. http://oro.open.ac.uk/40660/

Tromp, E., & Pechenizkiy, M. (2011). Senticorr: Multilingual sentiment analysis of personal correspondence. In *Proceedings of the 2011 IEEE 11th International Conference on Data Mining Workshops* (pp. 1247–1250). IEEE Computer Society. https://doi.org/10.1109/ICDMW.2011.152
