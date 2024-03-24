
import nltk
import re
import networkx as nx
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#nltk.download('stopwords')
#nltk.download('punkt')


text = input("Enter text :")

#tokenizing sentence
sentclean = sent_tokenize(text)


#removing stop words
stop = set(stopwords.words('English'))
stop.add('im')

#stemming
stemmer = PorterStemmer()

out = []

for s in sentclean:
    cleaned = ''.join(c for c in s if c.isalpha() or c.isspace())
    word = word_tokenize(cleaned)
    filtered = [stemmer.stem(w.lower()) for w in word if w.lower() not in stop]
    out.append(' '.join(filtered))


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(out)

# Compute pairwise cosine similarity scores
cosine_matrix = cosine_similarity(X, X)

# Create graph representation for the sentences
graph = nx.from_numpy_array(cosine_matrix)

# Compute the PageRank scores
scores = nx.pagerank(graph)

# Sort the sentences based on their PageRank scores
ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentclean)), reverse=True)

# Set the number of sentences you want in the summary
num_sentences_in_summary = 3

# Select the top sentences to form the summary
summary_sentences = []
for i in range(num_sentences_in_summary):
    if i < len(ranked_sentences):
        summary_sentences.append(ranked_sentences[i][1])

# Join the summary sentences to form the final summary
summary = ' '.join(summary_sentences)

print("Summary:")
print(summary)