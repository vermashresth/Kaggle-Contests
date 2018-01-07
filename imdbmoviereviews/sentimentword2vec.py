from nltk.corpus import stopwords
import string
from collections import Counter
from string import punctuation
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from numpy import array
from numpy import asarray
from numpy import zeros

#load doc into memory
def load_doc(filename):
	file=open(filename,'r')
	text=file.read()
	file.close()
	return text

def clean_doc(doc):
	tokens=doc.split()
	table=string.maketrans(string.punctuation,' '*len(string.punctuation))
	tokens=[w.translate(table) for w in tokens]
	tokens=[word for word in tokens if word.isalpha()]

	stop_words=set(stopwords.words('english'))
	tokens=[w for w in tokens if not w in stop_words]

	tokens=[word for word in tokens if len(word)>1]
	return tokens
"""
filename='txt_sentoken/pos/cv000_29590.txt'
text=load_doc(filename)
tokens=clean_doc(text)
print(tokens)
"""
def add_doc_to_vocab(filename,vocab):
	doc=load_doc(filename)
	tokens=clean_doc(doc)
	vocab.update(tokens)

def process_docs(directory,vocab,is_train):
	for filename in listdir(directory):
		if is_train and filename.startswith('cv9'):
			continue
		if not is_train and not filename.startswith('cv9'):
			continue
		path=directory+'/'+filename
		add_doc_to_vocab(path,vocab)


vocab=Counter()


process_docs('txt_sentoken/neg', vocab, True)
process_docs('txt_sentoken/pos', vocab, True)
# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
#print(vocab.most_common(50))

min_occ=2
tokens=[k for k,c in vocab.items() if c>=min_occ]
print(len(tokens))

def save_list(lines, filename):
	# convert lines to a single blob of text
	data = '\n'.join(lines)
	# open file
	file = open(filename, 'w')
	# write text
	file.write(data)
	# close file
	file.close()
 
# save tokens to a vocabulary file
save_list(tokens, 'vocab.txt')

# turn a doc into clean tokens
def doc_to_clean_lines(doc, vocab):
	clean_lines = list()
	lines = doc.splitlines()
	for line in lines:
		# split into tokens by white space
		tokens = line.split()
		# remove punctuation from each token
		table=string.maketrans(string.punctuation,' '*len(string.punctuation))
		tokens = [w.translate(table) for w in tokens]
		# filter out tokens not in vocab
		tokens = [w for w in tokens if w in vocab]
		clean_lines.append(tokens)
	return clean_lines

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		doc = load_doc(path)
		doc_lines = doc_to_clean_lines(doc, vocab)
		# add lines to list
		lines += doc_lines
	return lines

# load all training reviews
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)



# load training data
positive_lines = process_docs('txt_sentoken/pos', vocab, True)
negative_lines = process_docs('txt_sentoken/neg', vocab, True)
sentences = negative_lines + positive_lines
print('Total training sentences: %d' % len(sentences))

def clean_doc(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table=string.maketrans(string.punctuation,' '*len(string.punctuation))
	tokens=[w.translate(table) for w in tokens]
		
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens
def process_docs(directory, vocab, is_trian):
	documents = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load the doc
		doc = load_doc(path)
		# clean doc
		tokens = clean_doc(doc,vocab)
		# add to list
		documents.append(tokens)
	return documents


# load all training reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, True)
negative_docs = process_docs('txt_sentoken/neg', vocab, True)
train_docs = negative_docs + positive_docs
print train_docs[0]
# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

vocab_size = len(tokenizer.word_index) + 1








from gensim.models import Word2Vec
# train word2vec model
model = Word2Vec(sentences, size=100, window=5, workers=8, min_count=1)
# summarize vocabulary size in model
words = list(model.wv.vocab)
print('Vocabulary size: %d' % len(words))

filename = 'embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)

def load_embedding(filename):
	# load embedding into memory, skip first line
	file = open(filename,'r')
	lines = file.readlines()[1:]
	file.close()
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = asarray(parts[1:], dtype='float32')
	return embedding

	# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = zeros((vocab_size, 100))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		weight_matrix[i] = embedding.get(word)
	return weight_matrix



encoded_docs = tokenizer.texts_to_sequences(train_docs)
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define training labels
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])

# load all test reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, False)
negative_docs = process_docs('txt_sentoken/neg', vocab, False)
test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1



raw_embedding = load_embedding('embedding_word2vec.txt')
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
# create the embedding layer
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)

model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))