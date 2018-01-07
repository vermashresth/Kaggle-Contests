from nltk.corpus import stopwords
import string
from collections import Counter
from os import listdir

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
print(vocab.most_common(50))

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