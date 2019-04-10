#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

texts = pd.read_csv("texts.csv.bz2", sep='\t')

# note: as the texts may contain various symbols, including tabs and commas,
# you may to have specify 'sep' and maybe other extra arguments


print("Number of texts: %i" % texts.shape[0])
print("Number of titles: %i" % len(np.unique(texts.name.astype('str'))))

# initially use very small samples: for instance, 40 training
# and 10 validation
ntrain = 40
nVal = 10

# for simplicity, I recommend to create dictionary based on the
# merged training and validation data
sentences = np.concatenate((train.chunk.values, val.chunk.values),
                           axis=0)

## initialize the vectorizer
vectorizer = CountVectorizer(min_df=0)
## create the dictionary
vectorizer.fit(sentences)
# `fit` builds the vocabulary
## transform your data into the BOW array
X = vectorizer.transform(sentences).toarray()
# rows are sentences, columns are words

# innspect your X.  Do you have correct number of rows?
# Does the number of words look plausible?

## Implement cosine similarity:
## do it as a function of two vectors
## you have to matrix-multiply the corresponding vectors,
## and divide by their (Euclidean) norms.
## You may use np.linalg.norm for the norm.

## Implement TF-IDF.
## the tf part is just 1 + log-ing the counts
## the idf part is slighly more complex: you need counts of all words
## use something like X.sum(axis = 0) on numpy array to sum the
## elements columnwise.  Now you can easily calculate the
## IDF transform of the counts.
##
## Finally, ensure that tf and idf are multiplied correctly:
## the former is a matrix, the latter a vector.  It should be
## broadcasted automatically, but please check if it is.


