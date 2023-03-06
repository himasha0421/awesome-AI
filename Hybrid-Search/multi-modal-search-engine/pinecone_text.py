from typing import Callable, Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import HashingVectorizer

SparseVector = Dict[str, List]


class BM25(BaseEstimator):

    def __init__(self, tokenizer: Callable[[str], List[str]], n_features=2 ** 16, b=0.75, k1=1.6):
        """OKapi BM25 with HashingVectorizer

        Args:
            tokenizer: A function to converts text to a list of tokens
            n_features: The number of features to hash to
            b: BM25 parameters
            k1: BM25 parameters
        """
        # Fixed params
        self.n_features: int = n_features
        self.b: float = b
        self.k1: float = k1

        self._tokenizer: Callable[[str], List[str]] = tokenizer
        self._vectorizer = HashingVectorizer(
            n_features=self.n_features,
            token_pattern=None,
            tokenizer=tokenizer, norm=None,
            alternate_sign=False, binary=True)
        # Learned Params
        self.doc_freq: Optional[np.ndarray] = None
        self.n_docs: Optional[int] = None
        self.avgdl: Optional[float] = None

    def fit(self, X: List[str], y=None) -> "BM25":
        """Fit BM25 by calculating document frequency over the corpus"""
        X = self._vectorizer.transform(X)
        self.avgdl = X.sum(1).mean()
        self.n_docs = X.shape[0]
        self.doc_freq = X.sum(axis=0).A1
        return self

    def vectorize(self, text) -> SparseVector:
        sparse_array = self._vectorizer.transform(text)
        return {'indices': [int(x) for x in sparse_array.indices], 'values': sparse_array.data.tolist()}

    def get_params(self, deep=True):
        return {
            'avgdl': self.avgdl,
            'ndocs': self.n_docs,
            'doc_freq': list(self.doc_freq),
            'b': self.b,
            'k1': self.k1,
            'n_features': self.n_features
        }

    def set_params(self, **params):
        self.avgdl = params['avgdl']
        self.n_docs = params['ndocs']
        self.doc_freq = np.array(params['doc_freq'])
        self.b = params['b']
        self.k1 = params['k1']
        self.n_features = params['n_features']

    def transform_doc(self, doc: str) -> SparseVector:
        """Normalize document for BM25 scoring"""
        doc_tf = self._vectorizer.transform([doc])
        norm_doc_tf = self._norm_doc_tf(doc_tf)
        return {'indices': [int(x) for x in doc_tf.indices], 'values': norm_doc_tf.tolist()}

    def transform_query(self, query: str) -> SparseVector:
        """Normalize query for BM25 scoring"""
        query_tf = self._vectorizer.transform([query])
        indices, values = self._norm_query_tf(query_tf)
        return {'indices': [int(x) for x in indices], 'values': values.tolist()}

    def _norm_doc_tf(self, doc_tf) -> np.ndarray:
        """Calculate BM25 normalized document term-frequencies"""
        b, k1, avgdl = self.b, self.k1, self.avgdl
        tf = doc_tf.data
        norm_tf = tf / (k1 * (1.0 - b + b * (tf.sum() / avgdl)) + tf)
        return norm_tf

    def _norm_query_tf(self, query_tf):
        """Calculate BM25 normalized query term-frequencies"""
        idf = np.log((self.n_docs + 1) / (self.doc_freq[query_tf.indices] + 0.5))
        return query_tf.indices, idf / idf.sum()
