import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import scipy.sparse as sp
import itertools


DEFAULT_N = 1


class TfidfStringMatcher(object):
    def __init__(
        self, train_strs_set, ref_strs=None, ignore_new_tokens=False, decode_error='ignore'
    ):
        """
            train_strs_set can be a list of strings or a list of lists of strings
        """
        self.ignore_new_tokens = ignore_new_tokens
        self.tfidf_model = TfidfVectorizer(
            strip_accents='unicode', token_pattern=r'(?u)\b\w+\b', decode_error=decode_error
        )
        if type(train_strs_set[0]) == list:
            train_strs_set = list(itertools.chain.from_iterable(train_strs_set))
        self.tfidf_model.fit(train_strs_set)
        self.tfidf_model.vocabulary_size_ = len(self.tfidf_model.vocabulary_)
        self.tfidf_model.rare_idf_ = self.tfidf_model.idf_.max()
        self.tfidf_ref = None
        if ref_strs is not None:
            self.set_ref_strs(ref_strs)
        else:
            self.ref_strs = []

    def update_vocabulary(self, new_strs):
        _analyzer = self.tfidf_model.build_analyzer()
        new_tokens = list(set(itertools.chain.from_iterable([_analyzer(doc) for doc in new_strs])))
        new_token_count = 0
        for token in new_tokens:
            if token not in self.tfidf_model.vocabulary_:
                new_token_count += 1
                self.tfidf_model.vocabulary_[token] = self.tfidf_model.vocabulary_size_
                self.tfidf_model.vocabulary_size_ += 1
        if new_token_count == 0:
            return None
        A = self.tfidf_model._tfidf._idf_diag
        n = A.shape[0]
        B = np.zeros((n + new_token_count, new_token_count))

        for i in range(new_token_count):
            B[i + n, i] = self.tfidf_model.rare_idf_
        A = sp.vstack([A, np.zeros((new_token_count, n))])
        self.tfidf_model._tfidf._idf_diag = sp.hstack([A, B])
        if self.tfidf_ref is not None:
            self.tfidf_ref = sp.hstack(
                [self.tfidf_ref, np.zeros((self.tfidf_ref.shape[0], new_token_count))]
            )

    def set_ref_strs(self, ref_strs):
        if not self.ignore_new_tokens:
            self.update_vocabulary(ref_strs)
        self.ref_strs = np.array(ref_strs)
        self.tfidf_ref = self.tfidf_model.transform(ref_strs)

    def find_matches(self, queries, n=None, min_score=None, remove_top_match=False):
        """
            queries can be a string or list of strings to match against for self.ref_strs
            if n is not set and min_score is set, all matches with score > min_score
            will be returned
        """
        if n is None and min_score is None:
            n = DEFAULT_N
        if remove_top_match and min_score is None:
            min_score = 0
        if type(queries) == str:
            queries = [queries]

        if not self.ignore_new_tokens:
            self.update_vocabulary(queries)
        tfidf_queries = self.tfidf_model.transform(queries)
        sim_scores_set = linear_kernel(tfidf_queries, self.tfidf_ref)
        match_set = []
        black_list = np.zeros(len(self.ref_strs)) == 1  # black_list is initially all False
        for i in range(len(sim_scores_set)):
            sim_scores = sim_scores_set[i]
            indices = np.array(range(len(self.ref_strs)))
            if min_score is not None:
                ref_strs = self.ref_strs[(sim_scores >= min_score) & (~black_list)]
                indices = indices[(sim_scores >= min_score) & (~black_list)]
                sim_scores = sim_scores[(sim_scores >= min_score) & (~black_list)]
            else:
                ref_strs = self.ref_strs
            ss = (-sim_scores).argsort()
            top_scores = [sim_scores[j] for j in ss[:n]]
            top_matches = [ref_strs[j] for j in ss[:n]]
            top_indices = [indices[j] for j in ss[:n]]

            match_set.append(zip(top_indices, top_matches, top_scores))
            if remove_top_match and len(top_indices) > 0:
                black_list[top_indices[0]] = True
        return match_set

    def find_bulk_matches(self, queries, n=5, batch_size=1000):
        """
            if n is not set and min_score is set, all matches with score > min_score
            will be returned
        """
        if not self.ignore_new_tokens:
            self.update_vocabulary(queries)
        q_len = len(queries)
        tfidf_queries = self.tfidf_model.transform(queries)
        i = 0
        idxs_parts = []
        score_parts = []
        while i < q_len:
            tfidf_queries_sub = tfidf_queries[i:min(i + batch_size, q_len)]
            sim_scores_set = linear_kernel(tfidf_queries_sub, self.tfidf_ref)
            if n == 1:
                score_idxs = sim_scores_set.argmax(1)
                score_part = np.array(
                    [sim_scores_set[np.arange(tfidf_queries_sub.shape[0]), score_idxs]]
                ).T
                score_idxs = np.array([score_idxs]).T
            else:
                score_idxs = np.argpartition(-sim_scores_set, tuple(range(n)))[:, :n]
                score_part = sim_scores_set[np.arange(min(batch_size, q_len)), score_idxs.T].T
            score_parts.append(score_part)
            idxs_parts.append(score_idxs)
            i += batch_size
        scores = np.concatenate(score_parts)
        idxs = np.concatenate(idxs_parts)
        return (idxs, scores)
