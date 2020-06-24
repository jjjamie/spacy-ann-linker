# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Adapted from https://github.com/allenai/scispacy/blob/master/scispacy/candidate_generation.py
# for use with spaCy KnowledgeBase

from typing import List, Dict, Set, Tuple
import json
from collections import defaultdict
from pathlib import Path
import joblib
import hnswlib
import numpy as np
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from spacy.kb import Candidate, KnowledgeBase
from spacy.tokens import Doc, Span
from spacy.util import ensure_path, to_disk, from_disk
import srsly
import fasttext
from wasabi import Printer
from spacy_ann.types import AliasCandidate
from jbpython import Timer, get_logger


_log = get_logger(__name__)


class CandidateGeneratorFasttext:
    def __init__(self,
                 fasttext_model: fasttext.FastText,
                 *,
                 k: int = 5,
                 m_parameter: int = 100,
                 ef_search: int = 200,
                 ef_construction: int = 2000,
                 n_threads: int = 60,
            ):
        """Initialize a CandidateGenerator
        
        k (int): Number of neighbors to query
        m_parameter (int): M parameter value for nmslib hnsw algorithm
        ef_search (int): Set to the maximum recommended value. 
            Improves recall at the expense of longer **inference** time
        ef_construction (int): Set to the maximum recommended value. 
            Improves recall at the expense of longer **indexing** time
        n_threads (int): Number of threads to use when creating the index. 
            Change based on your machine.
        """
        self.k = k
        self.m_parameter = m_parameter
        self.ef_search = ef_search
        self.ef_construction = ef_construction
        self.n_threads = n_threads
        self.ann_index = None
        self.fasttext_model: fasttext.FastText = fasttext_model
        
    def _initialize(self,
                    aliases: List[str],
                    short_aliases: Set[str],
                    ann_index: hnswlib.Index,
                    alias_vectors: np.array):
        """Used in `fit` and `from_disk` to initialize the CandidateGenerator with computed
        # TF-IDF Vectorizer and ANN Index
        
        aliases (List[str]): Aliases with vectors contained in the ANN Index
        short_aliases (Set[str]): Aliases too short for a TF-IDF representation
        ann_index (FloatIndex): Computed ANN Index of TF-IDF representations for aliases
        alias_vectors (scipy.sparse.csr_matrix): Computed TF-IDF Sparse Vectors for aliases
        """
        self.aliases = aliases
        self.short_aliases = short_aliases
        self.ann_index = ann_index
        self.alias_vectors = alias_vectors


    def _fit_ann_index_hnswlib(self, alias_vectors: scipy.sparse.csr_matrix):
        # nmslib hyperparameters (very important)
        # guide: https://github.com/nmslib/nmslib/blob/master/python_bindings/parameters.md
        # m_parameter = 100
        # # `C` for Construction. Set to the maximum recommended value
        # # Improves recall at the expense of longer indexing time
        # construction = 2000
        # num_threads = 60  # set based on the machine
        (samples, features) = alias_vectors.shape
        _log.info(f"Fitting ann index on {samples} aliases")
        with Timer() as t:
            ann_index = hnswlib.Index('cosine', features)
            ann_index.init_index(samples, self.ef_construction, self.m_parameter, random_seed = 2)
            _log.info(f"{alias_vectors.shape}")
            ann_index.add_items(alias_vectors)
            ann_index.set_ef(self.ef_search)

        _log.info(f"Fitting ann index took {round(t.interval)} seconds")
        return ann_index

    def _get_vectorized(self, kb_aliases: List[str]):
        _log.info(f"Getting fasttext vectors for {len(kb_aliases)} aliases")
        with Timer() as t:
            alias_vectors = self._vectorize_aliases(kb_aliases)

        _log.info(f"Getting vectors took {round(t.interval)} seconds")

        assert len(kb_aliases) == np.size(alias_vectors, 0)

        return kb_aliases, alias_vectors

    def _vectorize_aliases(self, aliases: List[str]):
        alias_vectors = np.zeros((len(aliases), self.fasttext_model.get_dimension()))
        for i, a in enumerate(aliases):
            vector_for_alias = self.fasttext_model[a]
            alias_vectors[i,:] = vector_for_alias
        return alias_vectors

    def fit(self, kb_aliases: List[str]):
        """Build tfidf vectorizer and ann index.
        Warning: Running this function can take a lot of memory
        
        kb_aliases (List[str]): Aliases in the KnoweledgeBase to fit the ANN index on.
        
        RETURNS (CandidateGenerator): An initialized CandidateGenerator
        """        
        short_aliases = set([a for a in kb_aliases if len(a) < 4])

        # NOTE: here we are creating the tf-idf vectorizer with float32 type, but we can serialize the
        # resulting vectors using float16, meaning they take up half the memory on disk. Unfortunately
        # we can't use the float16 format to actually run the vectorizer, because of this bug in sparse
        # matrix representations in scipy: https://github.com/scipy/scipy/issues/7408
        
        aliases, alias_vectors = self._get_vectorized(kb_aliases)

        ann_index = self._fit_ann_index_hnswlib(alias_vectors)

        self._initialize(aliases, short_aliases, ann_index, alias_vectors)
        return self

    def _nmslib_knn_with_zero_vectors(
        self, vectors: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ann_index.knn_query crashes if any of the vectors is all zeros.
        This function is a wrapper around `ann_index.knn_query` that solves this problem. It works as follows:
        - remove empty vectors from `vectors`.
        - call `ann_index.knn_query` with the non-empty vectors only. This returns `neighbors`,
        a list of list of neighbors. `len(neighbors)` equals the length of the non-empty vectors.
        - extend the list `neighbors` with `None`s in place of empty vectors.
        - return the extended list of neighbors and distances.
        
        vectors (np.ndarray): Vectors used to query index for neighbors and distances
        k (int): k neighbors to consider
        
        RETURNS (Tuple[np.ndarray, np.ndarray]): Tuple of [neighbors, distances]
        """                

        empty_vectors_boolean_flags = np.array(vectors.sum(axis=1) != 0).reshape(-1,)
        empty_vectors_count = vectors.shape[0] - sum(empty_vectors_boolean_flags)

        # init extended_neighbors with a list of Nones
        extended_neighbors = np.empty((len(empty_vectors_boolean_flags),), dtype=object)
        extended_distances = np.empty((len(empty_vectors_boolean_flags),), dtype=object)

        if vectors.shape[0] - empty_vectors_count == 0:
            return extended_neighbors, extended_distances

        # remove empty vectors before calling `ann_index.knn_query`
        vectors = vectors[empty_vectors_boolean_flags]

        # call `knn_query` to get neighbors
        original_neighbours = self.ann_index.knn_query(vectors, k=k)
        # print(original_neighbours)

        neighbors, distances = original_neighbours
        # zip(*[(x[0].tolist(), x[1].tolist()) for x in original_neighbours])
        neighbors = list(neighbors)
        distances = list(distances)

        # neighbors need to be converted to an np.array of objects instead of ndarray of dimensions len(vectors)xk
        # Solution: add a row to `neighbors` with any length other than k. This way, calling np.array(neighbors)
        # returns an np.array of objects
        neighbors.append([])
        distances.append([])
        # interleave `neighbors` and Nones in `extended_neighbors`
        extended_neighbors[empty_vectors_boolean_flags] = np.array(neighbors)[:-1]
        extended_distances[empty_vectors_boolean_flags] = np.array(distances)[:-1]

        return extended_neighbors, extended_distances
    
    def require_ann_index(self):
        """Raise an error if the ann_index is not initialized
        
        RAISES:
            ValueError: ann_index not initialized
        """        
        # 
        if getattr(self, "ann_index", None) in (None, True, False):
            raise ValueError(f"ann_index not initialized. Have you run `cg.train` yet?")

    def __call__(self, mention_texts: List[str]) -> List[List[AliasCandidate]]:
        """Generate AliasCandidates for each mention in a batch of entity mentions.
        
        mention_texts (List[str]): List of entity mentions to generate AliasCandidates for
        
        RETURNS (List[List[AliasCandidate]]): List of AliasCandidates for each mention
        """        
        self.require_ann_index()

        # tfidf vectorizer crashes on an empty array, so we return early here
        if mention_texts == []:
            return []

        alias_vectors = self._vectorize_aliases(mention_texts)

        # `ann_index.knnQueryBatch` crashes if one of the vectors is all zeros.
        # `nmslib_knn_with_zero_vectors` is a wrapper around `ann_index.knnQueryBatch`
        # that addresses this issue.
        batch_neighbors, batch_distances = self._nmslib_knn_with_zero_vectors(alias_vectors, self.k)

        batch_candidates = []
        for mention, neighbors, distances in zip(
            mention_texts, batch_neighbors, batch_distances
        ):
            if mention in self.short_aliases:
                batch_candidates.append([AliasCandidate(alias=mention, similarity=1.0)])
                continue
            if neighbors is None:
                neighbors = []
            if distances is None:
                distances = []

            alias_candidates = []
            for neighbor_index, distance in zip(neighbors, distances):
                alias = self.aliases[neighbor_index]
                similarity = 1.0 - distance
                alias_candidates.append(AliasCandidate(alias=alias, similarity=similarity))

            batch_candidates.append(alias_candidates)

        return batch_candidates

    def from_disk(self, path: Path, **kwargs):
        """Deserialize CandidateGenerator data from disk
        
        path (Path): Directory to deserialize data from
        
        RETURNS (CandidateGenerator): Initialized Candidate Generator
        """        
        aliases_path = f"{path}/aliases.json"
        short_aliases_path = f"{path}/short_aliases.json"
        ann_index_path = f"{path}/ann_index.bin"
        tfidf_vectors_path = f"{path}/tfidf_vectors_sparse.npz"

        cfg = {}
        deserializers = {"cg_cfg": lambda p: cfg.update(srsly.read_json(p))}
        from_disk(path, deserializers, {})

        self.k = cfg.get("k", 5)
        self.m_parameter = cfg.get("m_parameter", 100)
        self.ef_search = cfg.get("ef_search", 200)
        self.ef_construction = cfg.get("ef_construction", 2000)
        self.n_threads = cfg.get("n_threads", 60)

        aliases = srsly.read_json(aliases_path)
        short_aliases = srsly.read_json(short_aliases_path)
        alias_vectors = scipy.load_npz(tfidf_vectors_path).astype(np.float32)

        ann_index = hnswlib.Index(space='cosine', dim=alias_vectors.shape[1])
        ann_index.set_num_threads(self.n_threads)
        ann_index.load_index(str(ann_index_path))
        ann_index.set_ef(self.ef_search)
    
        self._initialize(aliases, short_aliases, ann_index, alias_vectors)

        return self

    def to_disk(self, path: Path, **kwargs):
        """Serialize CandidateGenerator to disk
        
        path (Path): Directory to serialize to
        """
        cfg = {
            "k": self.k,
            "m_parameter": self.m_parameter,
            "ef_search": self.ef_search,
            "ef_construction": self.ef_construction,
            "n_threads": self.n_threads
        }
        serializers = {
            "cg_cfg": lambda p: srsly.write_json(p, cfg),
            "aliases": lambda p: srsly.write_json(p.with_suffix(".json"), self.aliases),
            "short_aliases": lambda p: srsly.write_json(p.with_suffix(".json"), self.short_aliases),
            "ann_index": lambda p: self.ann_index.save_index(str(p.with_suffix(".bin"))),
            "fasttext_vectors": lambda p: scipy.save_npz(
                p.with_suffix(".npz"), self.alias_vectors.astype(np.float16)
            ),
        }

        to_disk(path, serializers, {})
