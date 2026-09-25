"""Offline business entity resolution: strict I/O, sparse retrieval, pair features."""
import csv
import re
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

FIELDS = ['entity_id', 'business_name', 'business_address', 'country']

def table(path, fields):
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if reader.fieldnames != fields:
            raise ValueError(f'{path}: expected columns {fields}, got {reader.fieldnames}')
        rows = list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError(f'{path}: malformed TSV row')
    ids = [r[fields[0]] for r in rows]
    if any(not x or x.strip() != x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError(f'{path}: empty, padded or duplicate IDs')
    return rows

def read_sources(directory, split):
    sources = []
    for number in (1, 2, 3):
        rows = table(Path(directory) / f'{split}_source{number}.tsv', FIELDS)
        if any(not r['entity_id'].startswith(f'S{number}-') or ',' in r['entity_id'] for r in rows):
            raise ValueError('Invalid source ID prefix or comma in ID')
        sources.append(rows)
    return sources[0], sources[1] + sources[2]

def id_list(value):
    if value == '':
        return []
    ids = value.split(',')
    if any(not x or x.strip() != x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('Empty, padded or duplicate ID in list')
    return ids

def read_lists(path, column):
    return {r['source1_entity_id']: set(id_list(r[column]))
            for r in table(path, ['source1_entity_id', column])}

def truth_for(path, queries, targets):
    truth = read_lists(path, 'matched_entity_ids')
    if set(truth) != {r['entity_id'] for r in queries}:
        raise ValueError('Ground truth must contain every S1 exactly once')
    valid = {r['entity_id'] for r in targets}
    if any(not ids <= valid for ids in truth.values()):
        raise ValueError('Ground truth contains unknown target IDs')
    return truth

def normalize(value):
    value = ''.join(c for c in unicodedata.normalize('NFKD', value.casefold())
                    if not unicodedata.combining(c))
    return ' '.join(re.sub(r'[^\w\s]', ' ', value.replace('&', ' and ')).split())

def ratio(a, b):
    return SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0.0

def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0

def features(a, b):
    name1, name2 = normalize(a['business_name']), normalize(b['business_name'])
    addr1, addr2 = normalize(a['business_address']), normalize(b['business_address'])
    c1, c2 = normalize(a['country']), normalize(b['country'])
    n1, n2 = set(re.findall(r'\d+', addr1)), set(re.findall(r'\d+', addr2))
    return [ratio(name1, name2), jaccard(set(name1.split()), set(name2.split())),
            float(bool(name1) and name1 == name2), ratio(addr1, addr2),
            jaccard(set(addr1.split()), set(addr2.split())),
            float(bool(addr1) and addr1 == addr2), jaccard(n1, n2),
            float(bool(n1 and n2) and not n1 & n2),
            float(bool(c1 and c2) and c1 == c2), float(bool(c1 and c2) and c1 != c2),
            float(not name1 or not name2), float(not addr1 or not addr2)]

class Retriever:
    """Union top-k positive-cosine neighbors per field; no dense N x M matrix.

    Exact sparse retrieval is a reproducible baseline, not a million-scale ANN index.
    Target vectors are built once per run and reused across all S1 queries.
    """
    def __init__(self, targets, k=20):
        if k < 1:
            raise ValueError('k must be positive')
        self.targets, self.k, self.indices = targets, k, []
        for field in ('business_name', 'business_address'):
            corpus = [normalize(r[field]) for r in targets]
            if not any(corpus):
                continue
            vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5),
                                        dtype=np.float32, max_features=200000)
            matrix = vectorizer.fit_transform(corpus).tocsr()
            self.indices.append((field, vectorizer, matrix))

    def candidates(self, query):
        selected = set()
        for field, vectorizer, matrix in self.indices:
            vector = vectorizer.transform([normalize(query[field])])
            scores = (vector @ matrix.T).tocsr()
            ranked = sorted(zip(scores.indices, scores.data),
                            key=lambda pair: (-pair[1], self.targets[pair[0]]['entity_id']))
            selected.update(i for i, score in ranked[:self.k] if score > 0)
        return sorted(selected, key=lambda i: self.targets[i]['entity_id'])

def macro_f05(truth, predictions):
    if set(truth) != set(predictions) or not truth:
        raise ValueError('Scoring requires identical nonempty S1 sets')
    scores = []
    for key, actual in truth.items():
        predicted = predictions[key]
        if not actual:
            scores.append(float(not predicted))
        else:
            tp = len(actual & predicted)
            scores.append(1.25 * tp / (0.25 * len(actual) + len(predicted)))
    return float(np.mean(scores))

def write_lists(path, queries, values, column):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(['source1_entity_id', column])
        for q in queries:
            writer.writerow([q['entity_id'], ','.join(sorted(values[q['entity_id']]))])

def validate(directory, queries, targets):
    candidates = read_lists(Path(directory) / 'candidate_pairs.tsv', 'candidate_entity_ids')
    matches = read_lists(Path(directory) / 'matching_results.tsv', 'matched_entity_ids')
    expected, valid = {q['entity_id'] for q in queries}, {t['entity_id'] for t in targets}
    if set(candidates) != expected or set(matches) != expected:
        raise ValueError('Outputs must contain all and only S1 IDs')
    for key in expected:
        if not matches[key] <= candidates[key] <= valid:
            raise ValueError(f'{key}: matches outside candidates or invalid target IDs')
    return matches
