"""Similarity primitives; blank/blank never becomes positive identity evidence."""
import re
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein,Jaro,JaroWinkler

def exact(a,b):return float(bool(a) and a==b)
def levenshtein(a,b):return float(Levenshtein.normalized_similarity(a,b)) if a and b else 0.
def jaro(a,b):return float(Jaro.normalized_similarity(a,b)) if a and b else 0.
def jaro_winkler(a,b):return float(JaroWinkler.normalized_similarity(a,b)) if a and b else 0.
def jaccard(a,b):
    a,b=set(a),set(b);return len(a&b)/len(a|b) if a and b else 0.
def overlap(a,b):
    a,b=set(a),set(b);return len(a&b)/min(len(a),len(b)) if a and b else 0.
def containment(a,b):
    a,b=set(a),set(b);return float(bool(a and b) and (a<=b or b<=a))
def token_set(a,b):return fuzz.token_set_ratio(a,b)/100 if a and b else 0.
def token_sort(a,b):return fuzz.token_sort_ratio(a,b)/100 if a and b else 0.
def length_ratio(a,b):return min(len(a),len(b))/max(len(a),len(b)) if a and b else 0.
def prefix(a,b):
    n=0
    for x,y in zip(a,b):
        if x!=y:break
        n+=1
    return n/min(len(a),len(b)) if a and b else 0.
def numbers(a):return set(re.findall(r'\b\d+\b',a))
def postal(a):return set(re.findall(r'\b\d{5,6}\b',a))
def house(a):
    found=re.search(r'\b\d+[a-z]?\b',a);return found.group() if found else ''
def cosine_pairs(vectorizer,left,right):
    import numpy as np
    if vectorizer is None:return np.zeros(len(left),dtype=np.float32)
    return np.asarray(vectorizer.transform(left).multiply(vectorizer.transform(right)).sum(axis=1)).ravel().astype(np.float32)
