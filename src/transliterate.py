"""Offline AnyAscii (ISC) transliteration; never replaces Unicode fields."""
import unicodedata
from anyascii import anyascii

def transliterate(text):
    return anyascii(text).casefold()

def scripts(text):
    found=set()
    for c in text:
        if not unicodedata.category(c).startswith('L'): continue
        name=unicodedata.name(c,'')
        script=next((s for s in ('LATIN','DEVANAGARI','BENGALI','GURMUKHI','GUJARATI','TAMIL','TELUGU','KANNADA','MALAYALAM','ARABIC','CYRILLIC','GREEK','HEBREW','HANGUL','HIRAGANA','KATAKANA') if s in name),'HAN' if 'CJK' in name else 'OTHER')
        found.add(script)
    return sorted(found)

def script_label(text):
    s=scripts(text)
    return '+'.join(s) if s else 'NONE'
