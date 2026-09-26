"""NFKC plus mark-preserving normalization; configurable conservative mappings."""
import unicodedata
from transliterate import transliterate, script_label

DEFAULT={
 'name_mappings':{'pvt':'private','ltd':'limited','corp':'corporation','inc':'incorporated','co':'company'},
 'address_mappings':{'rd':'road','ave':'avenue','ln':'lane','hwy':'highway','apt':'apartment'},
 'ambiguous_st_to_street':False,
 'suffixes':['private','limited','corporation','incorporated','company','llc','llp']}

def clean(text):
    text=unicodedata.normalize('NFKC',text).casefold().replace('&',' and ')
    # Preserve combining marks (e.g. Devanagari vowel signs), letters and numbers.
    text=''.join(c if unicodedata.category(c)[0] in 'LNM' or c.isspace() else ' ' for c in text)
    return ' '.join(text.split())

def normalize_record(row,config=None,trans_config=None):
    cfg=DEFAULT if config is None else config
    enabled=True if trans_config is None else trans_config['enabled']
    out=dict(row)
    for field,kind in [('business_name','name'),('business_address','address')]:
        out[field+'_raw']=row[field]
        mappings=dict(cfg[kind+'_mappings'])
        if kind=='address' and cfg.get('ambiguous_st_to_street'):mappings['st']='street'
        value=' '.join(mappings.get(t,t) for t in clean(row[field]).split())
        out[field+'_norm']=value
        out[field+'_transliterated']=clean(transliterate(value)) if enabled else ''
    tokens=out['business_name_norm'].split()
    while tokens and tokens[-1] in cfg['suffixes']:tokens.pop()
    out['business_name_core']=' '.join(tokens)
    out['country_norm']=clean(row['country'])
    out['script']=script_label(row['business_name'])
    return out
