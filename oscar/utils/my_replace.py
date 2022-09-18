import random
from types import new_class
import spacy
from nltk.corpus import wordnet as wn
from lemminflect import getInflection
import json

nlp = spacy.load('en_core_web_sm')

random.seed(12345)

REPLACE_TAG = ['NN', 'NNS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] # [NNP, NNPS]
# REPLACE_POS = ['NOUN', 'VERB', 'ADJ', 'ADV']
REPLACE_POS = ['NOUN']
POS_TO_TAGS = {'NOUN': ['NN', 'NNS'], 
               'ADJ': ['JJ', 'JJR', 'JJS'],
               'VERB': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
               'ADV': ['RB', 'RBR', 'RBS']}


LABEL_LIST = 'VG-SGG-dicts-vgoi6-clipped.json'

with open(LABEL_LIST, 'r') as f:
    label_map = json.load(f)

    label_list = list(label_map['label_to_idx'].keys())


from spacy.tokens import Doc
Doc.set_extension('_synonym_sent', default=False, force=True)
Doc.set_extension('_synonym_intv', default=False, force=True)
Doc.set_extension('_ori_syn_intv', default=False, force=True)
Doc.set_extension('_antonym_sent', default=False, force=True)
Doc.set_extension('_antonym_intv', default=False, force=True)
Doc.set_extension('_ori_ant_intv', default=False, force=True)


## word net

def get_synonym(token):
    lemma = token.lemma_
    text = token.text
    tag = token.tag_
    pos = token.pos_
    word_synset = set()
    if pos not in REPLACE_POS:
        return list(word_synset)

    synsets = wn.synsets(text, pos=eval("wn."+pos))
    for synset in synsets:
        words = synset.lemma_names()
        for word in words:
            #word = wnl.lemmatize(word, pos=eval("wn."+pos))
            if word.lower() != text.lower() and word.lower() != lemma.lower():
                # inflt = getInflection(word, tag=tag)
                # word = inflt[0] if len(inflt) else word
                word = word.replace('_', ' ')
                word_synset.add(word)

    return list(word_synset)


def get_hypernyms(token):
    lemma = token.lemma_
    text = token.text
    tag = token.tag_
    pos = token.pos_
    word_hypernyms = set()
    if pos not in REPLACE_POS:
        return list(word_hypernyms)

    synsets = wn.synsets(text, pos=eval("wn."+pos))
    for synset in synsets:
        for hyperset in synset.hypernyms():
            words = hyperset.lemma_names()
            for word in words:
                #word = wnl.lemmatize(word, pos=eval("wn."+pos))
                if word.lower() != text.lower() and word.lower() != lemma.lower():
                    # inflt = getInflection(word, tag=tag)
                    # word = inflt[0] if len(inflt) else word
                    word = word.replace('_', ' ')
                    word_hypernyms.add(word)

    return list(word_hypernyms)


def get_antonym(token):
    lemma = token.lemma_
    text = token.text
    tag = token.tag_
    pos = token.pos_
    word_antonym = set()
    if pos not in REPLACE_POS:
        return list(word_antonym)

    synsets = wn.synsets(text, pos=eval("wn."+pos))
    for synset in synsets:
        for synlemma in synset.lemmas():
            for antonym in synlemma.antonyms():
                word = antonym.name()
                #word = wnl.lemmatize(word, pos=eval("wn."+pos))
                if word.lower() != text.lower() and word.lower() != lemma.lower():
                    # inflt = getInflection(word, tag=tag)
                    # word = inflt[0] if len(inflt) else word
                    word = word.replace('_', ' ')
                    word_antonym.add(word)

    return list(word_antonym)


def get_lemminflect(token):
    text = token.text
    lemma = token.lemma_
    tag = token.tag_
    pos = token.pos_
    word_lemminflect = set()
    if pos not in REPLACE_POS:
        return list(word_lemminflect)

    tags = POS_TO_TAGS[pos]
    for tg in tags:
        if tg == tag: continue
        inflects = getInflection(lemma, tag=tg)
        for word in inflects:
            if word.lower() != text.lower():
                word_lemminflect.add(word)

    return list(word_lemminflect)


## word replace
REPLACE_ORIGINAL = 0
REPLACE_LEMMINFLECT = 1
REPLACE_SYNONYM = 2
REPLACE_HYPERNYMS = 3
REPLACE_ANTONYM = 4
REPLACE_RANDOM = 5
REPLACE_ADJACENCY = 6

REPLACE_NONE = -100

SYNONYM_RATIO = 1/3
HYPERNYMS_RATIO = 1/3
LEMMINFLECT_RATIO = 1/3
ADJ_RATIO = 1/2

def random_noun(word):
    choice_A, choice_B = random.sample(label_list, 2)
    choice = choice_B if choice_A == word.lemma_ else choice_A
    return choice

def random_label(word):
    choice_A, choice_B = random.sample(label_list, 2)
    choice = choice_B if choice_A == word else choice_A
    return choice

def search_replacement(doc, candidate_index, replace_type, max_num):
    sr_rep = []
    if max_num < 1:
        return sr_rep

    for r_idx in candidate_index:
        token = doc[r_idx]
        rep = None
        # ANT words
        if replace_type == REPLACE_RANDOM:
            rep = random_noun(token)
        # Other replace rules
        elif replace_type == REPLACE_SYNONYM:
            reps = get_synonym(token)
            rep = random.choice(reps) if reps else None
        elif replace_type == REPLACE_HYPERNYMS:
            reps = get_hypernyms(token)
            rep = random.choice(reps) if reps else None
        elif replace_type == REPLACE_LEMMINFLECT:
            reps = get_lemminflect(token)
            rep = random.choice(reps) if reps else None
        else:
            pass

        if rep and rep.lower() != token.text.lower():
            sr_rep.append((r_idx, rep, replace_type))

        if len(sr_rep) >= max_num:
            break

    return sr_rep

def replace_word(doc, replace_ratio=0.5):
    synonym_sent = []
    antonym_sent = []

    length = len(doc)

    rep_index = []
    pos_word = {p:[] for p in REPLACE_POS}
    for index, token in enumerate(doc):
        if token.pos_ in REPLACE_POS:
            rep_index.append(index)
            pos_word[token.pos_].append(token.text)

    if len(rep_index) == 0:
        rep_index = random.sample(range(length), int(length*replace_ratio) )
    rep_num = int(len(rep_index) * replace_ratio)

    syn_rand = random.random()

    syn_index = rep_index[:]
    random.shuffle(syn_index)
    ant_index = rep_index[:]
    random.shuffle(ant_index)

    syn_replace = []
    ant_replace = [] # [(rep_idx, rep_word, rep_type)]

    ############### Antonym Replacement ####################
    ant_replace = search_replacement(doc, candidate_index=ant_index, replace_type=REPLACE_RANDOM, max_num=rep_num)

    ############### Synonym Replacement ####################
    # if syn_rand < HYPERNYMS_RATIO:
    #     syn_replace = search_replacement(doc, candidate_index=syn_index, replace_type=REPLACE_HYPERNYMS, max_num=rep_num)

    # if not syn_replace and syn_rand < HYPERNYMS_RATIO + SYNONYM_RATIO:
    #     syn_replace = search_replacement(doc, candidate_index=syn_index, replace_type=REPLACE_SYNONYM, max_num=rep_num)

    # if not syn_replace:
    #     syn_replace = search_replacement(doc, candidate_index=syn_index, replace_type=REPLACE_LEMMINFLECT, max_num=rep_num)

    ############### Replacement ####################
    all_replace = ant_replace + syn_replace
    all_replace = sorted(all_replace, key=lambda x:x[0], reverse=True)

    rep_idx, rep_word, rep_type = all_replace.pop() if all_replace else (None, None, None)

    for index, token in enumerate(doc):
        syn = ant = token.text

        while index == rep_idx:
            if rep_type in [REPLACE_SYNONYM, REPLACE_HYPERNYMS, REPLACE_LEMMINFLECT]:
                syn = rep_word
            elif rep_type in [REPLACE_ANTONYM, REPLACE_RANDOM]:
                ant = rep_word

            rep_idx, rep_word, rep_type = all_replace.pop() if all_replace else (None, None, None)

        synonym_sent.append(syn)
        antonym_sent.append(ant)

    doc._._synonym_sent = synonym_sent
    doc._._antonym_sent = antonym_sent

    return doc

def word_replace(text, ratio=0.5):
    doc = replace_word(nlp(text), ratio)

    syn_sent = " ".join(doc._._synonym_sent)
    ant_sent = " ".join(doc._._antonym_sent)

    return syn_sent, ant_sent

def label_replace(labels, ratio):
    replace_num = int(len(labels) * ratio)

    replace_ids = random.sample(list(range(len(labels))), replace_num)

    new_labels = [lab if i not in replace_ids else random_label(lab) for i, lab in enumerate(labels)]

    return new_labels

if __name__ == '__main__':
    print(word_replace('Sky view of a blue and yellow biplane flying near each other'))
    print(word_replace('Sky view of a blue and yellow biplane flying near each other'))
    print(label_replace(['bird', 'human', 'tree', 'car', 'cat'], ratio=0.5))
