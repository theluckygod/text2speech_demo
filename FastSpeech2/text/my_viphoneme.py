from re import sub
from viphoneme import vi2IPA_split
import logging

consonants = ['tʰw', 'ŋ͡m', 'k͡p', 'cw', 'jw', 'bw', 'vw', 'ʈw', 'ʂw', 'fw', 'tʰ', 'tʃ', 'xw', 'ŋw', 'dw', 'ɣw', 'zw', 'mw', 'hw', 'lw', 'kw', 'nw', 't∫', 'ɲw', 'sw', 'tw', 'ʐw', 'dʒ', 'ɲ', 'θ', 'l', 'w', 'd', '∫', 'p', 'ɣ', '!', 'ð', 'ʧ', 'ʒ', 'ʐ', 'z', 'v', 'g', '_', 'ʤ', '.', 'b', 'h', 'n', 'ʂ', 'k', 'm', ' ', 'c', 'j', 'x', 'ʈ', ',', 's', 'ŋ', 'ʃ', '?', 'r', ':', 'η', 'f', ';', 't', "'"]
vowels = ['oʊ', 'ɯəj', 'ɤ̆j', 'ʷiə', 'ɤ̆w', 'ɯəw', 'ʷet', 'iəw', 'uəj', 'ʷen', 'ʷɤ̆', 'ʷiu', 'kwi', 'uə', 'eə', 'oj', 'ʷi', 'ăw', 'aʊ', 'ɛu', 'ɔɪ', 'ʷɤ', 'ɤ̆', 'ʊə', 'zi', 'ʷă', 'eɪ', 'aɪ', 'ew', 'iə', 'ɯj', 'ʷɛ', 'ɯw', 'ɤj', 'ɔ:', 'əʊ', 'ʷa', 'ɑ:', 'ɔj', 'uj', 'ɪə', 'ăj', 'u:', 'aw', 'ɛj', 'iw', 'aj', 'ɜ:', 'eo', 'iɛ', 'ʷe', 'i:', 'ɯə', 'ʌ', 'ɪ', 'ɯ', 'ə', 'u', 'o', 'ă', 'æ', 'ɤ', 'i', 'ɒ', 'ɔ', 'ɛ', 'ʊ', 'a', 'e']
en_to_vi = {'p': ['p'], 'b': ['b'], 't': ['t'], 'd': ['d'], 't∫': [u'tʃ'], 'ʧ': ['c'], 'dʒ': ['c'], 'ʤ': ['c'], 'k': ['k'], 'g': ['ɣ'], 'f': ['f'], 'v': ['v'], 'ð': ['d'], 'θ': ['tʰ'], 's': ['s'], 'z': ['j'], '∫': ['ʂ'], 'ʃ': ['ʂ'], 'ʒ': ['z'], 'm': ['m'], 'n': ['n'], 'η': ['ŋ'], 'l': ['l'], 'r': ['ʐ'], 'w': ['kw'], 'j': ['j'], 'ɪ': ['i'], 'i:': ['i'], 'ʊ': ['ɯə', 'k'], 'u:': ['u'], 'e': ['ɛ'], 'ɛ': ['ɤ'], 'ə': ['ɤ'], 'ɜ:': ['ɤ'], 'ɒ': ['ɔ'], 'ɔ:': ['o'], 'æ': ['a'], 'ʌ': ['ɤ̆'], 'ɑ:': ['ɔ'], 'ɪə': ['iə'], 'ʊə': ['uə'], 'eə': ['ɛ'], 'eɪ': ['ă', 'j'], 'ɔɪ': ['o', 'j'], 'aɪ': ['a', 'j'], 'oʊ': ['ɤ̆', 'w'], 'aʊ': ['a', 'w']}
onglides = ['ʷa', 'ʷa', 'ʷa', 'ʷa', 'ʷa', 'ʷa', 'ʷa', 'ʷa', 'ʷa', 'ʷa', 'ʷa', 'ʷă', 'ʷă', 'ʷă', 'ʷă', 'ʷă', 'ʷă', 'ʷɛ', 'ʷɛ', 'ʷɛ', 'ʷɛ', 'ʷɛ', 'ʷɛ', 'ʷɛ', 'ʷɛ', 'ʷɛ', 'ʷɛ', 'ʷɛ', 'ʷa', 'ʷa', 'ʷa', 'ʷa', 'ʷa', 'ʷa', 'ʷă', 'ʷă', 'ʷă', 'ʷă', 'ʷă', 'ʷă', 'ʷɤ̆', 'ʷɤ̆', 'ʷɤ̆', 'ʷɤ̆', 'ʷɤ̆', 'ʷɤ̆', 'ʷɛ', 'ʷɛ', 'ʷɛ', 'ʷɛ', 'ʷɛ', 'ʷɛ', 'ʷe', 'ʷe', 'ʷe', 'ʷe', 'ʷe', 'ʷe', 'ʷɤ', 'ʷɤ', 'ʷɤ', 'ʷɤ', 'ʷɤ', 'ʷɤ', 'ʷi', 'ʷi', 'ʷi', 'ʷi', 'ʷi', 'ʷi', 'ʷiə', 'ʷiə', 'ʷiə', 'ʷiə', 'ʷiə', 'ʷiə', 'ʷiə', 'ʷiə', 'ʷiə', 'ʷiə', 'ʷiə', 'ʷiə', 'ʷiu', 'ʷiu', 'ʷiu', 'ʷiu', 'ʷiu', 'ʷiu', 'ʷiu', 'ʷiu', 'ʷiu', 'ʷiu', 'ʷiu', 'ʷen', 'ʷen', 'ʷen', 'ʷen', 'ʷen', 'ʷen', 'ʷet', 'ʷet', 'ʷet', 'ʷet', 'ʷet', 'ʷet']
codas = ['p', 't', 'k', 'm', 'n', 'ŋ', 'ɲ', 'tʃ']
onset = { u'tr' : u'ʈ' }

custom_vowels = ['iu']
custom_vowels2 = ['wɤ̆', 'wiu', 'wɤ', 'wă', 'wɛ', 'wa', 'we', 'wiə', 'wi']
tones = ['1', '3', '6', '2', '5', '4']

#vowels = vowels + custom_vowels + custom_vowels2

def normalize_phs(phs_list):
    i = 0
    while i < len(phs_list):
        # Check duplicate conda
        if phs_list[i] in codas:
            while (i < len(phs_list) - 1) and phs_list[i+1] in codas:
                del phs_list[i+1]

        # Modify onglide
        if phs_list[i] in onglides:
            phs_list[i] = phs_list[i][1:]
            if len(phs_list[i-1]) > 1:
                phs_list[i-1] = phs_list[i-1][:-1]
                phs_list.insert(i, 'w')

        i+=1
    return phs_list

def parse_sub_word(phs):
    # ['', 'ɛ', 'k', "'", '5', ' ', 's', 'p', 'ɛ', 't', "'", '1']
    # -> [['ɛ', 'k', "'", '5'], ['s', 'p', 'ɛ', 't', "'", '1']]
    res_phs = []

    start = 0
    phs.append(' ')
    for p in range(len(phs)):
        if phs[p] == ' ' or phs[p] == '':
            if p > start:
                res_phs.append(phs[start:p])
            start = p + 1
    return res_phs

def cvt_format_en2vi(sub_word):
    # ['s', 'p', 'ɛ', 't', "'", '1'] -> ['s', 'ɤ2', 'p', 'ɛ5', 't']
    tone = sub_word[-1]
    sub_word = sub_word[:-2] # skip ["'", '1'] 
    if tone == '5':
        tone = '5'
    elif tone == '1':
        tone = '2'
    else:
        logging.warn(f"Invalid tone {tone} in En2Vi")
        tone = '2'
    pos = 0 # 0: vị trí phụ âm, 1: vị trí nguyên âm, 2: vị trí đuôi
    p = 0
    while p < len(sub_word):
        if pos == 0 and sub_word[p] in ['s', 'g'] and sub_word[p + 1] not in vowels + ['w']: # s -> sờ
            sub_word = sub_word[:p + 1] + ['ɤ2'] + sub_word[p + 1:]
            p += 1
        elif sub_word[p] in vowels or sub_word[p] == 'w':
            if sub_word[p] != 'w':
                sub_word[p] += tone
            pos = 2
        elif pos == 2:
            if sub_word[p] in ['v', 'f']:
                return sub_word[:p] + ['p']
            elif sub_word[p] not in ['ɲ', 'm', 'j', 'tʃ', 'k', 'n', 'ŋ', 'p', 't', 'k͡p', 'ŋ͡m']:
                return sub_word[:p] + ['t']
            else:
                return sub_word[:p + 1]
        p += 1
    return sub_word

def vi2IPA_en2vi(sentence, delimit):
    vi_union_en = ['e', 'd', 'm', 'w', 'v', 'l', 't', 'b', 'n', 'j', 'f', 'k', 'z', 's', 'p', 'g']
    english_phoneme = en_to_vi.keys()
    phs = vi2IPA_split(sentence, delimit)[:-9].split(delimit)
    if '\'' not in phs:
        return phs

    i = 0
    while i < len(phs):
        # Check 'tr'
        if (i < len(phs) - 1) and ((phs[i]+phs[i+1]) in onset.keys()):
            en_phoneme = phs[i]+phs[i+1]
            phs[i] = onset[en_phoneme]
            del phs[i+1]
            i+=1
            continue
        # Check 2 vowel
        if (i < len(phs) - 1) and ((phs[i]+phs[i+1]) in ['ɪə', 'ʊə', 'eə', 'eɪ', 'ɔɪ', 'aɪ', 'oʊ', 'aʊ']):
            en_phoneme = phs[i]+phs[i+1]
            phs[i], phs[i+1] = en_to_vi[en_phoneme]
            i+=2
            # Delete codas
            while i < len(phs) and (phs[i] != '\'' and phs[i] != '.' and phs[i] != '1' and phs[i] != '5'):
                del phs[i]
            continue

        # Check 1 vowel
        if phs[i] in english_phoneme and phs[i] not in vi_union_en:
            phs[i] = en_to_vi[phs[i]][0]

        i+=1

    cvt_phs = []
    for w in parse_sub_word(phs):
        cvt_w = cvt_format_en2vi(w)
        cvt_phs += cvt_w
    phs = cvt_phs

    phs = normalize_phs(phs)

    return phs
    
def _is_concat_2_vowels(v1, v2):
    if v1 + v2 in vowels:
        return True
    return False

def get_all_syms():
    all_vowels = []
    for v in vowels:
        for t in tones:
            all_vowels.append(v + t)
    return consonants + all_vowels

def get_my_viphoneme_list(ph_list):
    #Returns my viphoneme in list. ex. ['t', 'o', 'j', '5'] -> ['t', 'o5', 'j']

    #Parameters:
    #   ph_list (list): phonemes of words which is parsed by viphoneme

    #Returns:
    #   my_ph (list): my style phonemes

    keep_vowel = False
    should_removed_list = []
    for i in range(len(ph_list)):
        if ph_list[i] in vowels: # find vowels
            if keep_vowel == True:
                if i_keep == i - 1 and _is_concat_2_vowels(ph_list[i_keep], ph_list[i]): # concat 2 vowels
                    ph_list[i_keep] += ph_list[i]
                    should_removed_list.append(i)
                else:
                    ph_list[i_keep] += '1'
                    i_keep = i
            else: # find vowels
                keep_vowel = True
                i_keep = i
        elif ph_list[i] in tones and keep_vowel == True: # concat tone to vowel
            ph_list[i_keep] += ph_list[i]
            should_removed_list.append(i)
            keep_vowel = False
        elif ph_list[i] in ["'", " ", "_"]: # remove special symbol
            should_removed_list.append(i)

    if keep_vowel == True:
        ph_list[i_keep] += '1'

    for i in should_removed_list[::-1]:
        ph_list.pop(i)

    return ph_list

def get_cleaned_viphoneme_list(sentence):
    #Returns cleaned viphoneme in list. ex. "tôi là ai" -> ['t', 'o', 'j', '1', ' ', 'l', 'a', '2', ' ', 'a', 'j', '1']

    #Parameters:
    #   sentence (str): a sentence

    #Returns:
    #   cleaned_ph (list): a list of phonemes which have been cleaned
    cleaned_phs = vi2IPA_en2vi(sentence, '/')#[:-9].split('/')
    if len(cleaned_phs) > 0 and cleaned_phs[0] == '':
        cleaned_phs = cleaned_phs[1:]
    return cleaned_phs
