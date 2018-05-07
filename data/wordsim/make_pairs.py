from nltk import wordnet
import random

wn = wordnet.wordnet

with open('all_gold_target_words', 'r') as f:
  avoid_these = list(x.strip().split('.')[0] for x in f.readlines())

def add_pair(pairs, word1, word2, sim):
  if word1 in avoid_these or word2 in avoid_these:
    return

  tup = (word1, word2, sim)
  alt_tup = (word2, word1, sim)

  if tup not in pairs and alt_tup not in pairs:
    pairs.append(tup)
    print('%s: %s and %s are ' %('SIM' if sim else 'DIF', word1, word2))

def sufficiently_different_random_word(target):
  synsets = list(wn.all_synsets(target.pos()))

  similar = [target]
  similar.extend(target.hyponyms())
  similar.extend(target.hypernyms())

  current = target

  while current in similar:
    current = random.choice(synsets)

  return current.lemma_names()[0]

if __name__ == '__main__':
  pairs = []

  for synset in wn.all_synsets():
    print(synset)
    if len(synset.lemmas()) < 2:
      continue

    hyponyms = synset.hyponyms()

    word = synset.lemma_names()[0]
    simword = synset.lemma_names()[1]
    randomword = sufficiently_different_random_word(synset)
 
    add_pair(pairs, word, simword, True)
    add_pair(pairs, word, randomword, False)
  
  import pdb; pdb.set_trace()



