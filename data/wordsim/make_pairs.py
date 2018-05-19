from nltk import wordnet
import random
import editdistance
import itertools
import codecs
import glob
import os
import pickle
import numpy as np
from scipy.spatial.distance import cdist

wn = wordnet.wordnet

random.seed(1234)
np.random.seed(1234)

def get_image_features_for_word(word_to_path_dict, word):
  list_of_features = []
  word = word.replace('_', ' ')
  if word in word_to_path_dict:
    for f_name in glob.glob(os.path.join(word_to_path_dict[word], '*.pkl')):
      with open(f_name, 'rb') as fp:
        vect = pickle.load(fp, encoding='latin1')
        if not np.any(np.isnan(vect)):
          list_of_features.append(vect)
    return np.array(list_of_features)
  else:
    return None


def get_feature_paths_dict():
  # FEATURE_PATHS = '/home/daphnei/nlpgrid/word_absolute_paths.tsv'
  FEATURE_PATHS = '/nlp/data/bcal/features/word_absolute_paths.tsv'
  output = {}
  # replace = ('/nlp/data/bcal/features/', '/home/daphnei/nlpgrid/')
  # with open(FEATURE_PATHS, 'r') as f:
  with codecs.open(FEATURE_PATHS, encoding="utf-8") as f:
    for line in f:
      word, path = line.strip().split('\t')
      # path = path.replace(replace[0], replace[1])
      output[word] = path
  return output

def get_alternative_distances(word1, word2, word_to_features):
  features_1 = get_image_features_for_word(word_to_features, word1)
  features_2 = get_image_features_for_word(word_to_features, word2)

  if features_1 is not None and len(features_1) > 0 and features_2 is not None and len(features_2) > 0:
    similarities = 1 - cdist(features_1, features_2, metric='cosine') 
    average_max1 = np.sum(np.max(similarities, axis=1)) / features_1.shape[0]
    average_max2 = np.sum(np.max(similarities, axis=0)) / features_2.shape[0]
    averaged_similarity = 0.5 * (average_max1 + average_max2)
    return averaged_similarity
  else:
    return None

def add_pair(pairs, word1, word2, sim, config):
  if word1 in config['avoid_these'] or word2 in config['avoid_these']:
    return

  dist = get_alternative_distances(
      word1, word2, config['word_to_feature_paths_dict'])
  if dist is None:
    return

  tup = (word1, word2, sim)
  alt_tups = [(word1, word2, True), (word1, word2, False), (word2, word1, True), (word2, word1, False)]

  if all(t not in pairs.keys() for t in alt_tups):
    pairs[tup] = [dist]
    print('%s: %s and %s' %('SIM' if sim else 'DIF', word1, word2))

synset_cache = {}
def get_synset_cached(pos):
  if pos in synset_cache:
      return synset_cache[pos]
  else:
    l = list(wn.all_synsets(pos))
    synset_cache[pos] = l
    return l

def sufficiently_different_random_word(target, config):
  synsets = get_synset_cached(target.pos())

  similar = [target]
  similar.extend(target.hyponyms())
  similar.extend(target.hypernyms())

  current = target

  while current in similar or current.lemma_names()[0] not in config['word_to_feature_paths_dict']:
    current = random.choice(synsets)

  return current.lemma_names()[0]

if __name__ == '__main__':
  train_pairs = {}
  test_pairs = {}

  random.seed(1234)

  config = {}
  with open('all_gold_target_words', 'r') as f:
    config['avoid_these'] = list(x.strip().split('.')[0] for x in f.readlines())

  config['word_to_feature_paths_dict'] = get_feature_paths_dict()

  for synset in wn.all_synsets():
    if len(synset.lemmas()) < 2:
      continue

    hyponyms = synset.hyponyms()

    all_pairs = list(itertools.combinations(synset.lemma_names(), 2))
    for word, simword in all_pairs:
        if editdistance.eval(word, simword) > 2:
            randomword = sufficiently_different_random_word(synset)

            if random.random() <= 0.95:
              add_pair(train_pairs, word, simword, True, config)
              add_pair(train_pairs, word, randomword, False, config)
            else:
              add_pair(test_pairs, word, simword, True, config)
              add_pair(test_pairs, word, randomword, False, config)
  
  import pdb; pdb.set_trace()
  with open('train_word_pairs.tsv', 'w') as f:
    for pair in train_pairs.keys():
      f.write('\t'.join(str(x) for x in pair) + '\t')
      f.write('\t'.join(str(x) for x in train_pairs[pair]) + '\n')

  with open('test_word_pairs.tsv', 'w') as f:
    for pair in test_pairs.keys():
      f.write('\t'.join(str(x) for x in pair) + '\t')
      f.write('\t'.join(str(x) for x in test_pairs[pair]) + '\n')


