import nltk
import pandas as pd
import math
import sys
import json
import string
from nltk.corpus import wordnet, stopwords
# from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


class Node:
    def __init__(self, id, parent, pos_tag, phrase):
        self.id = "-1"
        self.parent = parent
        self.pos_tag = pos_tag
        self.phrase = ""
        self.children = {}


class Tree:
    def __init__(self):
        self.root = Node(1, -1, "", "ROOT")
        self.node_locator = {1: self.root}

    def find_node(self, id):
        if (id in self.node_locator):
            return id

        else:
            return -1

    def create_child(self, parent):
        self.node_locator[parent].children[len(self.node_locator) + 1] = 1
        self.node_locator[len(self.node_locator) + 1] = Node(len(self.node_locator) + 1, parent, "", "")

        return len(self.node_locator)

    def append_data(self, node, char, mode):
        if (mode == 0):
            self.node_locator[node].phrase += char

        else:
            self.node_locator[node].pos_tag += char

    def get_parent(self, node):
        return self.node_locator[node].parent

    def print_tree(self):
        i = 1
        print("Children:")
        while i < len(self.node_locator) + 1:
            print(str(i) + ": " + str(self.node_locator[i].children))
            i += 1

        print("\n")

        i = 1
        print("\nWords Attached:")
        while i < len(self.node_locator) + 1:
            print(str(i) + ": " + str(self.node_locator[i].phrase))
            i += 1

        print("\n")

        i = 1
        print("\nTags:")
        while i < len(self.node_locator) + 1:
            print(str(i) + ": " + str(self.node_locator[i].pos_tag))
            i += 1

        print("\n")


class Entailment_System:  # Index: 0 for training, 1 for development, 2 for testing
    def __init__(self, training_set, development_set, test_set):
        self.data_set = [training_set, development_set, test_set]
        self.data_frame = [pd.DataFrame(index=range(550115),
                                        columns=["gold_label", "premise", "hypothesis", "premise_binary_parse",
                                                 "hypothesis_binary_parse", "premise_parse", "hypothesis_parse",
                                                 "premise_length", "hypothesis_length", "synonyms_sent1",
                                                 "antonyms_sent1", "synonyms_sent2", "antonyms_sent2"])] * 3

    def extract_parse_tree(self, sentence, mode):
        tree_level = 0
        parse_tree = None
        current_node = 1
        prev_char = ""
        while (True):
            pos_extracted = False
            for char in sentence:
                if (char == "("):
                    if (parse_tree == None):
                        parse_tree = Tree()

                    else:
                        current_node = parse_tree.create_child(current_node)
                        pos_extracted == False

                    tree_level += 1
                    pos_extracted = False

                elif (char == ")"):
                    # Phrase Cleaning Routine
                    clean_phrase = ""
                    i = 0
                    while (i < len(parse_tree.node_locator[current_node].phrase.split())):
                        clean_phrase += parse_tree.node_locator[current_node].phrase.split()[i]
                        if (i < len(parse_tree.node_locator[current_node].phrase.split()) - 1):
                            clean_phrase += " "
                        i += 1

                    parse_tree.node_locator[current_node].phrase = clean_phrase

                    if (mode == 1):  # POS Tag Cleaning Routine
                        parse_tree.node_locator[current_node].pos_tag = \
                        parse_tree.node_locator[current_node].pos_tag.split()[0]

                    tree_level -= 1
                    current_node = parse_tree.get_parent(current_node)

                elif (parse_tree != None):
                    if (char.isupper() == False and prev_char.isupper() == True):
                        pos_extracted = True

                    if ((mode == 1 and pos_extracted == False) or char == "$"):
                        parse_tree.append_data(current_node, char, 1)
                        if (char in string.punctuation):
                            pos_extracted = True

                    else:
                        parse_tree.append_data(current_node, char, 0) #Ignore if period?

                if (char != " "):
                    prev_char = char

                if (parse_tree == None): #DEBUG
                    return -1

            if (tree_level == 0):
                break

        return parse_tree

    def finding_pos(self, lemmaWord):
        split_word = str(lemmaWord).split(".")
        if split_word[1] == ('j'):
            return 'ADJ'
        elif split_word[1] == ('v'):
            return 'VERB'
        elif split_word[1] == ('n'):
            return 'NOUN'
        elif split_word[1] == ('r'):
            return 'ADV'
        elif split_word[1] == ('s'):
            return 'ADJ_SAT'
        else:
            return ''

    def find_antonyms(self, line):
        # converts the string of sentence to a dictionary of words as keys and antonyms as their values
        word_list = line.split(" ")
        antonyms_dictionary = {}
        for word in word_list:
            antonyms = []
            if word not in stop_words:
                for set in wordnet.synsets(word):
                    for set_word in set.lemmas():
                        if set_word.antonyms():
                            pos_tag = self.finding_pos(set_word)
                            tuple_value = (pos_tag, set_word.name())
                            antonyms.append(tuple_value)
                antonyms_dictionary[word] = antonyms
        return antonyms_dictionary

    def find_synonyms(self, line):
        # create a dictionary with the word as a key and the value a tuple of (
        word_list = line.split(" ")
        synonymns_dictionary = {}
        for word in word_list:
            # stop_words = set(stopwords.words('english'))
            if word not in stop_words:
                synonymns = []
                for set in wordnet.synsets(word):
                    for set_word in set.lemmas():
                        pos_tag = self.finding_pos(set_word)
                        tuple_value = (pos_tag, set_word.name())
                        synonymns.append(tuple_value)
                synonymns_dictionary[word] = synonymns
        return synonymns_dictionary

    def get_unigrams(self, sentence):
        return setence.split(' ')

    def get_bigrams(self, sentence):
        return list(nltk.bigrams(sentence.split(' ')))

    def unigram_cross_count(self, unigrams1, unigrams2):
        count = 0
        for uni1 in unigram1:
            for uni2 in unigram2:
                if uni1 == uni2:
                    count += 1
        return count

    def bigram_cross_count(self, bigrams1, bigrams2):
        count = 0
        for bi1 in bigram1:
            for bi2 in bigram2:
                if bi1[0] == bi2[0] and bi1[1] == bi2[1]:
                    count += 1
        return count

    def ascii_diff(self,sentence1, sentence2):
        diff = 0
        words = sentence1.split(' ')
        for word in words:
            for c in word:
                diff += ord(c)
        words = sentence2.split(' ')
        for word in words:
            for c in word:
                diff -= ord(c)
        return diff

    def read_data(self, index):
        with open(self.data_set[index], 'r') as data_file:                      # Read Training Data
            i = 0
            for line in data_file:
                if (i < 1000000):                                               # Line Limiter
                    feature_vector = [None] * 20                                  # Create Feature Vector
                    data_line = json.loads(line)
                    feature_vector[0] = data_line["gold_label"]  # Extract Gold Label
                    feature_vector[1] = data_line["sentence1"]  # Extract Premise Sentence
                    feature_vector[2] = data_line["sentence2"]  # Extract Hypothesis Sentence
                    feature_vector[3] = self.extract_parse_tree(data_line["sentence1_binary_parse"], 0)
                    feature_vector[4] = self.extract_parse_tree(data_line["sentence2_binary_parse"], 0)
                    feature_vector[5] = self.extract_parse_tree(data_line["sentence1_parse"], 1)
                    feature_vector[6] = self.extract_parse_tree(data_line["sentence2_parse"], 1)
                    feature_vector[7] = len(data_line["sentence1"].split())
                    feature_vector[8] = len(data_line["sentence2"].split())
                    # synonyms and antonyms of the sentence 1
                    feature_vector[9] = self.find_synonyms(feature_vector[1])
                    feature_vector[10] = self.find_antonyms(feature_vector[1])

                    # synonyms and antonyms for the second sentence

                    feature_vector[11] = self.find_synonyms(feature_vector[2])
                    feature_vector[12] = self.find_antonyms(feature_vector[2])

                    # sentence 1 unigrams and bigrams
                    feature_vector[13] = self.get_unigrams(feature_vector[1])
                    feature_vector[14] = self.get_bigrams(feature_vector[1])

                    # sentence 2 unigrams and bigrams
                    feature_vector[15] = self.get_unigrams(feature_vector[2])
                    feature_vector[16] = self.get_bigrams(feature_vector[2])

                    # unigram cross count, bigram cross count, and acsii sum difference
                    feature_vector[17] = self.unigram_cross_count(feature_vector[13], feature_vector[15])
                    feature_vector[18] = self.bigram_cross_count(feature_vector[14], feature_vector[16])
                    feature_vector[19] = self.ascii_diff(feature_vector[1], feature_vector[2])

                    # if (feature_vector[3] != -1 or feature_vector[4] != -1 or feature_vector[5] != -1 or feature_vector[6] != -1):
                    #    self.data_frame[index].append(pd.Series(feature_vector), ignore_index=True)

                    if (feature_vector[3] != -1 or feature_vector[4] != -1 or feature_vector[5] != -1 or feature_vector[
                        6] != -1):
                        self.data_frame[index].loc[i, self.data_frame[index].columns] = feature_vector

                    # if(i == 90):
                    #    print(feature_vector[1])
                    #    print(self.data_frame[index].loc[90, self.data_frame[index].columns])
                    #    break

                i += 1  # Line Limiter Increment
                print(i)  # Progress Indicator


entailment_system_instance = Entailment_System(sys.argv[1], sys.argv[2], sys.argv[3])
entailment_system_instance.read_data(0)
