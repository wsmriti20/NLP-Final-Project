import nltk
import pandas as pd
import math
import sys
import json
import string
from nltk.corpus import wordnet, stopwords
# from nltk.corpus import stopwords
from pandas import DataFrame
import sklearn
from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


stop_words = set(stopwords.words('english'))
stop_words = []

class Object:
    def __init__(self, main_obj, type):
        self.main_obj = main_obj
        self.type = type
        self.descriptors = []

class Node:
    def __init__(self, id, parent, pos_tag, phrase):
        self.id = id
        self.parent = parent
        self.pos_tag = pos_tag
        self.phrase = phrase
        self.children = {}
        self.visited = False


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

        self.stored_data = []

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
                        parse_tree.node_locator[current_node].pos_tag = parse_tree.node_locator[current_node].pos_tag.split()[0]

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

    def modify_dependency_tree(self, tree, pos_list):
        #Extend Phrase into child node
        new_tree = tree
        main_node_list = list(new_tree.node_locator)
        main_node_list.sort(reverse=True)
        for node in main_node_list:
            if(not new_tree.node_locator[node].children):
                if (len(new_tree.node_locator[node].phrase.split()) == 2):
                    parent_id = int(new_tree.node_locator[node].id)

                    #increment nodes
                    node_list = list(new_tree.node_locator)
                    node_list.sort(reverse=True)
                    for node_index in node_list:
                        if (new_tree.node_locator[node_index].children):
                            child_list = list(new_tree.node_locator[node_index].children)
                            child_list.sort(reverse=True)
                            for child_node in child_list:
                                if (child_node > parent_id):
                                    new_tree.node_locator[node_index].children.pop(child_node)
                                    new_tree.node_locator[node_index].children[child_node + 1] = 1

                        if (node_index > parent_id):
                            new_tree.node_locator[node_index + 1] = Node(node_index + 1, new_tree.node_locator[node_index].parent, "", new_tree.node_locator[node_index].phrase)

                            for temp_child in new_tree.node_locator[node_index].children:
                                new_tree.node_locator[node_index + 1].children[temp_child] = 1

                            new_tree.node_locator.pop(node_index)

                    new_tree.node_locator[parent_id].children[parent_id + 1] = 1
                    new_tree.node_locator[parent_id + 1] = Node(str(parent_id + 1), parent_id, "", new_tree.node_locator[node].phrase.split()[1])
                    new_tree.node_locator[node].phrase = new_tree.node_locator[node].phrase.split()[0]

        #new_tree.print_tree()

        #Get POS tags from grammar tree
        i = 0
        j = 2
        while(i < len(pos_list)):
            if(new_tree.node_locator[j].phrase != "" and new_tree.node_locator[j].phrase != "."):
                #print(pos_list)
                #print("i: " + str(i) + " | j: " + str(j))
                new_tree.node_locator[j].pos_tag = pos_list[i]
                i += 1
            j += 1

        new_tree.print_tree()

        #Remove Determiners from Tree
        total_removed = 0
        new_list = list(new_tree.node_locator)
        new_list.sort()
        for node in new_list:
            if (node in new_tree.node_locator):
                if(new_tree.node_locator[node].pos_tag == "DT"):
                    print(node)
                    new_tree.node_locator.pop(node)
                    total_removed += 1

                    node_list = list(new_tree.node_locator)
                    node_list.sort()
                    for node_index2 in node_list:
                        if(new_tree.node_locator[node_index2].children):
                            for child_node in list(new_tree.node_locator[node_index2].children):
                                if(child_node > node):
                                    new_tree.node_locator[node_index2].children.pop(child_node)
                                    new_tree.node_locator[node_index2].children[child_node - 1] = 1

                        if(node_index2 > node):
                            new_tree.node_locator[node_index2 - 1] = new_tree.node_locator[node_index2]

                    new_tree.node_locator.pop(len(new_tree.node_locator))

        return new_tree

    def grab_objects(self, tree):
        objects = []
        main_tags = {"NN": 1, "NNS": 1, "NNP": 1, "NNPS": 1, "VB": 1, "VBD": 1, "VBG": 1, "VBN": 1, "VBP": 1, "VBZ": 1, "PRP" : 1, "PRP$": 1}
        descriptor_tags = {"JJ" : 1, "JJR": 1, "JJS": 1, "RB": 1, "RBR": 1, "RBS": 1}
        for node in tree.node_locator:
            if (not tree.node_locator[node].children):
                if(tree.node_locator[node].pos_tag in main_tags):
                    new_object = Object(tree.node_locator[node].phrase, tree.node_locator[node].pos_tag)

                    tree.node_locator[node].visited = True
                    for sibling in tree.node_locator[tree.node_locator[node].parent].children:
                        if(tree.node_locator[sibling].pos_tag in descriptor_tags and tree.node_locator[sibling].visited == False):
                            new_object.descriptors.append(tree.node_locator[sibling].phrase)
                            tree.node_locator[sibling].visited = True

                    objects.append(new_object)

        return objects

    def depth_first_search_pos(self, tree):
        pos_tags = []
        for node in tree.node_locator:
            if(not tree.node_locator[node].children and tree.node_locator[node].pos_tag not in string.punctuation):
                pos_tags.append(tree.node_locator[node].pos_tag)

        return pos_tags


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

    def find_antonyms(self, word):
        # converts the string of sentence to a dictionary of words as keys and antonyms as their values
        antonyms = []
        if word not in stop_words:
            for set in wordnet.synsets(word):
                for set_word in set.lemmas():
                    if set_word.antonyms():
                        antonyms.append(set_word.name())
        return antonyms

    def find_synonyms(self, word):
        synonymns = []
        if word not in stop_words:
            for set in wordnet.synsets(word):
                for set_word in set.lemmas():
                    synonymns.append(set_word.name())
        return synonymns

    def get_unigrams(self, sentence):
        return sentence.split(' ')

    def get_bigrams(self, sentence):
        return list(nltk.bigrams(sentence.split(' ')))

    def unigram_cross_count(self, unigrams1, unigrams2):
        count = 0
        for uni1 in unigrams1:
            for uni2 in unigrams2:
                if uni1 == uni2:
                    count += 1
        return count

    def bigram_cross_count(self, bigrams1, bigrams2):
        count = 0
        for bi1 in bigrams1:
            for bi2 in bigrams2:
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

    def calculate_similarity_code(self, line1, line2):
        # object_list_sent1 = self.grab_objects(line1)
        # object_list_sent2 = self.grab_objects(line2)
        noun_sent1_list = []
        noun_sent2_list = []
        verb_sent1_list = []
        verb_sent2_list = []
        
        #  "VB": 1, "VBD": 1, "VBG": 1, "VBN": 1, "VBP": 1, "VBZ": 1, 
        for obj in self.grab_objects(line1):
            if obj.type in ["NN", "NNS", "NNP", "NNPS", "PRP", "PRP$"]:
                # add into the noun obj list
                noun_sent1_list.append(obj)
            else:
                # for the verb obj list
                verb_sent1_list.append(obj)

        for obj in self.grab_objects(line2):
            if obj.type in ["NN", "NNS", "NNP", "NNPS", "PRP", "PRP$"]:
                # add into the noun obj list
                noun_sent1_list.append(obj)
            else:
                # for the verb obj list
                verb_sent2_list.append(obj)
        
        ## get rid of this please
        short_noun_list = noun_sent1_list if len(noun_sent1_list) < len(noun_sent2_list) else noun_sent2_list
        short_verb_list = verb_sent1_list if len(verb_sent1_list) < len(verb_sent2_list)  else verb_sent2_list  
        long_noun_list = noun_sent1_list if len(noun_sent1_list) > len(noun_sent2_list) else noun_sent2_list
        long_verb_list = verb_sent1_list if len(verb_sent1_list) > len(verb_sent2_list) else verb_sent2_list

        similarity_score = 0
        noun_score_list = []
        # optimize the loops for the love of GOD
        for noun1 in short_noun_list:
            ns1 = wordnet.synsets(noun1.main_obj)[0] if wordnet.synsets(noun1.main_obj) else None
            for noun2 in long_noun_list:
                ns2 = wordnet.synsets(noun2.main_obj)[0] if wordnet.synsets(noun2.main_obj) else None
                if ns2 and ns1:
                    x = ns1.wup_similarity(ns2) if ns1.wup_similarity(ns2) else 0
                    noun_score_list.append(ns1.wup_similarity(ns2))

                # adding similarity scores for the desciptors if any 
                descriptor_score_list = []
                if noun1.descriptors and noun2.descriptors:
                    for d1 in noun1.descriptors:
                        for d2 in noun2.descriptors:
                            ds1 = wordnet.synsets(d1)[0] if wordnet.synsets(d1) else None
                            ds2 = wordnet.synsets(d2)[0] if wordnet.synsets(d2) else None
                            if ds1 and ds2:
                                x = ds1.wup_similarity(ds2) if ds1.wup_similarity(ds2) else 0
                                descriptor_score_list.append(x)
                        similarity_score += max(descriptor_score_list)
            if noun_score_list:
                similarity_score += max(noun_score_list)

        
        # adding similarity scores for the verbs if any
        verbs_score_list = []
        for verb1 in short_verb_list:
            for verb2 in long_verb_list:
                vs1 = wordnet.synsets(verb1.main_obj)[0] if wordnet.synsets(verb1.main_obj) else None
                vs2 = wordnet.synsets(verb2.main_obj)[0] if wordnet.synsets(verb2.main_obj) else None
                if vs1 and vs2:
                    x = vs1.wup_similarity(vs2) if vs1.wup_similarity(vs2) else 0
                    verbs_score_list.append(x)
            if verbs_score_list:
                similarity_score += max(verbs_score_list)

        return similarity_score

    def calculate_antonymy_score(self, line1, line2):
        noun_sent1_list = []
        noun_sent2_list = []
        verb_sent1_list = []
        verb_sent2_list = []
        antonymy_score = 0

        #  "VB": 1, "VBD": 1, "VBG": 1, "VBN": 1, "VBP": 1, "VBZ": 1, 
        for obj in self.grab_objects(line1):
            if obj.type in ["NN", "NNS", "NNP", "NNPS", "PRP", "PRP$"]:
                # add into the noun obj list
                noun_sent1_list.append(obj)
            else:
                # for the verb obj list
                verb_sent1_list.append(obj)

        for obj in self.grab_objects(line2):
            if obj.type in ["NN", "NNS", "NNP", "NNPS", "PRP", "PRP$"]:
                # add into the noun obj list
                noun_sent1_list.append(obj)
            else:
                # for the verb obj list
                verb_sent2_list.append(obj) 

        short_noun_list = noun_sent1_list if len(noun_sent1_list) < len(noun_sent2_list) else noun_sent2_list
        short_verb_list = verb_sent1_list if len(verb_sent1_list) < len(verb_sent2_list)  else verb_sent2_list  
        long_noun_list = noun_sent1_list if len(noun_sent1_list) > len(noun_sent2_list) else noun_sent2_list
        long_verb_list = verb_sent1_list if len(verb_sent1_list) > len(verb_sent2_list) else verb_sent2_list

        for noun1 in long_noun_list:
            for noun2 in short_noun_list:
                antonyms_list = self.find_antonyms(noun2.main_obj)
                if antonyms_list:
                    if noun1.main_obj in antonyms_list:
                        antonymy_score += 1

                if noun1.descriptors and noun2.descriptors:
                    for d1 in noun1.descriptors:
                        for d2 in noun2.descriptors:
                            if d1.main_obj in self.find_antonyms(d2.main_obj):
                                antonymy_score += 1
        
        for verb1 in long_verb_list:
            for verb2 in verb_sent2_list:
                if verb1.main_obj in self.find_antonyms(verb2.main_obj):
                    antonymy_score += 1

        return antonymy_score

    def calculate_synonymity_score(self, line1, line2):
        noun_sent1_list = []
        noun_sent2_list = []
        verb_sent1_list = []
        verb_sent2_list = []
        synonymity_score = 0

        #  "VB": 1, "VBD": 1, "VBG": 1, "VBN": 1, "VBP": 1, "VBZ": 1, 
        for obj in self.grab_objects(line1):
            if obj.type in ["NN", "NNS", "NNP", "NNPS", "PRP", "PRP$"]:
                # add into the noun obj list
                noun_sent1_list.append(obj)
            else:
                # for the verb obj list
                verb_sent1_list.append(obj)

        for obj in self.grab_objects(line2):
            if obj.type in ["NN", "NNS", "NNP", "NNPS", "PRP", "PRP$"]:
                # add into the noun obj list
                noun_sent1_list.append(obj)
            else:
                # for the verb obj list
                verb_sent2_list.append(obj) 

        short_noun_list = noun_sent1_list if len(noun_sent1_list) < len(noun_sent2_list) else noun_sent2_list
        short_verb_list = verb_sent1_list if len(verb_sent1_list) < len(verb_sent2_list)  else verb_sent2_list  
        long_noun_list = noun_sent1_list if len(noun_sent1_list) > len(noun_sent2_list) else noun_sent2_list
        long_verb_list = verb_sent1_list if len(verb_sent1_list) > len(verb_sent2_list) else verb_sent2_list

        for noun1 in long_noun_list:
            for noun2 in short_noun_list:
                if noun1.main_obj in self.find_synonyms(noun2.main_obj):
                    synonymity_score += 1

                if noun1.descriptors and noun2.descriptors:
                    for d1 in noun1.descriptors:
                        for d2 in noun2.descriptors:
                            if d1.main_obj in self.find_synonyms(d2.main_obj):
                                synonymity_score += 1
        
        for verb1 in long_verb_list:
            for verb2 in verb_sent2_list:
                if verb1.main_obj in self.find_synonyms(verb2.main_obj):
                    synonymity_score += 1

        return synonymity_score

    def read_data(self, index):
        bad_nodes = 0
        with open(self.data_set[index], 'r') as data_file:                      # Read Training Data
            i = 0
            actual_features_list_of_lists = []
            for line in data_file:
                row_line = []
                if (i < 100000):                                               # Line Limiter
                    feature_vector = [None] * 19                                  # Create Feature Vector
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
                    

                    # feature_vector[9] = self.calculate_similarity_code(feature_vector[5], feature_vector[6])                   
                    # feature_vector[10] = self.calculate_antonymy_score(feature_vector[5], feature_vector[6])
                    # feature_vector[11] = self.calculate_synonymity_score(feature_vector[5], feature_vector[6])
                    
                    row_line.append(data_line["gold_label"])  # Extract Gold Label
                    row_line.append(len(data_line["sentence1"].split()))
                    row_line.append(len(data_line["sentence2"].split()))
                    row_line.append(self.calculate_similarity_code(feature_vector[5], feature_vector[6]))                
                    row_line.append(self.calculate_antonymy_score(feature_vector[5], feature_vector[6]))
                    row_line.append(self.calculate_synonymity_score(feature_vector[5], feature_vector[6]))
                    
                    # print(feature_vector[10],feature_vector[11], feature_vector[9])

                    # sentence 1 unigrams and bigrams
                    feature_vector[13] = self.get_unigrams(feature_vector[1])
                    feature_vector[14] = self.get_bigrams(feature_vector[1])

                        # sentence 2 unigrams and bigrams
                    feature_vector[15] = self.get_unigrams(feature_vector[2])
                    feature_vector[16] = self.get_bigrams(feature_vector[2])

                        # unigram cross count, bigram cross count, and acsii sum difference
                    row_line.append(self.unigram_cross_count(feature_vector[13], feature_vector[15]))
                    row_line.append(self.bigram_cross_count(feature_vector[14], feature_vector[16]))
                    row_line.append(self.ascii_diff(feature_vector[1], feature_vector[2]))
                    row_line.append(abs(feature_vector[7]-feature_vector[8]))
                        # print(row_line)
                    
                    actual_features_list_of_lists.append(row_line)
                    
                    if(i == 100000):
                        # print("bad_nodes: " + str(bad_nodes))
                        actual_features_list_of_lists.append(row_line)
                        return 0

                i += 1  # Line Limiter Increment
        return actual_features_list_of_lists

# entailment_system_instance = Entailment_System(sys.argv[1], sys.argv[2], sys.argv[3])
entailment_system_instance = Entailment_System("snli_1.0_train.jsonl", "snli_1.0_dev.jsonl", "snli_1.0_test.jsonl")
dft = entailment_system_instance.read_data(0)
x_train =  DataFrame(dft, \
    columns=["Labels", "Len of Sent1", "Len of Sent2", \
        "Similarity Score", \
        "Antonym Score", \
        "Synonym Score", \
        "Unigram Cross Cnt", "Bigram Cross Cnt", "Ascii Diff", "Len Diff"])


y_train = x_train.Labels
x_train = x_train.drop(["Labels"], axis = 1)

# print(y_train)
x_test =  DataFrame(entailment_system_instance.read_data(2), \
    columns=["Labels", "Len of Sent1", "Len of Sent2", \
            "Similarity Score", \
            "Antonym Score", \
            "Synonym Score", \
            "Unigram Cross Cnt", "Bigram Cross Cnt", "Ascii Diff", "Len Diff"])
y_test = x_test.Labels
x_test = x_test.drop(["Labels"], axis = 1)

# print("\n\n", y_test)

Clf = DecisionTreeClassifier(criterion = "entropy")
Clf = Clf.fit(x_train, y_train)
y_pred = Clf.predict(x_test)

print(accuracy_score(y_test, y_pred)*100)