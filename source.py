import nltk
import pandas as pd
import math
import sys
import json
import string

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
        self.node_locator = {1 : self.root}

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
        if(mode == 0):
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

class Entailment_System: # Index: 0 for training, 1 for development, 2 for testing
    def __init__(self, training_set, development_set, test_set):
        self.data_set = [training_set, development_set, test_set]
        self.data_frame = [pd.DataFrame(index = range(550115),
        columns = ["gold_label", "premise", "hypothesis", "premise_binary_parse", "hypothesis_binary_parse", "premise_parse", "hypothesis_parse", "premise_length", "hypothesis_length"])] * 3

    def extract_parse_tree(self, sentence, mode):
        tree_level = 0
        parse_tree = None
        current_node = 1
        prev_char = ""
        while(True):
            pos_extracted = False
            for char in sentence:
                if(char == "("):
                    if (parse_tree == None):
                        parse_tree = Tree()

                    else:
                        current_node = parse_tree.create_child(current_node)
                        pos_extracted == False

                    tree_level += 1
                    pos_extracted = False

                elif(char == ")"):
                    #Phrase Cleaning Routine
                    clean_phrase = ""
                    i = 0
                    while (i < len(parse_tree.node_locator[current_node].phrase.split())):
                        clean_phrase += parse_tree.node_locator[current_node].phrase.split()[i]
                        if (i < len(parse_tree.node_locator[current_node].phrase.split()) - 1):
                            clean_phrase += " "
                        i += 1

                    parse_tree.node_locator[current_node].phrase = clean_phrase

                    if(mode == 1): #POS Tag Cleaning Routine
                        parse_tree.node_locator[current_node].pos_tag = parse_tree.node_locator[current_node].pos_tag.split()[0]

                    tree_level -= 1
                    current_node = parse_tree.get_parent(current_node)

                elif (parse_tree != None):
                    if(char.isupper() == False and prev_char.isupper() == True):
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

    def read_data(self, index):
        with open(self.data_set[index], 'r') as data_file:                      # Read Training Data
            i = 0
            for line in data_file:
                if (i < 1000000):                                               # Line Limiter
                    feature_vector = [None] * 9                                 # Create Feature Vector
                    data_line = json.loads(line)
                    feature_vector[0] = data_line["gold_label"]                 # Extract Gold Label
                    feature_vector[1] = data_line["sentence1"]                  # Extract Premise Sentence
                    feature_vector[2] = data_line["sentence2"]                  # Extract Hypothesis Sentence
                    feature_vector[3] = self.extract_parse_tree(data_line["sentence1_binary_parse"], 0)
                    feature_vector[4] = self.extract_parse_tree(data_line["sentence2_binary_parse"], 0)
                    feature_vector[5] = self.extract_parse_tree(data_line["sentence1_parse"], 1)
                    feature_vector[6] = self.extract_parse_tree(data_line["sentence2_parse"], 1)
                    feature_vector[7] = len(data_line["sentence1"].split())
                    feature_vector[8] = len(data_line["sentence2"].split())

                    #if (feature_vector[3] != -1 or feature_vector[4] != -1 or feature_vector[5] != -1 or feature_vector[6] != -1):
                    #    self.data_frame[index].append(pd.Series(feature_vector), ignore_index=True)

                    if (feature_vector[3] != -1 or feature_vector[4] != -1 or feature_vector[5] != -1 or feature_vector[6] != -1):
                        self.data_frame[index].loc[i, self.data_frame[index].columns] = feature_vector

                    #if(i == 90):
                    #    print(feature_vector[1])
                    #    print(self.data_frame[index].loc[90, self.data_frame[index].columns])
                    #    break

                i += 1                                                          # Line Limiter Increment
                print(i)                                                        # Progress Indicator

entailment_system_instance = Entailment_System(sys.argv[1], sys.argv[2], sys.argv[3])
entailment_system_instance.read_data(0)
