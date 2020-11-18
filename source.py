import nltk
import pandas as pd
import math
import sys
import json

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

    def append_phrase(self, node, char):
            self.node_locator[node].phrase += char

    def get_parent(self, node):
        return self.node_locator[node].parent

    def print_tree(self):
        i = 1
        print("Children:\n")
        while i < len(self.node_locator) + 1:
            print(str(i) + ": " + str(self.node_locator[i].children))
            i += 1

        i = 1
        print("Words Attached:")
        while i < len(self.node_locator) + 1:
            print(str(i) + ": " + str(self.node_locator[i].phrase))
            i += 1

class Entailment_System:
    def __init__(self, training_set, development_set, test_set):
        self.training_set = training_set
        self.development_set = development_set
        self.test_set = test_set
        self.data_frame = [pd.DataFrame()] * 3                                  # 0 for training, 1 for development, 2 for testing

    def extract_parse_tree(self, sentence):
        tree_level = 0
        parse_tree = None
        current_node = 1
        while(True):
            for char in sentence:
                if(char == "("):
                    if (parse_tree == None):
                        parse_tree = Tree()

                    else:
                        current_node = parse_tree.create_child(current_node)

                    tree_level += 1

                elif(char == ")"):
                    #String Cleaning Routine
                    clean_string = ""
                    i = 0
                    while (i < len(parse_tree.node_locator[current_node].phrase.split())):
                        clean_string += parse_tree.node_locator[current_node].phrase.split()[i]
                        if (i < len(parse_tree.node_locator[current_node].phrase.split()) - 1):
                            clean_string += " "
                        i += 1

                    parse_tree.node_locator[current_node].phrase = clean_string

                    tree_level -= 1
                    current_node = parse_tree.get_parent(current_node)

                else:
                    parse_tree.append_phrase(current_node, char) #Ignore if period?


            if (tree_level == 0):
                break

        return parse_tree

    def read_data(self):
        with open(self.training_set, 'r') as train_file:                        # Read Training Data
            i = 0
            for line in train_file:
                if (i < 2):                                                     # Line Limiter
                    feature_vector = [None] * 9                                 # Create Feature Vector
                    data_line = json.loads(line)
                    feature_vector[0] = data_line["gold_label"]                 # Extract Gold Label
                    feature_vector[1] = data_line["sentence1"]                  # Extract Premise Sentence
                    feature_vector[2] = data_line["sentence2"]                  # Extract Hypothesis Sentence
                    feature_vector[3] = self.extract_parse_tree(data_line["sentence1_binary_parse"])
                    feature_vector[4] = self.extract_parse_tree(data_line["sentence2_binary_parse"])
                    #feature_vector[5] = self.extract_parse_tree(data_line)["sentence2_parse"]
                    #feature_vector[6] = self.extract_parse_tree(data_line)["sentence2_parse"]
                    feature_vector[7] = len(data_line["sentence1"].split())
                    feature_vector[8] = len(data_line["sentence2"].split())

                    feature_vector[3].print_tree()
                    feature_vector[4].print_tree()

                    i += 1                                                      # Line Limiter Increment

entailment_system_instance = Entailment_System(sys.argv[1], sys.argv[2], sys.argv[3])
entailment_system_instance.read_data()
