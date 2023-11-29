import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag























# your work finish her 
# import nltk
# import numpy as np
# nltk.download('punkt') ##packege with pre_trained tokenizer to allowe  line 6 to work
# # nltk.download('all')
# from nltk.stem.porter import PorterStemmer # for the stemming
# stemmer=PorterStemmer()
# # nltk.download('punkt')  from nltk import word_tokenize.

# def tokenize(sentence):
#     return nltk.word_tokenize(sentence)

# def stem(word):
#     return stemmer.stem(word.lower())

# def bag_of_words(tokenize_sentence,all_words):
    
#     tokenize_sentence=[stem(w)for w in tokenize_sentence ]   #this is list comprehension tool 
#     bag=np.zeros(len(all_words),dtype=np.float32)             ##arry with same size only zeroez    ## creat bag and insilazet with zero like the beloow ex
#     for idx,w, in enumerate(all_words):    ## will give as the index and the current word
#       if w in tokenize_sentence:
#         bag[idx]=1.0
#     return bag

# your work finish her




# sentence = ["hello", "how", "are","you"]
# words =["hi","hello","i","you","bye","thank","cool"]
# bog=bag_of_words(sentence,words)
# print(bog)

   # sentence = ["hello", "how", "are","you"]
  #words =["hi","hello","i","you","bye","thank",cool]  look in the pattern 
  #bog=    [0  ,    1  , 0  , 1  ,  0  ,  0     , 0]
   
   
   
# #  bellowe this is the test  for the tokenizer founction 

# a="what is your name?"
# print(a)
# a_tokenizer=tokenize(a)
# print(a_tokenizer)


# # bellowe this is the test  for the Steaming founction 

# words=["Organize","Organization","organizing",]
# steam_words=[stem(w) for w in words]
# print(steam_words)




############# gpt help
# import nltk
# from nltk.stem.porter import PorterStemmer

# nltk.download('punkt')  # Download the punkt tokenizer

# # Initialize the Porter Stemmer
# stemmer = PorterStemmer()

# def tokenize(sentence):
#     return nltk.word_tokenize(sentence)

# def stem(word):
#     return stemmer.stem(word.lower())

# def bag_of_words(tokenized_sentence, all_words):
#     # Implement the logic for creating a bag of words representation
#     pass

# # Example usage
# sentence = "what is your name?"
# print(sentence)
# tokenized_sentence = tokenize(sentence)
# print("Tokenized:", tokenized_sentence)

# stemmed_words = [stem(word) for word in tokenized_sentence]
# print("Stemmed:", stemmed_words)

# # Example usage of bag_of_words (once implemented)
# all_words = ["what", "is", "your", "name", "example", "other", "words"]
# bag = bag_of_words(stemmed_words, all_words)
# print("Bag of Words:", bag)
