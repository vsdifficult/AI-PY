from collections import defaultdict
import re, math

def tokenze(message): 
    message = message.lower()  
    all_words = re.findall("[a-z0-9]+", message) 
    return set(all_words ) 

def count_words(training_set): 

    counts = defaultdict(lambda: [0, 0]) 

    for message, is_spam in training_set: 
        for word in tokenze(message=message): 
            counts[word][0 if is_spam else 1] += 1 
    return counts 

# learning\istockphoto-1052448556-612x612.jpg
def word_probabilities(counts, total_spams, total_non_spams, k=0.5): 
    """Преобразовать частности words_counts в список триплетов слово w, p(w | spam) и p(w | ~spam)""" 
    return [ 
        (w, (spam + k) / (total_spams + 2 * k), (non_spam + k) / (total_non_spams + 2*k)) 
        for w, (spam, non_spam) in counts.iteritems()
    ] 

def probability(word_probs, message): 
    message_words = tokenze(message=message) 
    log_prob_if_spam = log_prob_if_spam = 0.0 

    for word, prob_if_spam, prob_if_not_spam in word_probs: 

        if word in message_words: 
            log_prob_if_spam += math.log(prob_if_spam) 
            log_prob_if_spam += math.log(prob_if_not_spam) 

        else: 
            log_prob_if_spam += math.log(1.0 - prob_if_spam) 
            log_prob_if_spam += math.log(1.0 - prob_if_not_spam) 
             
    prob_if_spam = math.exp(log_prob_if_spam) 
    prob_if_not_spam = math.exp(log_prob_if_spam) 

    return prob_if_spam / (prob_if_spam + prob_if_not_spam) 

class NaiveBaeysClassifier: 

    def __init__(self, k=0.5) -> None:
        self.k = k 
        self.word_pobs = [] 

    def train(self, training_set): 

        num_spams = len([
            is_spam for messages, is_spam in training_set if is_spam
        ]) 

        num_non_spam = len(training_set) - num_spams 

        word_counts = count_words(training_set=training_set) 
        self.word_pobs = word_probabilities(word_counts, 
                                            num_spams, 
                                            num_non_spam, self.k) 
        
    def classify(self, messgae): 
        return probability(self.word_pobs, message=messgae) 
    

