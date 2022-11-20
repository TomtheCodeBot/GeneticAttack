import numpy as np
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
import pickle
from goog_lm import LM
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from module.preprocessing import load_train_test_imdb_data
from module.preprocessing import clean_text
from model.RobustNaiveBayes import RobustNaiveBayesClassifierPercentage
from model.Wrapper import WrapperModel
from module.methods import GeneticAtack
from module.attack import attackModel
import configparser
import pandas as pd

np.random.seed(1001)
tf.random.set_seed(1001)
config = configparser.ConfigParser()
config.read('config.ini')
VOCAB_SIZE  = int( config["GENERAL"]["VOCAB_SIZE"])

if __name__=="__main__":
    with open('aux_files/dataset_%d.pkl' %VOCAB_SIZE, 'rb') as f:
        dataset = pickle.load(f)
    ### Loading IMDB data for training.
    IMDB_DATASET = 'aclImdb/'
    train_data, test_data = load_train_test_imdb_data(IMDB_DATASET)

    ### Creating vectorizer.
    vectorizer = CountVectorizer(stop_words="english",
                                preprocessor=clean_text, 
                                min_df=0,max_features=50000)

    training_features = vectorizer.fit_transform(train_data["text"])
    training_labels = train_data["sentiment"]
    test_features = vectorizer.transform(test_data["text"])
    test_labels = test_data["sentiment"]

    ### Traing Robust Naive Bayes model.
    RNB = RobustNaiveBayesClassifierPercentage(100)
    RNB.fit(training_features, training_labels)
    y_pred_RNB = RNB.predict(test_features)
    acc = accuracy_score(test_labels, y_pred_RNB)
    print("RNB ccuracy on the IMDB dataset: {:.2f}".format(acc*100))

    ### Setup GloVe and GoogleLM.
    goog_lm = LM()

    with open('aux_files/dataset_%d.pkl' %VOCAB_SIZE, 'rb') as f:
        dataset = pickle.load(f)
        
    doc_len = [len(dataset.test_seqs2[i]) for i in 
           range(len(dataset.test_seqs2))]
    dist_mat = np.load('aux_files/dist_counter_%d.npy' %VOCAB_SIZE)
    # Prevent returning 0 as most similar word because it is not part of the dictionary
    dist_mat[0,:] = 100000
    dist_mat[:,0] = 100000

    skip_list = np.load('aux_files/missed_embeddings_counter_%d.npy' %VOCAB_SIZE)
    max_len = 250
    test_x = pad_sequences(dataset.test_seqs2, maxlen=max_len, padding='post')
    test_y = np.array(dataset.test_y)
    

    ### Change config for script that compatible with TF.1.
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    if tf.compat.v1.get_default_session():
        sess.close()
        
    ### Start Attacking RNB.
    pop_size = 60
    n1 = 8
    model = WrapperModel(dataset,RNB,vectorizer)
    ga_atttack = GeneticAtack(sess, model, model, model, dataset, dist_mat, 
                                    skip_list,
                                    goog_lm, max_iters=30, 
                                    pop_size=pop_size,
                                    n1 = n1,
                                    n2 = 4,
                                    use_lm = True, use_suffix=False)
    sample_size=int( config["GENERAL"]["SAMPLE_SIZE"])
    test_size=int( config["GENERAL"]["TEST_SIZE"])
    orig_list,adv_list,dist_list = attackModel(dataset,test_x,test_y,model,ga_atttack)
    df = pd.DataFrame(list(zip(orig_list,adv_list,dist_list)),
               columns =['Original_text', 'Adverserial_text','Percent_changed'])