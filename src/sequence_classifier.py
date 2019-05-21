import glob
import nltk
import argparse
import configparser

import numpy as np

from tensorflow import keras
from pathlib import PurePath
from skopt import BayesSearchCV

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, fbeta_score

from nltk import word_tokenize, sent_tokenize, TweetTokenizer

from model_utils import build_model, get_cv_param_grid, print_metrics


def run_sequence_classifier(data_dir, model_type, config):
    """Run the top level classifier setup and execution code, producing test prediction metrics at the end

    Args:
        data_dir: (string) root directory where the training data lies (i.e., ../data/guttenberg)
        model_type: (string) the type of model to run, either fcn or rnn
    """
    # Initialization
    all_sentences = []
    all_labels = []
    nltk.download('punkt')
    nltk.download('stopwords')

    random_seed = config.getint('DEFAULT_MODEL_PARAMS', 'RANDOM_SEED')
    default_num_epochs = config.getint('DEFAULT_MODEL_PARAMS', 'NUM_EPOCHS')
    default_batch_size = config.getint('DEFAULT_MODEL_PARAMS', 'BATCH_SIZE')
    default_learning_rate = config.getfloat('DEFAULT_MODEL_PARAMS', 'LEARNING_RATE')
    default_dropout = config.getfloat('DEFAULT_MODEL_PARAMS', 'DROPOUT')
    cross_validate = config.get('DEFAULT_MODEL_PARAMS', 'CROSS_VALIDATE')

    layer_string = config.get('DEFAULT_MODEL_PARAMS', 'LAYER_SIZES')
    default_layer_sizes = tuple(map(int, layer_string.split(',')))

    # Leverate nltk's Twitter tokenizer since it seemed to work better than punkt here
    #  TODO: in my research, removing common "stopwords" should improve performance, but it's not implemented here yet
    tokenizer_words = TweetTokenizer(preserve_case=False)
    #tokenizer_words = nltk.data.load('tokenizers/punkt/english.pickle')
    stopwords = nltk.corpus.stopwords.words('english')

    # Recursively iterate through input folders, extracting text data and labels (taken as the author folder name)
    for filename in glob.iglob(data_dir + '**/**.txt', recursive=True):
        with open(filename, 'rb') as input_file:
            book_data = input_file.read().decode(encoding='utf-8')

            # Set delimiters for actual book text, not including headers and footers
            #  TODO: This needs cleaning and can be better optimized, given that it still pulls in some metadata
            begin = book_data.rfind("START OF TH")
            end = book_data.rfind("END OF TH")
            data = book_data[begin:end]

        sentences = [tokenizer_words.tokenize(t) for t in nltk.sent_tokenize(data)]
        all_sentences += sentences
        all_labels += len(sentences) * [PurePath(filename).parent.name]

    # Encode the sentences in terms via indexed integer embeddings
    #  NOTE: the classes are very evenly balanced, so we don't need to up/downsample data
    vocabulary = sorted(set([word for sublist in all_sentences for word in sublist]))
    index_encoder = {u: i for i, u in enumerate(vocabulary)}

    # Encode all sentences based on numerical word index created above
    encoded_sentences = []
    for sent in all_sentences:
        encoding = np.array([index_encoder[word.lower()] for word in sent])
        encoded_sentences.append(encoding)

    # Pad all sentence vectors to the length of the longest in the data set
    max_length = max([len(sent) for sent in encoded_sentences])
    encoded_sentences = keras.preprocessing.sequence.pad_sequences(encoded_sentences,
                                                                   value=0,
                                                                   padding='post',
                                                                   maxlen=max_length)
    # Also binarize the two class labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)

    # Split the train and test sets -- with shuffling! -- based off an 80/20 ratio split
    X_train, X_test, y_train, y_test = train_test_split(encoded_sentences,
                                                        encoded_labels,
                                                        test_size=.2,
                                                        shuffle=True,
                                                        stratify=encoded_labels,
                                                        random_state=random_seed)

    # Build the model with default params, passing the vocabulary size in to dictate the input layer size
    classifier = KerasClassifier(build_fn=build_model,
                                 vocab_size=len(vocabulary),
                                 model_type=model_type,
                                 epochs=default_num_epochs,
                                 batch_size=default_batch_size,
                                 learning_rate=default_learning_rate,
                                 dropout=default_dropout,
                                 random_seed=random_seed,
                                 layer_sizes=default_layer_sizes)

    # Fit the model to the training data, then predict on the held-out test set
    sklearn_pipeline = Pipeline(steps=[('classifier', classifier)])

    # If we want to cross validate to tune hyperparameters, use a Bayes search of the parameter space
    if cross_validate:
        classifier = BayesSearchCV(estimator=sklearn_pipeline,
                                   search_spaces=get_cv_param_grid(model_type),
                                   scoring=make_scorer(fbeta_score, beta=1),
                                   iid=False,
                                   cv=2,
                                   refit=True,
                                   n_jobs=1,
                                   n_iter=5,
                                   verbose=100,
                                   random_state=random_seed)

    # Fit the model to the training data, then evaluate model on test set
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    print_metrics(y_test, pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', help="Directory containing training data [default: /data/gutenberg/]",
                        default="../data/gutenberg/")
    parser.add_argument('--config_file', '-c', help="Configuration file with default model parameters [default: /default_parameters.ini]",
                        default="../default_parameters.ini")
    parser.add_argument('--model_type', '-m', help="Model type to run, either fcn or rnn [default: rnn]",
                        default="rnn")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)

    run_sequence_classifier(args.data_dir, args.model_type, config)
