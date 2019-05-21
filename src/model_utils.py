import tensorflow as tf

from tensorflow.keras import layers, losses
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def get_cv_param_grid(model_type):
    """Gets the predefined cross-validation parameter search space for a given model type. I'm creating a nominal,
       non-exhaustive search space here since the time isn't there to test everything

    Args:
        model_type: (string) the model type being run, either fcn or rnn
    """
    if model_type == 'fcn':
        param_grid = {'classifier__epochs': [30, 40, 50],
                      'classifier__batch_size': [64, 256, 512],
                      'classifier__dropout': [0., .25, .5],
                      'classifier__learning_rate': [.01, .001, .0001],
                      'classifier__layer_sizes': [(16, 32), (32, 64, 128), (64, 128, 256, 128)]}
    elif model_type == 'rnn':
        param_grid = {'classifier__epochs': [10, 20, 30],
                      'classifier__batch_size': [64, 256, 512],
                      'classifier__dropout': [0., .25, .5],
                      'classifier__learning_rate': [.01, .001, .0001],
                      'classifier__layer_sizes': [(16, 32, 64, 32), (32, 64, 128, 256, 128), (64, 128, 256, 128, 64)]}

    return param_grid


def print_metrics(y_test, pred):
    """Prints the final performance metrics on the test set.

    Args:
        y_test: (list) the labeled test set ground truth data
        pred: (list) predictions yielded from the classifier model
    """
    confusion_mat = confusion_matrix(y_test, pred)
    print('--------------------------------------------------')
    print('Confusion Matrix:')
    print(confusion_mat)

    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print('--------------------------------------------------')
    print('Precision: {} *** Recall: {} *** Overall F1 Score: {}'.format(precision, recall, f1))
    print('--------------------------------------------------')


def build_model(model_type, vocab_size, layer_sizes, dropout, learning_rate, random_seed):
    """Builds the model based on the Tensorflow.Keras sequential implementation.

    Args:
        y_test: (list) the labeled test set ground truth data
        pred: (list) predictions yielded from the classifier model
    """
    model = tf.keras.Sequential()

    # The models are built here sequentially, depending on the desired type
    #  TODO: explore effects of batch normalization
    if model_type == 'fcn':
        model.add(layers.Embedding(vocab_size, layer_sizes[0]))
        model.add(layers.GlobalAveragePooling1D())
        for layer_size in layer_sizes:
            #layers.BatchNormalization(),
            model.add(layers.Dense(layer_size, activation=tf.nn.relu))
            model.add(layers.Dropout(dropout, seed=random_seed))
        model.add(layers.Dense(1, activation=tf.nn.sigmoid))

    elif model_type == 'rnn':
        model.add(layers.Embedding(vocab_size, layer_sizes[0]))
        for layer_size in layer_sizes[:-2]:
            model.add(layers.Bidirectional(tf.keras.layers.LSTM(layer_size, return_sequences=True)))
        model.add(layers.Bidirectional(tf.keras.layers.LSTM(layer_sizes[-1])))
        model.add(layers.Dense(layer_sizes[-2], activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(tf.optimizers.Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model