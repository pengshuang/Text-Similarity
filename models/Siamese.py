import keras
from keras.layers import *
from keras import Model


def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))


class Siamese():
    def model(self, embeddings_matrix, maxlen, word_index):
        left_input = Input(shape=(maxlen,), dtype='int32')
        right_input = Input(shape=(maxlen,), dtype='int32')
        embedding_layer = Embedding(len(word_index) + 1,
                                    embeddings_matrix.shape[1],
                                    weights=[embeddings_matrix],
                                    input_length=maxlen,
                                    trainable=False)
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)

        # two lstm layer share the parameters
        shared_lstm = LSTM(128)
        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)
        malstm_distance = Add(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                                output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
        model = Model([left_input, right_input], [malstm_distance])
        model.compile(
            loss='categorical_crossentropy',
            optimizer=adam)
        return model

