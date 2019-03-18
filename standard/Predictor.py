# CryptoCurrency Price Predictor base on LSTM-GRU Neural Network
# Author: @VT-tech 

# Copyright 2018 The VT tech co. All Rights Reserved.
#
# Licensed under the Apache License, Version 1.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/vt-technologies-us/CryptoBeet/blob/master/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Bring in all of the public InBeet interface into this
# module.

import numpy as np

from keras import activations
from keras.callbacks import History
from keras.layers import BatchNormalization
from keras.layers import advanced_activations
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import GRU, LSTM
from keras.models import Sequential, load_model
from tensorflow.python.client import device_lib
from standard import config

class Predictor:
    LOG = True

    def __init__(self, mode='ActGRU', epochs=10, input_shape=(None, config.feature_size), output_shape=2):
        self.mode = mode
        self.model = None
        self._history = None
        self._epochs = epochs
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.make_model()

    def make_model(self):
        self.model = Sequential()

        # keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True,
        # kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros',
        # unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
        # activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
        # bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1,
        # return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)

        # keras.layers.GRU(units, activation='tanh',
        # recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform',
        # recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None,
        # recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        # recurrent_constraint=None, bias_constraint=None, dropout=0.0,
        # recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False,
        # stateful=False, unroll=False, reset_after=False)

        # Initializer:
        # Zeros, Ones, Constant, RandomNormal, RandomUniform, TruncatedNormal, VarianceScaling, Orthogonal,
        # Identity: lecun_uniform, glorot_normal, glorot_uniform, he_normal, lecun_normal, he_uniform,

        # -------------------------------------------- GRU Architectures -----------------------------------------------
        if self.mode == 'GRU':

            self.model.add(GRU(100, input_shape=self.input_shape, return_sequences=True, ))
            self.model.add(Dropout(0.35))
            self.model.add(GRU(200, return_sequences=False, use_bias=True, ))
            self.model.add(Dropout(0.35))

        elif self.mode == 'DashGRU':

            self.model.add(GRU(100, input_shape=self.input_shape, recurrent_activation='elu', return_sequences=True, ))
            self.model.add(Dropout(0.35))
            self.model.add(Activation('elu'))
            self.model.add(GRU(150, recurrent_activation='elu', return_sequences=False, ))
            self.model.add(Dropout(0.35))
            self.model.add(Activation('elu'))

        elif self.mode == 'TWSTDGRU':

            self.model.add(GRU(100, input_shape=self.input_shape, recurrent_activation='relu', return_sequences=True, ))
            self.model.add(Dropout(0.35))
            self.model.add(advanced_activations.LeakyReLU(alpha=0.3))
            self.model.add(GRU(150, recurrent_activation='relu', return_sequences=False, ))
            self.model.add(Dropout(0.35))
            self.model.add(advanced_activations.LeakyReLU(alpha=0.3))

        elif self.mode == 'TWSTD2GRU':

            self.model.add(GRU(100, input_shape=self.input_shape, recurrent_activation='relu', return_sequences=True, ))
            self.model.add(Dropout(0.35))
            self.model.add(Activation('elu'))
            self.model.add(GRU(150, recurrent_activation='relu', return_sequences=False, ))
            self.model.add(Dropout(0.35))
            self.model.add(Activation('elu'))

        elif self.mode == 'ActGRU':

            self.model.add(GRU(input_shape=self.input_shape, units=100, return_sequences=True))
            self.model.add(Dropout(0.35))
            self.model.add(Activation('elu'))
            self.model.add(GRU(150, return_sequences=False))
            self.model.add(Dropout(0.35))
            self.model.add(Activation('elu'))

        elif self.mode == 'CoreGRU':

            self.model.add(GRU(100, input_shape=self.input_shape, return_sequences=True, ))
            self.model.add(Dropout(0.35))
            self.model.add(advanced_activations.LeakyReLU(alpha=0.3))
            self.model.add(GRU(200, return_sequences=False, ))
            self.model.add(Dropout(0.35))
            self.model.add(advanced_activations.LeakyReLU(alpha=0.3))

        # ------------------------------------------ LSTM Architectures ------------------------------------------------

        elif self.mode == 'LSTM':

            self.model.add(LSTM(100, input_shape=self.input_shape, return_sequences=True, ))
            self.model.add(Dropout(0.35))
            self.model.add(LSTM(200, return_sequences=False, ))
            self.model.add(Dropout(0.35))

        elif self.mode == 'DashLSTM':

            self.model.add(LSTM(100, input_shape=self.input_shape, recurrent_activation='elu', return_sequences=True))
            self.model.add(Dropout(0.35))
            self.model.add(Activation('relu'))
            self.model.add(LSTM(150, recurrent_activation='elu', return_sequences=False))
            self.model.add(Dropout(0.35))
            self.model.add(Activation('relu'))

        elif self.mode == 'TWSTDLSTM':

            self.model.add(
                LSTM(100, input_shape=self.input_shape, recurrent_activation='relu', return_sequences=True, ))
            self.model.add(Dropout(0.35))
            self.model.add(Activation('elu'))
            self.model.add(LSTM(150, recurrent_activation='relu', use_bias=True, return_sequences=False, ))
            self.model.add(Dropout(0.35))
            self.model.add(Activation('elu'))

        elif self.mode == 'ActLSTM':

            self.model.add(LSTM(input_shape=self.input_shape, units=100, return_sequences=True))
            self.model.add(Dropout(0.35))
            self.model.add(Activation('relu'))
            self.model.add(LSTM(150, return_sequences=False))
            self.model.add(Dropout(0.35))
            self.model.add(Activation('relu'))

        elif self.mode == 'CoreLSTM':

            self.model.add(LSTM(100, input_shape=self.input_shape, return_sequences=True, ))
            self.model.add(Dropout(0.35))
            self.model.add(Activation('elu'))
            self.model.add(LSTM(200, return_sequences=False, ))
            self.model.add(Dropout(0.35))
            self.model.add(Activation('elu'))

            # ------------------------------------------ MIXED Architectures -----------------------------------------------

        elif self.mode == 'LSTMGRU':

            self.model.add(GRU(100, input_shape=self.input_shape, return_sequences=True, ))
            self.model.add(Dropout(0.35))
            self.model.add(LSTM(200, return_sequences=False, ))
            self.model.add(Dropout(0.35))

        elif self.mode == 'DashLSTMGRU':

            self.model.add(GRU(100, input_shape=self.input_shape, recurrent_activation='elu', return_sequences=True))
            self.model.add(Dropout(0.35))
            self.model.add(Activation('relu'))
            self.model.add(LSTM(150, recurrent_activation='elu', return_sequences=False))
            self.model.add(Dropout(0.35))
            self.model.add(Activation('relu'))

        elif self.mode == 'TWSTDLSTMGRU':

            self.model.add(GRU(100, input_shape=self.input_shape, recurrent_activation='relu', return_sequences=True, ))
            self.model.add(Dropout(0.35))
            self.model.add(Activation('elu'))
            self.model.add(LSTM(150, recurrent_activation='relu', use_bias=True, return_sequences=False, ))
            self.model.add(Dropout(0.35))
            self.model.add(advanced_activations.LeakyReLU(alpha=0.3))

        elif self.mode == 'ActLSTMGRU':

            self.model.add(GRU(input_shape=self.input_shape, units=100, return_sequences=True))
            self.model.add(Dropout(0.35))
            self.model.add(Activation('elu'))
            self.model.add(LSTM(150, return_sequences=False))
            self.model.add(Dropout(0.35))
            self.model.add(Activation('relu'))

        elif self.mode == 'CoreLSTMGRU':

            self.model.add(GRU(100, input_shape=self.input_shape, return_sequences=True, ))
            self.model.add(Dropout(0.35))
            self.model.add(Activation('elu'))
            self.model.add(LSTM(200, return_sequences=False, ))
            self.model.add(Dropout(0.35))
            self.model.add(advanced_activations.LeakyReLU(alpha=0.3))

        # ----------------------------------------------- Final Layers -------------------------------------------------

        self.model.add(Dense(units=self.output_shape))

        self.model.add(Activation('linear'))

        self.model.compile(loss='mse', optimizer='rmsprop')
        return self.model

    def learn(self, x, y, initiate=False):
        if initiate:
            self.make_model()

        self._history = self.model.fit(x, y, batch_size=64, epochs=self._epochs,
                                       validation_split=0.25,
                                       verbose=(2 if self.LOG else 0))

        return self.model

    def predict(self, x):
        assert self.model is not None

        y_predict = self.model.predict(x)
        return y_predict

    def evaluate(self, x, y):
        assert self.model is not None

        evaluation = self.model.evaluate(x, y, verbose=0)
        return evaluation

    def plot_training_history(self, ax, name=None):
        ax.plot(self._history.history['loss'])
        ax.plot(self._history.history['val_loss'])
        ax.set_title(f'model loss {name}')
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')
        ax.legend(['train', 'validation'], loc='upper left')

    def plot_result(self, ax, x, name=None, t=None):
        y = self.predict(x)
        if t is not None and t.shape[0] > 0:
            ax.plot(t, y, label='predict')
        else:
            ax.plot(y, label='predict')
        # ax.set_title(f'model loss {name}')
        ax.set_ylabel(f'price {name}')
        if t is None or t.shape[0] == 0:
            ax.set_xlabel('time in 5 min')

    def save(self, filename):
        self.model.save(f'models/model_{filename}_{self.mode}.h5')

    def load(self, filename):
        self.model = load_model(f'models/model_{filename}_{self.mode}.h5')


class EnsemblePredictor(Predictor):
    def __init__(self, mode=['TWSTDGRU', 'ActGRU', 'DashGRU',], epochs=10, input_shape=(None, config.feature_size),
                 output_shape=2):
        super(EnsemblePredictor, self).__init__(mode, epochs, input_shape, output_shape)

    def make_model(self):
        self.model = {}

        # ------------------------------------------- GRU Architectures ------------------------------------------------

        if 'GRU' in self.mode:
            model = Sequential()
            model.add(GRU(100, input_shape=self.input_shape, return_sequences=True, ))
            model.add(Dropout(0.35))
            model.add(GRU(200, return_sequences=False, ))
            model.add(Dropout(0.35))
            self.model['GRU'] = model

        if 'DashGRU' in self.mode:
            model = Sequential()

            model.add(GRU(input_shape=self.input_shape, recurrent_activation='elu', units=100, return_sequences=True))
            model.add(Dropout(0.35))
            model.add(Activation('elu'))
            model.add(GRU(150, recurrent_activation='elu', return_sequences=False))
            model.add(Dropout(0.35))
            model.add(Activation('elu'))
            self.model['DashGRU'] = model

        if 'TWSTDGRU' in self.mode:
            model = Sequential()
            model.add(GRU(100, input_shape=self.input_shape, recurrent_activation='relu', return_sequences=True, ))
            model.add(Dropout(0.35))
            model.add(advanced_activations.LeakyReLU(alpha=0.3))
            model.add(GRU(150, recurrent_activation='relu', return_sequences=False, ))
            model.add(Dropout(0.35))
            model.add(advanced_activations.LeakyReLU(alpha=0.3))
            self.model['TWSTDGRU'] = model

        if 'TWSTD2GRU' in self.mode:
            model = Sequential()
            model.add(GRU(100, input_shape=self.input_shape, recurrent_activation='relu', return_sequences=True, ))
            model.add(Dropout(0.35))
            model.add(Activation('elu'))
            model.add(GRU(150, recurrent_activation='relu', return_sequences=False, ))
            model.add(Dropout(0.35))
            model.add(Activation('elu'))
            self.model['TWSTD2GRU'] = model

        if 'ActGRU' in self.mode:
            model = Sequential()

            model.add(GRU(input_shape=self.input_shape, units=100, return_sequences=True))
            model.add(Dropout(0.35))
            model.add(Activation('elu'))
            model.add(GRU(150, return_sequences=False))
            model.add(Dropout(0.35))
            model.add(Activation('elu'))
            self.model['ActGRU'] = model

        if 'CoreGRU' in self.mode:
            model = Sequential()

            model.add(GRU(100, input_shape=self.input_shape, return_sequences=True, ))
            model.add(Dropout(0.35))
            model.add(advanced_activations.LeakyReLU(alpha=0.3))
            model.add(GRU(200, return_sequences=False, ))
            model.add(Dropout(0.35))
            model.add(advanced_activations.LeakyReLU(alpha=0.3))
            self.model['CoreGRU'] = model

        # ------------------------------------------ LSTM Architectures ------------------------------------------------

        if 'LSTM' in self.mode:
            model = Sequential()

            model.add(LSTM(100, input_shape=self.input_shape, return_sequences=True, ))
            model.add(Dropout(0.35))
            model.add(LSTM(200, return_sequences=False, ))
            model.add(Dropout(0.35))
            self.model['LSTM'] = model

        if 'DashLSTM' in self.mode:
            model = Sequential()

            model.add(GRU(100, input_shape=self.input_shape, recurrent_activation='elu', return_sequences=True, ))
            model.add(Dropout(0.35))
            model.add(Activation('elu'))
            model.add(GRU(200, recurrent_activation='elu', return_sequences=False, ))
            model.add(Dropout(0.35))
            model.add(advanced_activations.LeakyReLU(alpha=0.3))
            self.model['DashLSTM'] = model

        if 'TWSTDLSTM' in self.mode:
            model = Sequential()

            model.add(LSTM(100, input_shape=self.input_shape, recurrent_activation='relu', return_sequences=True, ))
            model.add(Dropout(0.35))
            model.add(Activation('elu'))
            model.add(LSTM(150, recurrent_activation='relu', return_sequences=False, ))
            model.add(Dropout(0.35))
            model.add(Activation('elu'))
            self.model['TWSTDLSTM'] = model

        if 'ActLSTM' in self.mode:
            model = Sequential()

            model.add(LSTM(input_shape=self.input_shape, units=100, return_sequences=True))
            model.add(Dropout(0.35))
            model.add(Activation('relu'))
            model.add(LSTM(150, return_sequences=False))
            model.add(Dropout(0.35))
            model.add(Activation('relu'))
            self.model['ActLSTM'] = model

        if 'CoreLSTM' in self.mode:
            model = Sequential()

            model.add(LSTM(100, input_shape=self.input_shape, return_sequences=True, ))
            model.add(Dropout(0.35))
            model.add(Activation('elu'))
            model.add(LSTM(200, return_sequences=False, ))
            model.add(Dropout(0.35))
            model.add(Activation('elu'))
            self.model['CoreLSTM'] = model

        # ------------------------------------------ MIXED Architectures -----------------------------------------------

        if 'LSTMGRU' in self.mode:
            model = Sequential()

            model.add(GRU(100, input_shape=self.input_shape, return_sequences=True, ))
            model.add(Dropout(0.35))
            model.add(LSTM(200, return_sequences=False, ))
            model.add(Dropout(0.35))
            self.model['LSTMGRU'] = model

        if 'DashLSTMGRU' in self.mode:
            model = Sequential()

            model.add(GRU(100, input_shape=self.input_shape, recurrent_activation='elu', return_sequences=True, ))
            model.add(Dropout(0.35))
            model.add(Activation('elu'))
            model.add(LSTM(200, recurrent_activation='elu', return_sequences=False, ))
            model.add(Dropout(0.35))
            model.add(advanced_activations.LeakyReLU(alpha=0.3))
            self.model['DashLSTMGRU'] = model

        if 'TWSTDLSTMGRU' in self.mode:
            model = Sequential()

            model.add(GRU(100, input_shape=self.input_shape, recurrent_activation='relu', return_sequences=True, ))
            model.add(Dropout(0.35))
            model.add(Activation('elu'))
            model.add(LSTM(150, recurrent_activation='relu', return_sequences=False, ))
            model.add(Dropout(0.35))
            model.add(advanced_activations.LeakyReLU(alpha=0.3))
            self.model['TWSTDLSTMGRU'] = model

        if 'ActLSTMGRU' in self.mode:
            model = Sequential()

            model.add(GRU(input_shape=self.input_shape, units=100, return_sequences=True))
            model.add(Dropout(0.35))
            model.add(Activation('elu'))
            model.add(LSTM(150, return_sequences=False))
            model.add(Dropout(0.35))
            model.add(Activation('relu'))
            self.model['ActLSTMGRU'] = model

        if 'CoreLSTMGRU' in self.mode:
            model = Sequential()

            model.add(GRU(100, input_shape=self.input_shape, return_sequences=True, ))
            model.add(Dropout(0.35))
            model.add(Activation('elu'))
            model.add(LSTM(200, return_sequences=False, ))
            model.add(Dropout(0.35))
            model.add(advanced_activations.LeakyReLU(alpha=0.3))
            self.model['CoreLSTMGRU'] = model

        # ----------------------------------------------- Final Layers -------------------------------------------------

        for mode, model in self.model.items():
            model.add(Dense(units=self.output_shape))
            model.add(Activation('linear'))
            model.compile(loss='mse', optimizer='rmsprop')

        return self.model

    def learn(self, x, y, initiate=False):
        if initiate:
            self.make_model()

        self._histories = {}

        for mode, model in self.model.items():
            self._histories[mode] = model.fit(x, y, batch_size=64, epochs=self._epochs,
                                              validation_split=0.25,
                                              verbose=(2 if self.LOG else 0))

        # todo
        self._history = History()
        setattr(self._history, 'history', dict())
        self._history.history['loss'] = np.asarray([h.history['loss'] for h in self._histories.values()]).mean(axis=0)
        self._history.history['val_loss'] = \
            np.asarray([h.history['val_loss'] for h in self._histories.values()]).mean(axis=0)

        return self.model

    def predict(self, x):
        assert self.model is not None

        _predicts = []

        for model in self.model.values():
            _predicts.append(model.predict(x))

        y_predict = np.asarray(_predicts).mean(axis=0)
        return y_predict

    def evaluate(self, x, y):
        assert self.model is not None

        _evaluations = []

        for model in self.model.values():
            _evaluations.append(model.evaluate(x, y, verbose=0))

        evaluation = np.asarray(_evaluations).mean(axis=0)
        return evaluation

    def save(self, filename):
        for mode, model in self.model.items():
            model.save(f'models/ensemble_model_{filename}_{mode}.h5')

    def load(self, filename):
        self.model = {}
        for mode in self.mode:
            self.model[mode] = load_model(f'models/ensemble_model_{filename}_{mode}.h5')


def main():
    ep = EnsemblePredictor(input_shape=(None, 1), output_shape=1)
    ep.learn(np.array([[[1], [2], [3]], [[2], [3], [4]], [[4], [5], [6]]]), np.array([4, 5, 6]))

    import matplotlib.pyplot as plt
    ep.plot_training_history(plt.figure().add_subplot(111))
    plt.show()

    print(ep.predict(np.array([[[2], [3], [4]]])))


if __name__ == '__main__':
    print(device_lib.list_local_devices())
    main()
