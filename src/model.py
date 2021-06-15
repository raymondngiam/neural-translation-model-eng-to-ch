import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Masking, LSTM, Embedding, Dense

class EndTokenEmbedLayer(Layer):
    def __init__(self):
        super(EndTokenEmbedLayer, self).__init__()

    def build(self, input_shape):
        self.embedding_size = input_shape[-1]
        self.embedding = self.add_weight(shape=(self.embedding_size,),
                                         initializer='random_normal',
                                         name='end_token_embedding')
  
    def call(self, inputs):
        one_row = tf.reshape(self.embedding,(-1,1,self.embedding_size))
        end_token_output = tf.tile(one_row,[tf.shape(inputs)[0],1,1])
        return tf.concat((inputs,end_token_output),axis=1)

def Encoder(input_shape):
    inputs = Input(input_shape)
    h = EndTokenEmbedLayer()(inputs)
    h = Masking(mask_value=0.)(h)
    lstm , hidden_state, cell_state = LSTM(512,return_sequences=True,return_state=True)(h)
    model = Model(inputs=inputs, outputs=[hidden_state, cell_state])
    return model

class Decoder(Model):
    def __init__(self,input_embedding_dim):
        super(Decoder, self).__init__()
        self.embedding = Embedding(input_dim = input_embedding_dim[0],
                                   output_dim = input_embedding_dim[1],
                                   mask_zero = True)
        self.lstm = LSTM(units=512, return_sequences=True, return_state=True)
        self.dense = Dense(units=input_embedding_dim[0])

    def call(self,inputs,hidden_state = None,cell_state = None):
        h = self.embedding(inputs)
        if hidden_state != None and cell_state != None:
            lstm,hidden,cell = self.lstm(h,initial_state =[hidden_state,cell_state])
        else:
            lstm,hidden,cell = self.lstm(h)
        h = self.dense(lstm)
        return h,hidden,cell

class NeuralTranslationModel(Model):
    def __init__(self,encoder_input_shape,decoder_input_shape):
        super(NeuralTranslationModel, self).__init__()
        self.encoder = Encoder(input_shape=encoder_input_shape)
        self.decoder = Decoder(input_embedding_dim=decoder_input_shape)
        self.model_trainable_variables = self.encoder.trainable_variables + \
                                         self.decoder.trainable_variables    
  
    def chinese_data_io(self,chinese_data):
        input_data = chinese_data[:,0:tf.shape(chinese_data)[1]-1]
        output_data = chinese_data[:,1:tf.shape(chinese_data)[1]]
        return(input_data,output_data)

    def call(self,inputs):
        (encoder_in, decoder_in)=inputs
        hidden_state ,cell_state = self.encoder(encoder_in)
        dense_output, _, _ = self.decoder(decoder_in, hidden_state, cell_state)
        return dense_output

    @tf.function
    def train_step(self,data):        
        (english,chinese) = data
        chinese_input, chinese_output = self.chinese_data_io(chinese)  
        with tf.GradientTape() as tape:        
            hidden_state ,cell_state = self.encoder(english)
            dense_output, _, _ = self.decoder(chinese_input, hidden_state, cell_state)
            loss = tf.math.reduce_mean(self.compiled_loss(chinese_output,dense_output))
            grads = tape.gradient(loss, self.model_trainable_variables)
            self.optimizer.apply_gradients(zip(grads,
                                               self.model_trainable_variables))
            self.compiled_metrics.update_state(chinese_output,dense_output)
        return {m.name:m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        (english,chinese) = data
        chinese_input, chinese_output = self.chinese_data_io(chinese) 
        hidden_state ,cell_state = self.encoder(english)
        dense_output, _, _ = self.decoder(chinese_input, hidden_state, cell_state)
        loss = tf.math.reduce_mean(self.compiled_loss(chinese_output,dense_output))
        self.compiled_metrics.update_state(chinese_output,dense_output)
        return {m.name:m.result() for m in self.metrics}