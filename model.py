import numpy as np
import tensorflow as tf
from tensorflow.keras import Model 
from tensorflow.keras.layers import LSTM, Dense, MultiHeadAttention, Embedding, LayerNormalization, Dropout
import einops
from tensorflow.keras.applications import MobileNetV3Large

class ImageFeatureExtractor(Model):
    def __init__(self, image_model):
        super(ImageFeatureExtractor, self).__init__()
        
        self.model = image_model
        
    def call(self, x):
        
        return self.model(x)
  
class DecoderLSTM(Model):
    def __init__(self, units, num_heads, key_dims, vocab_size, emb_size, dropout_rate):
        super().__init__()
        
        self.embd  = Embedding(vocab_size, emb_size, mask_zero=True)
        self.lstm = LSTM(units, return_sequences=True)
        self.drp1 = Dropout(dropout_rate)
        
        self.attn = MultiHeadAttention(num_heads, key_dims)
        self.nrm1 = LayerNormalization()
        self.dense = Dense(units, activation='relu')
        self.drp2 = Dropout(dropout_rate)
        
        self.outputs = Dense(vocab_size, activation='softmax')
        
    def call(self, x, encoder_output):
        
        x = self.embd(x)
        x = self.drp1(x)
        x = self.lstm(x)
        
        attn, attn_scores = self.attn(x, encoder_output, return_attention_scores=True)
        self.last_attention_scores = attn_scores
        
        x = x + self.nrm1(attn)
        x = self.dense(x)
        x = self.drp2(x)
        
        return self.outputs(x)
    
class CaptionMeLSTM(Model):
    def __init__(self, image_model, num_heads, key_dims, units, vocab_size, emb_size, dropout_rate):
        super(CaptionMeLSTM, self).__init__()
        
        self.encoder_model = ImageFeatureExtractor(image_model)
        self.decoder_model = DecoderLSTM(units, num_heads, key_dims, vocab_size, emb_size, dropout_rate)
        
    def call(self, inputs):
        x, y = inputs['encoder_inputs'], inputs['decoder_inputs']
        
        enc_out = self.encoder_model(x)
        enc_out = einops.rearrange(enc_out, 'b h w c -> b (h w) c')
        
        dec_out = self.decoder_model(y, enc_out)
        
        return dec_out
    
if __name__ == '__main__':
    vocab_size = 8633
    seq_len    = 33
    emb_size  = 256
    num_heads = 6
    key_dims = 256
    units = 256
    dropout_rate = .3

    rand_img = np.random.random((1, 299, 299, 3))/255
    rand_txt = np.random.random((1, 33))/255
    image_model = MobileNetV3Large(include_top=False, include_preprocessing=True)
    image_model.trainable = False

    model = CaptionMeLSTM(image_model, num_heads, key_dims, units, vocab_size, emb_size, dropout_rate)
    model({"encoder_inputs": rand_img, "decoder_inputs": rand_txt})
    model.load_weights('./weights/model_weights.h5')