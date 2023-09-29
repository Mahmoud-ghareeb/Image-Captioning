import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import cv2
import random
from utils import read_and_resize
from model import CaptionMeLSTM
from tensorflow.keras.applications import MobileNetV3Large

def inference(caption_me, src, wrd_to_idx, idx_to_wrd):
    image = cv2.imread(src)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(2,2,1)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

    def caption(src):
        test_img = read_and_resize(src)[None, :]

        tokens = []
        start = '[START]'

        vect = wrd_to_idx[start]
        inputs = np.reshape(vect, (1, 1))
        for i in range(33):
            outputs = caption_me({"encoder_inputs":test_img, "decoder_inputs": inputs}).numpy()
            preds = outputs[:, i, :]
            idx = tf.argmax(preds, axis=-1)[:, None]
            wrd = idx_to_wrd[idx.numpy()[0][0]]
            if wrd == '[END]' or wrd == '[START]':
                break
            # print(wrd, end=" ")
            tokens.append(wrd)
            inputs = tf.concat([inputs, idx], axis=1)

        return ' '.join(tokens)

    return caption(src)


if __name__=='__main__':
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

    caption_me = CaptionMeLSTM(image_model, num_heads, key_dims, units, vocab_size, emb_size, dropout_rate)
    caption_me({"encoder_inputs": rand_img, "decoder_inputs": rand_txt})
    caption_me.load_weights('weights/model_weights.h5')

    with open('results/vocab.pkl', 'rb') as f:
        idx_to_wrd, wrd_to_idx = pickle.load(f)

    src = 'imgs/test.jpg'
    result = inference(caption_me, src, wrd_to_idx, idx_to_wrd)
    print("Inference Result:", result)
