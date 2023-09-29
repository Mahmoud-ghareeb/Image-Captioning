# Caption ME 🖼️

Welcome to **Caption ME**, a project dedicated to generating insightful captions for images. Dive deep into our approach, which uniquely combines the powers of convolutional neural networks and sequence models.

## 🧠 Model Architecture

The design philosophy behind our model architecture involves splitting the process into two main parts:

### 1. Encoder:

For the encoding mechanism, I leveraged the power of the pre-trained `MobileNetV3Large` model. This serves as our primary feature extractor, diving deep into the nuances of the images and translating them into a form digestible by our decoder.

### 2. Decoder:

The decoder has been meticulously designed with three major components:

   - **LSTM**: Captures the sequential nature of captions, ensuring smooth and natural descriptions.
   - **Multi-head attention**: Empowers our model to focus on salient features of the image dynamically, leading to more relevant captions.
   - **Dense Layer**: Outputs the final word predictions, adding to our caption sequence.

## 📊 Results

In terms of performance, this model stands out! After rigorous training and validation:
- Achieved an accuracy of a staggering **82%**, which outperforms many other models documented in similar notebooks.
- Recorded a commendable **BLEU score of 50.0**, underlining the linguistic quality of the generated captions.

## 📸 Inference

The real magic unveils during inference. The model provides captivating and apt descriptions, bringing out the essence of the images and giving them a voice.

---

Your journey with **Caption ME** promises to be insightful. Dive in, explore, and let's give words to images!
