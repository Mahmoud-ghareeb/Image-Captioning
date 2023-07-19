# Caption ME

A project that generates captions about images 

## Model
I have split the model into two parts
1. Encoder part <br>
   in the encoder part, I used a pre-trained MobileNetV3Large model to decode the features in the image
3. decoder part <br>
   I made the decoder part consist of Three parts
   1. LSTM
   2. Multi-head attention
   3. Dense for output

 ## results
I was able to get a pretty high accuracy score compared to other notebooks with an accuracy of 82% and a BELU score with 50.0

## inference
I got a pretty good description of the images
