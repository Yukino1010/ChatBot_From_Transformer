# ChatBot_From_Transformer

## Introduction and Results

This time I am going to use Transformer that I mentioned in my last repository to build a Chinese chatbot. <br>

The model structure will be same as my last impliment, but this time I'm going to use BertTokenizer to encode the sentence to token and 
add some randomness to the model's output to make sure that the same input can have some different results.<br>

Here is the result:<br>

<p align="center">
<img width="400px" src="https://github.com/Yukino1010/ChatBot_From_Transformer/blob/master/result/result1.png"/>
<img width="400px" src="https://github.com/Yukino1010/ChatBot_From_Transformer/blob/master/result/result2.png"/>
</p>

<br>
This result seems fantastic, the model can indeed handle the pattern of daily conversation. <br>
For example, when I say 你今年幾歲? (how old are you this year?) on the first picture the model reply 幼稚園畢業啦 (just graduated from kindergarten). <br>

In this case we can see the model should knew the meaning of 「幾歲」 to generate the sentence, and the things what surprised me was that the model reply a period of time (just graduated from kindergarten) instead of a point in time (8 years old), I think it may be caused by the nature of my dataset.

Finally, I also found something very funny. On the last line of the second picture, I say 你是笨蛋嗎? (Are you stupid?) and the model replys 我們都一樣 (we are the same!) 
😄😄
