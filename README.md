## Neural Translation Model (English-to-Chinese)

---

### Overview

This project is to develop a neural translation model to translate **english** sentences **into chinese** using recurrent neural networks, namely LSTM (Long Short Term Memory).

The project was adopted from Coursera course **Customising your models with TensorFlow 2**. The original project was to create a **english-to-german** translation model.

Model layout of the **english-to-chinse neural translation model** is a follows:

<img src='images/neural_translation_model_chinese.png'>

Legend:

<img src='images/neural_translation_model_key.png'>

<p>Image credit: Image was adopted and modified based on <i>Coursera course - Customising your models with TensorFlow 2</i>'s capstone project image.</p>

---

### Dataset

We are using the **Chinese (Mandarin) - English** language dataset from http://www.manythings.org/anki/.

This dataset consists of over 24,697 pairs of sentences in English and Chinese.

Download the dataset from <a href='http://www.manythings.org/anki/cmn-eng.zip'>this link</a>, and extract the `cmn.txt` raw text file into the path `<repo_root>/data/`.

---

### Data Preprocessing

Notebook: <a href='01-DataPreprocessing.ipynb'>01-DataPreprocessing.ipynb</a>

Raw data from the dataset is preprocessed to split the chinese texts into an array of individual characters, appended with `<start>` and `<end>` tokens.

A sample of 5 preprocessed text sequences is as shown below:

<img src='images/DataPreprocessing-01.png'>

Final preprocessed dataset consists of `24089` pairs of english-to-chinese translation sentenses.

---

### Tokenization

Notebook: <a href='02-Tokenization.ipynb'>02-Tokenization.ipynb</a>

The chinese characters is then tokenized using `tensorflow.keras.preprocessing.text.Tokenizer`.

Word count for top 10 highest frequency characters is as shown below:

<img src='images/Tokenization-01.png'>

There are `3438` unique characters in the chinese sequences, include `<start>`, `<end>` tokens and punctuation marks like `。，？！`.

With the tokenization completed, the individual chinese characters in the sentences are then converted into their respective tokens as shown below:

<img src='images/Tokenization-02.png'>

---