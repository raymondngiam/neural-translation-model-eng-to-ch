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