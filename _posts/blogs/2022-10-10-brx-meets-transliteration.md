---
layout: post
title:  "BRX meets Trans - Bodo Machine Transliteration"
date:   2022-10-10 01:56:00 +0530
categories: blogs
---

For a week, I have been trying to find an approach to the transliteration problem for the Bodo language. It is one of the interesting problems that aims to convert a language in one written script to another (often its original form). After looking for research direction for Indian languages came across the [Aksharantar](https://ai4bharat.iitm.ac.in/aksharantar) dataset released by the AI4Bharat team from IIT Madras.

So, the next thought that came is to train a deep-learning model. After looking around for a while, I came across the IndicXlit model - a Transformer based model.

**Let's explore**

First, let us install AI4Bharat Transliteration python package.

```python
pip install ai4bharat-transliteration
```
Let's load Bodo XlitEngine, doing it the first time will download the model. In case rescore is set to True it will download the language model for rescoring.

{% highlight python %}
from ai4bharat.transliteration import XlitEngine
b = XlitEngine("brx", beam_width=5, rescore=False)
{% endhighlight %}

It is really easy to use the python package and transliteration is possible by using the ```translit_word()``` function. Let us test it for the _**Nwngni**_ (meaning = yours).

```python
transliterated_output = b.translit_word("nwngni", topk=5)
print(transliterated_output['brx'])
# Output = {'brx': ['नोंनि', 'नौंनि', 'नोङनि', 'नोंगनि', 'नोंनो']}
```

It returns a list of transliterated words. We can see the word is correctly transliterated to _**नोंनि**_ in Devanagari. Yah! It is pretty good. Hmm, now what? The output is pretty good. We shall transliterate more words. It is more fun to see transliteration at the sentence level.

**Let's explore further**

The first thought - **NEED DATASET!** 

Luckily, I already had mined parallel segments (to be released publicly soon) of the Bodo corpus that contains 130 sentences in both Roman and Devanagari script. So, the idea is to first convert the sentences from Roman &rarr; Devanagari.

```python
from bodotokenizer import tokenize
from indicnlp.tokenize import indic_tokenize

def sentence_bodo_tokenize(x):
        """Tokenize sentences in Bodo into tokens using BodoTokenizer"""
        return tokenize(x)
    
def sentence_english_tokenize(x):
        """Tokenize English sentences into tokens using IndicNLP tokenizer"""
        l = []
        for t in indic_tokenize.trivial_tokenize(x):
            l.append(t)
        x = ' '.join(l)
        return x
```
```python
def devanagarized(x):
    """Transliterate Roman to Devanagri script
    """
    l = []
    for t in x.split():
        o = b.translit_word(t, topk=5)
        l.append(o['brx'][0])
        
    x = ' '.join(l)
    return x
```

```python
import pandas as pd

# Corpus path
path = '/home/maharaj/repositories/bodonlp/bodo-bs-parallel-corpus/dataset/bodo_bhasha_sangam_parallel_corpus_with_roman.csv'

corpus = pd.read_csv(path, header=None)

devanagari_sentences = corpus[0]

roman_sentences = corpus[1]

devanagari_tokenized_sentences = devanagari_sentences.apply(sentence_bodo_tokenize)

roman_tokenized_sentences = roman_sentences.apply(sentence_english_tokenize)

# Transliterated Bodo Devanagari sentences
devanagari_transliterated_tokenized_sentences = roman_tokenized_sentences.apply(devanagarized)
```

The ```devanagarized()``` function transliterate Roman to Devanagari. The variable ```devanagari_transliterated_tokenized_sentences``` contains the Devanagari sentences from Roman. Let's evaluate the performance. We would need performance metrics to measure transliteration quality. One of the performance metrics we can use is [Word Error Rate](https://en.wikipedia.org/wiki/Word_error_rate)    (WER). We will use one  ```jiwer``` python package to compute the WER.

```python
pip install jiwer
```
```python
from jiwer import wer

ground_truth = devanagari_tokenized_sentences.tolist()
hypothesis = devanagari_transliterated_tokenized_sentences.tolist()

error = wer(ground_truth, hypothesis)

print(error)
# Output: 0.17073170731707318
```

Great! We get WER as ~ 0.17 (lower the wer, better the transliteration). 

**Let's solve some fun problem now**

In the digital advent and explosion of Natural language usage in the digital realm, English has been a language of choice for communication for various socio-economic reasons. Looking back at the history of the evolution of the Bodo language, we see its struggle and adoption of various scripts for writing despite having a rich oral tradition. One of the readily available materials on the web is Bodo song lyrics in Roman form. So, how about we transliterate it? Let us find lyrics resources on the web first. Found one [Bodo Song Lyrics Blogspot](https://bodosongslyrics.blogspot.com/), it contains Bodo song lyrics in Roman script. Now, we need to get the lyrics.

Hmm. How? **Web Scraping**. Since the contents are on multiple pages, instead of scraping it page-wise, I looked for [sitemap.xml](https://bodosongslyrics.blogspot.com/sitemap.xml) file. The sitemap contains a list of URLs of the post content. Next, we need to extract the URLs of the blog post from the XML file.

```python
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

tree = ET.parse('sitemap.xml')
root = tree.getroot()
tag = "{http://www.sitemaps.org/schemas/sitemap/0.9}loc"

urls = []

for descendent in root.iter(tag):
    urls.append(descendent.text)

print(urls)
```

Now, we found urls of the blog post. 

```bash
https://bodosongslyrics.blogspot.com/2022/07/hello-jarwo-bodo-song-lyrics.html
https://bodosongslyrics.blogspot.com/2022/07/khanai-gwja-bodo-song-lyrics.html
https://bodosongslyrics.blogspot.com/2022/06/bwisagi-sikhwla-bodo-song-lyrics.html
https://bodosongslyrics.blogspot.com/2022/06/jahang-jahang-bodo-song-lyrics.html
https://bodosongslyrics.blogspot.com/2022/06/sobaijwng-samojwng-lyrics.html
https://bodosongslyrics.blogspot.com/2022/03/dalmia-cement-jwng-lyrics-bodo-song.html
https://bodosongslyrics.blogspot.com/2021/07/phwi-phwi-phwi-bodo-song-lyricsnikita.html
https://bodosongslyrics.blogspot.com/2021/07/akhai-nwngni-lyrics-swmkhwr-daimary.html
https://bodosongslyrics.blogspot.com/2019/06/fwilwi-agwi-fwi-bodo-song-lyrics.html
https://bodosongslyrics.blogspot.com/2019/06/tu-tu-tu-agwi-sona-bodo-song-lyrics.html
https://bodosongslyrics.blogspot.com/2019/06/kharson-kharson-bodo-song-lyrics.html
https://bodosongslyrics.blogspot.com/2019/06/fagun-fagun-bwisagu-song-lyrics-singer.html
https://bodosongslyrics.blogspot.com/2019/06/nebai-thagwn-ang-bodo-song-lyrics.html
https://bodosongslyrics.blogspot.com/2019/06/gwrbw-khwona-lyrics.html
https://bodosongslyrics.blogspot.com/2019/06/siri-siri-mwthw-mwthw-lyrics.html
https://bodosongslyrics.blogspot.com/2019/06/langlaibai-lyrics-singer-jubeen-garg.html
https://bodosongslyrics.blogspot.com/2019/06/alai-aron-lyrics.html
https://bodosongslyrics.blogspot.com/2019/05/bir-jwhwlao-lyrics.html
https://bodosongslyrics.blogspot.com/2019/05/awi-bajwi-bodo-modern-bwisagu-song.html
https://bodosongslyrics.blogspot.com/2019/05/anjali-unflugged-bodo-song.html
```

We scrape the lyrics ```extract_lyrics()``` from the blog using ```BeautifulSoup```. The scraped contents of unnecessary text, such texts are removed. It can be categorized into three sub-parts unnecessary text before the actual lyrics' content, within the content of the lyrics, and below the content of the lyrics. The ```remove_metadata()```, ```remove_music_clues()``` and ```remove_footnote()``` functions clean the content.

```python
def extract_lyrics(url=''):
    if not url:
        return
    req = requests.get(url)
    parse = BeautifulSoup(req.text, 'html.parser')
    
    body_content = parse.find("div", class_="post-body post-content")

    body = body_content.find_all("span")

    # just to debug
    for i, b in enumerate(body):
        # print(i, b.text.strip())
        pass
    
    lyrics = []

    for i, t in enumerate(body):
        k = remove_metadata(t.text.strip())
        k = remove_music_clues(k)
        k = remove_footnote(k)
        
        if k:
            lyrics.append(k)

    ly = {
        "song-name": "",
        "lyrics": []
    }

            
    ly["song-name"] = song_name(url)
    ly["lyrics"] = lyrics

    return ly

def remove_metadata(text):
    text = re.sub("Vocal.*", "", text, flags=re.I)
    text = re.sub("Master.*", "", text, flags=re.I)
    text = re.sub("Direction.*", "", text, flags=re.I)
    text = re.sub("Starring.*", "", text, flags=re.I)
    text = re.sub("DOP.*", "", text, flags=re.I)
    text = re.sub("Arial.*", "", text, flags=re.I)
    text = re.sub("Singer.*", "", text, flags=re.I)
    text = re.sub("Starring.*", "", text, flags=re.I)
    text = re.sub("Mijing.*", "", text, flags=re.I)
    text = re.sub("Concept.*", "", text, flags=re.I)
    text = re.sub(".*Lyrics.*", "", text, flags=re.I)
    text = re.sub("Makeup.*", "", text, flags=re.I)
    text = re.sub("Make-Up.*", "", text, flags=re.I)
    text = re.sub("\nCo Cast.*", "", text, flags=re.I)
    text = re.sub("\n.*Music.*", "", text, flags=re.I)
    text = re.sub("\n.*Guiter.*", "", text, flags=re.I)
    text = re.sub("\n.*Light.*", "", text, flags=re.I)
    text = re.sub("\n.*", "", text, flags=re.I)
    text = re.sub("Produced.*", "", text, flags=re.I)
    text = re.sub("Assist.*", "", text, flags=re.I)
    text = re.sub("Aeriel.*", "", text, flags=re.I)
    text = re.sub("Lyricist.*", "", text, flags=re.I)
    text = re.sub("Genres.*", "", text, flags=re.I)
    text = re.sub("Romantic.*", "", text, flags=re.I)
    text = re.sub("Producer.*", "", text, flags=re.I)
    text = re.sub("Riya Brahma.*", "", text, flags=re.I)
    text = re.sub("Audio.*", "", text, flags=re.I)
    text = re.sub("Recording.*", "", text, flags=re.I)
    text = re.sub("Mixing.*", "", text, flags=re.I)
    text = re.sub("Video.*", "", text, flags=re.I)
    text = re.sub("Director.*", "", text, flags=re.I)
    text = re.sub("Editor.*", "", text, flags=re.I)
    text = re.sub("Production.*", "", text, flags=re.I)
    text = re.sub("Phwi phwi phwi Bodo Melody song is sung by Nikita Boro and written by Ibson Lal Baruah", "", text, flags=re.I)
    return text

def remove_music_clues(text):
    text = re.sub("Music.*", "", text, flags=re.I)
    text = re.sub("\..*", "", text, flags=re.I)
    text = re.sub("times", "", text, flags=re.I)
    return text

def remove_footnote(text):
    text = re.sub("Thanks for visiting Bodo Song Lyrics Site.", "", text, flags=re.I)
    text = re.sub("Thanks.*", "", text, flags=re.I)
    text = re.sub("Related.*", "", text, flags=re.I)
    text = re.sub("Visiting.*", "", text, flags=re.I)
    text = re.sub("You make.*", "", text, flags=re.I)
    return text
```

Finally, we got the lyrics. Below is one of the sample lyrics.

```
Hello Jarwo Dinailai Manithw Lajigusu
Hello Jarwo Dinailai Manithw Lajigusu
Gwswa Nwngnily manithw Jadwng Hakhu Dakhu
Gwswa Nwngnily manithw Jadwng Hakhu Dakhu
Awi Jarwo Dinwilai Manithw Usu Khuthu
Gwswa Nwngnily Manwthw jadwng hakhu dakhu
Bwisagu bwthwrkhai jadwng nama gwswa Aothu Aothu
Birglangny gwgswjwng obao thangnw nwgly swothu swothu
Rotho Dathangtho Agwi Angjwng dose tha
Khintha angnwba Nwngni gwswni khwtha
Rotho Dathangtho Agwi Angjwng dose tha
Khintha angnwba Nwngni gwswni khwtha
Gwswni khwthakhwo bungalaswi
Manw Thakhwmalo tharswi
Neywi neywi Nwngni somabwsw barlai langswi
Gwswni khwthakhwo bungalaswi
Manw Thakhwmalo tharswi
Neywi neywi Nwngni somabwsw barlai langswi
Mwswokhwo bayw hathayao naina
Hinjaokhwo layw akholao naina
Mwswokhwo bayw hathayao naina
Hinjaokhwo layw akholao naina
Bibari mwnbari lari lari
Juli Jalangjwbby fari fari
Nwnghalo Thanwswi Nama Harsingwi siri siri
Nwnghalo Thanwswi Nama Harsingwi siri siri
Hello Jarwo Dinailai Manithw Lajigusu
Gwswa Nwngnily manithw Jadwng Hakhu Dakhu
Awi Jarwo Dinwilai Manithw Usu Khuthu
Gwswa Nwngnily Manwthw jadwng hakhu dakhu
Bwisagu bwthwrkhai jadwng nama gwswa Aothu Aothu
Birglangny gwgswjwng obao thangnw nwgly swothu swothu
```
Now, we can the same functions from Section II, ```sentence_english_tokenize()``` to tokenize the text and ```devanagarized()``` to transliterate to Devanagari script line by line. Below is the transliterated lyric.

```
हेलल जारौ दिनायलाय मानिथो लाजिगुसु
हेलल जारौ दिनायलाय मानिथो लाजिगुसु
गोसोआ नोंनिलि मानिथो जादों हाखु दाखु
गोसोआ नोंनिलि मानिथो जादों हाखु दाखु
आयै जारौ दिनैलाय मानिथो उसु खुथु
गोसोआ नोंनिलि मानोथो जादों हाखु दाखु
बैसागु बोथोरखाय जादों नामा गोसोआ आवथु आवथु
बिरग्लांनि गोगसोजों अबाव थांनो नोग्लि सौथु सौथु
रथ दाथांथ आगै आंजों दसे था
खिन्था आंनोबा नोंनि गोसोनि खोथा
रथ दाथांथ आगै आंजों दसे था
खिन्था आंनोबा नोंनि गोसोनि खोथा
गोसोनि खोथाखौ बुंगालासै
मानो थाखोमाल थारसै
नेयै नेयै नोंनि समाबोसो बारलाय लांसै
गोसोनि खोथाखौ बुंगालासै
मानो थाखोमाल थारसै
नेयै नेयै नोंनि समाबोसो बारलाय लांसै
मोसौखौ बायो हाथायाव नायना
हिन्जावखौ लायो आखलाव नायना
मोसौखौ बायो हाथायाव नायना
हिन्जावखौ लायो आखलाव नायना
बिबारि मोनबारि लारि लारि
जुलि जालांजोबबि फारि फारि
नोंहाल थानोसै नामा हारसिङै सिरि सिरि
नोंहाल थानोसै नामा हारसिङै सिरि सिरि
हेलल जारौ दिनायलाय मानिथो लाजिगुसु
गोसोआ नोंनिलि मानिथो जादों हाखु दाखु
आयै जारौ दिनैलाय मानिथो उसु खुथु
गोसोआ नोंनिलि मानोथो जादों हाखु दाखु
बैसागु बोथोरखाय जादों नामा गोसोआ आवथु आवथु
बिरग्लांनि गोगसोजों अबाव थांनो नोग्लि सौथु सौथु
```

Although the transliteration is not perfect it provides Bodo with language-technologies inclusion and the possibility of improvement in the domain of Natural language processing. Having such kind of transliteration system greatly impacts other major NLP tasks such as:

1. Dataset creation: The majority of the text on social media is communicated through English alphabets - this is due to the large skewness in the world languages and its language-technologies availability. Hence, curation of the large text from social media in Roman form would require transliteration technology to be used to convert the text corpora into its original form/script.  

2. On-fly text conversion interface: The dataset creation process often requires humans to manually write language text for example in machine translation - trained translators translate the sentence from one to another, to create the dataset. The user experience and speed of the translation depends on the translator's ability to map the English alphabet keyboard to its language keyboard layout. So, having an interface that allows translators to write in roman script and transliterate in the original written script is helpful.

**Source Code**

I hope it was an interesting read. I have released this [tiny exploration source code](https://github.com/maharajbrahma/bodo-lyrics-transliteration) on GitHub under MIT License. Please feel free to check it out. The source code contains functions or code snippets which I have excluded in the blog for the sake of simplicity.


It was a very fun exploration for me!

**Acknowledgments**

1. Super thanks to AI4Bharat team for making the dataset and model opensource: [https://ai4bharat.org/aksharantar](https://ai4bharat.org/aksharantar)

2. Thanks to Anoop Kunchukuttan for IndicNLP Library: [https://github.com/anoopkunchukuttan/indic_nlp_library](https://github.com/anoopkunchukuttan/indic_nlp_library)

3. Thanks to [https://bodosongslyrics.blogspot.com/](https://bodosongslyrics.blogspot.com/) for Bodo songs lyrics

**References**

1. Madhani, Y., Parthan, S., Bedekar, P., Khapra, R., Seshadri, V., Kunchukuttan, A., ... & Khapra, M. M. (2022). [Aksharantar: Towards building open transliteration tools for the next billion users.](https://arxiv.org/abs/2205.03018) arXiv preprint arXiv:2205.03018.

2. [https://github.com/AI4Bharat/IndicXlit](https://github.com/AI4Bharat/IndicXlit)

3. [https://pypi.org/project/ai4bharat-transliteration/](https://pypi.org/project/ai4bharat-transliteration/)

4. [https://huggingface.co/datasets/ai4bharat/Aksharantar/tree/main](https://huggingface.co/datasets/ai4bharat/Aksharantar/tree/main)