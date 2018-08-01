# `LyaRT;Grid`

In this Universe everything is neat, even stochasticity.
Lyman alpha Resonant Scatter is dominated by random processes.
Although there is chaos, there is also an order.

## WARNING : THIS IS BETA VERSION! BUGS CAN BE FOUND, BE AWARE!

### Origins and motivation

Due to the Lyman alpha Radiative Transfer large complexity, the efforts of understanding it moved from pure analytic studies to the so-called radiative transfer Monte Carlo (RTMC) codes that by simulating Lyman alpha photons in arbitrary gas geometries. These codes provide useful information about the fraction of photons that manage to escape and the out coming lyman alpha line profiles. The RTMC approach has shown to reproduce the observed properties of LAEs.

`LyaRT;Grid` is a publicly available `python` package based on a RTMC (Orsi et al. 2012) able to predict large amounts of Lyman alpha line profiles and escape fractions with high accuracy. We designed this code hoping that it helps researches all over the wolrd to get a better understanding of the Universe.

The main premises of `LyaRT;Grid` are **fast** and **simple**. 

+ **Fast** : This code is able to predict Lyman alpha escape fractions and line profiles in an unprecedented low amount of time. In particular thousands of escape fractions and line profiles can be computed in less than a second of computational time. 

+ **Simple** : This code is *One-Line-Installing* and *One-Line-Running*. Everyone, from a good trained monkey, passing through undergrade studients, to researches with basic python knowlege should be able to use `LyaRT;Grid`.

### Installation

The easiest way to get `LyaRT;Grid` in your machine is through `pip`:

`pip install LyaRT_Grid`

This should do the trick. However, you can also install `LyaRT_Grid` downloading this repository and running

`pip install .`

in it.

Please, note that you should have about 1GB of free memory in you machine in order to install `LyaRT_Grid`. Also note that as ~1GB of data has to be downloaded it might take some time depending on your internet conection.

I don't like short installation guides, but really, there is nothing more to tell. However, if you find any trouble with installation please contact `sidgurung@cefca.es` or leave a comment. We will be pleased to help you! 
 
### Hands on the code.

( Assuming everything went smoothly in the installation... )

**Congratulations!!** You have just become one of the few chosen ones in the history of humankind to have the great pleasure of using `LyaRT_Grid`.



##### H5
###### H6

Alternatively, for H1 and H2, an underline-ish style:

Alt-H1
======

Alt-H2
------

Emphasis, aka italics, with *asterisks* or _underscores_.

Strong emphasis, aka bold, with **asterisks** or __underscores__.

Combined emphasis with **asterisks and _underscores_**.

Strikethrough uses two tildes. ~~Scratch this.~~

1. First ordered list item
2. Another item
⋅⋅* Unordered sub-list. 
1. Actual numbers don't matter, just that it's a number
⋅⋅1. Ordered sub-list
4. And another item.

⋅⋅⋅You can have properly indented paragraphs within list items. Notice the blank line above, and the leading spaces (at least one, but we'll use three here to also align the raw Markdown).

⋅⋅⋅To have a line break without a paragraph, you will need to use two trailing spaces.⋅⋅
⋅⋅⋅Note that this line is separate, but within the same paragraph.⋅⋅
⋅⋅⋅(This is contrary to the typical GFM line break behaviour, where trailing spaces are not required.)

* Unordered list can use asterisks
- Or minuses
+ Or pluses

Inline `code` has `back-ticks around` it.

```python
s = "Python syntax highlighting"
print s
```
