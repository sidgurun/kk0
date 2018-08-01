# `LyaRT;Grid`

In this Universe everything is neat, even stochasticity.
Lyman alpha Resonant Scatter is dominated by random processes.
Although there is chaos, there is also an order.

## WARNING : THIS IS BETA VERSION! BUGS EVERYWHERE, BE AWARE!

## Origins and motivation

Due to the Lyman alpha Radiative Transfer large complexity, the efforts of understanding it moved from pure analytic studies to the so-called radiative transfer Monte Carlo (RTMC) codes that by simulating Lyman alpha photons in arbitrary gas geometries. These codes provide useful information about the fraction of photons that manage to escape and the out coming lyman alpha line profiles. The RTMC approach has shown to reproduce the observed properties of LAEs.

`LyaRT;Grid` is a publicly available `python` package based on a RTMC (Orsi et al. 2012) able to predict large amounts of Lyman alpha line profiles and escape fractions with high accuracy. We designed this code hoping that it helps researches all over the wolrd to get a better understanding of the Universe.

The main premises of `LyaRT;Grid` are **fast** and **simple**. 

+ **Fast** : This code is able to predict Lyman alpha escape fractions and line profiles in an unprecedented low amount of time. In particular thousands of escape fractions and line profiles can be computed in less than a second of computational time. 

+ **Simple** : This code is *One-Line-Installing* and *One-Line-Running*. Everyone, from a good trained monkey, passing through undergrade studients, to researches with basic python knowlege should be able to use `LyaRT;Grid`.

Although `LyaRT;Grid` is completely open source and is available for everyone, please, if you are using this code ( for whatever reason ) we will be very glad to hear about you and the how you are using it. it's just a curiosity and efficiency matter. Maybe we can help you to have a smoother experience with `LyaRT;Grid`. For anything, please, contact `sidgurung@cefca.es`. Thank you.

## Installation

The easiest way to get `LyaRT;Grid` in your machine is through `pip`:

`pip install LyaRT_Grid`

This should do the trick. However, you can also install `LyaRT_Grid` downloading this repository and running

`pip install .`

in it.

Please, note that you should have about 1GB of free memory in you machine in order to install `LyaRT_Grid`. Also note that as ~1GB of data has to be downloaded it might take some time depending on your internet conection.

I don't like short installation guides, but really, there is nothing more to tell. However, if you find any trouble with installation please contact `sidgurung@cefca.es` or leave a comment. We will be pleased to help you! 
 
## Hands on the code.

( Assuming everything went smoothly in the installation... )

**Congratulations!!** You have just become one of the few chosen ones in the history of humankind to have the great pleasure of using `LyaRT_Grid`.

**The units**: This code uses velocites in km/s , column densities in cm^{-2} and wavelengths in meters.

First let's have a look to the dynamical range of the different parameters that `LyaRT;Grid` covers. For this we only need to do:

```python
import LyaRT_Grid as Lya

Lya.Print_the_grid_edges()
```


### Predicting thousands of Lyman alpha escape fractions.

Let's move to the easiest and one of the most powerfull products of `LyaRT;Grid`: predicting huge amounts of Lyman alpha escape fractions. 

In theory only one line is needed to predict the escape fraction for a thin shell geometry with expasion velocity (V) of 200km/s, logarithmic of column density (logNH) of 19.5 and dust optical depth (ta) of 0.1 :

```python
f_esc_Arr = Lya.RT_f_esc( 'Thin_Shell' , [ 200 ] , [ 19.5 ] , [ 0.1 ] )
```  

In this way `f_esc_Arr` is an Array of 1 dimension and length 1 that contains the predicted escape fraction for this configuration. 

However, `LyaRT;Grid` implements several gas geometries and is optimized to obtain large amount of escape fractions with only one line of code, so lets expand this a little bit more. If we want to compute the escape fraction in a thin shell outflow with the configurations { V , logNH , ta } , { 200 , 19.5 , 0.1 }, { 300 , 20.0 , 0.01 } and { 400 , 20.5 , 0.001 } we could do

```python
Geometry = 'Thin Shell' # Other options: 'Galactic Wind' or 'Bicone_X_Slab'

V_Arr     = [  200 ,  300 , 400   ] # Expansion velocity array in km/s

logNH_Arr = [ 19.5 , 20.0 , 20.5  ] # Logarithmic of column densities array in cm**-2

ta_Arr    = [  0.1 , 0.01 , 0.001 ] # Dust optical depth Array

f_esc_Arr = Lya.RT_f_esc( Geometry , V_Arr , logNH_Arr , ta_Arr ) 
```  

The variable `f_esc_Arr` is an Array of 1 dimension and length 3 that encloses the escape fractions for the configurations. If the user wants to change the outflow gas geometry just do

```python
Geometry = 'Galactic Wind' # Other options: 'Thin Shell' or 'Bicone_X_Slab'

f_esc_Arr = Lya.RT_f_esc( Geometry , V_Arr , logNH_Arr , ta_Arr ) 
```  

Note that only one geometry can be used at the same time. If you want to compute different escape fractions ( or line profiles ) for different configurations you will need to call `LyaRT;Grid` once per geometry. 

These examples shows how the `'Thin_Shell'` and `'Galactic_Wind'` geometries work. These geometries have spherical symmetry so there is no Line of Sight (LoS) dependence in the output escape fraction or line profile. However, `LyaRT;Grid` implements a non-spherical-symmetric geometry, the `'Bicone_X_Slab'` geometry (for details, again, we refer you to the presentation letter). In this particular geometry the escape fraction (and line profile) depends on the LoS. In particular, if you observed face-on (throught the biconical outflow) the optical depth is lower than observeing edge-on (through the static dense slab).

In order to tell `LyaRT;Grid` the orientation of observation ( edge-on or face-on ) the user needs to provide anther varible when calling the code: `Inside_Bicone_Arr`. If it's not given it is assumed that it's always observed face-on. This variable has to be a boolean array with the same sice as `V_Arr` , `logNH_Arr` ot `ta_Arr`. Additionally, the apperture angle of the bicone is 45deg, so to produce a set of escape fractions with random orientations in the biconenical geometry you should use:

```python
import numpy as np

Geometry = 'Bicone_X_Slab'

Area_in_bicone = 1. - np.cos( np.pi/4. ) # the apperture angle is pi/4

Inside_Bicone_Arr = np.random.rand( len(V_Arr) ) < Area_in_bicone

f_esc_Arr = Lya.RT_f_esc( Geometry , V_Arr , logNH_Arr , ta_Arr , Inside_Bicone_Arr=Inside_Bicone_Arr ) 
```  

#### Deeper options on predictint the escape fraction (Unuseful section?).


There are many algorithims implemented to compute `f_esc_Arr` and by default `LyaRT;Grid` uses machine learning decision tree regressor and a parametric equation for the escape fraction as a function of the dust optical depth (Go to the `LyaRT;Grid` presentation paper Gurung-Lopez et al. in prerp for more information). These settings were chosen as default since they give the best performance. However the user might want to change the computing algorithim so here there is a guide with all the options available.

+ `MODE` variable refers to mode in which the escape fraction is computed. There are 3 ways in which `LyaRT;Grid` can compute this. i) `'Raw'` Using the raw data from the RTMC (Orsi et al. 2012). ii) `'Parametrization'` Assume a parametruc equation between the escape fraction and the dust optical depth that allow the extend the calculation outside the grid with the highest accuracy (in `LyaRT;Grid`). iii) `'Analytic'` Use of the recalibrated analytic equations presented by Gurung-Lopez et al. 2018. Note that the analytic mode is not able in the bicone geometry although it is in the `'Thin_Shel'` and `'Galactic_Wind'`


+ `Algorithm` varible determines the technique used. This can be i) `'Intrepolation'`: lineal interpoation is used.  ii) `'Machine_Learning'` machine learning is used. To determine which machine learning algorithm you would like to use please, provide the variable `Machine_Learning_Algorithm`. The machine learning algorithms implemented are Decision tree regressor (`'Tree'`), Random forest regressor (`'Forest'`) and KN regressor (`'KN'`). The machine learning is implemented by `Sci-kit-learn`, please, visit their webside for more information (http://scikit-learn.org/stable/).

```python
MODE = 'Raw' # Other : 'Parametrization' , 'Analytic'

Algorithm = 'Intrepolation' # Other : 'Machine_Learning'

Machine_Learning_Algorithm = 'KN' # Other 'Tree' , 'Forest'

f_esc_Arr = Lya.RT_f_esc( Geometry , V_Arr , logNH_Arr , ta_Arr , MODE=MODE ) 
```  

Finally, any combination of `MODE` , `Algorithm` and `Machine_Learning_Algorithm` is allowed. However, note that the variable `Machine_Learning_Algorithm` is useless if `Algorithm='Intrepolation'`.

### Predicting thousands of Lyman alpha escape fractions.

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
