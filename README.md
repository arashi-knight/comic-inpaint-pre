# Advancing Comic Image Inpainting: A Novel Dual-Stream Fusion Approach with Texture Enhancements

## Introduction
In the process of comic localization, a crucial step is to fill in the pixels obscured due to the removal of dialogue boxes or sound effect text. Comic inpainting is more challenging than natural image restoration due to its abstract structure and texture, which complicates semantic interpretation and synthesis, as well as the importance of high-frequency details like lines. This paper proposes the Texture-Structure Fusion Network (TSF-Net) with dual-stream encoder, introducing the Dual-stream Space-Gated Fusion (DSSGF) module for effective feature interaction. Additionally, a Multi-scale Histogram Texture Enhancement (MHTE) module is designed to enhance texture information aggregation dynamically. Visual comparisons and quantitative experiments demonstrate the effectiveness of the method, proving its superiority over existing techniques in comic inpainting.
## Example

![img.png](test_image/img.png)

## Datasets

### 1) Images
As most of our training manga images are under copyright, we recommend you to use restored [Manga109 dataset](http://www.manga109.org/en/). 
If you want to obtain the comic dataset mentioned in the paper, you can contact luckylan_001@163.com

### 2) Structural lines
Our model is trained on structural lines extracted by [Li et al.](https://www.cse.cuhk.edu.hk/~ttwong/papers/linelearn/linelearn.html). You can download their publically available [testing code](https://github.com/ljsabc/MangaLineExtraction).

### 3) SVAE

The pre trained model of SVAE can be derived from [SVAE](https://drive.google.com/file/d/1QaXqR4KWl_lxntSy32QpQpXb-1-EP7_L/view) download

### 4) Masks
Our model is trained on both regular masks (randomly generated rectangle masks) and irregular masks (provided by [Liu et al. 2017](https://arxiv.org/abs/1804.07723)). You can download publically available Irregular Mask Dataset from [their website](http://masc.cs.gmu.edu/wiki/partialconv).
Alternatively, you can download [Quick Draw Irregular Mask Dataset](https://github.com/karfly/qd-imd) by Karim Iskakov which is combination of 50 million strokes drawn by human hand.