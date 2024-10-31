# BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

### 🔥 **Article**  
https://arxiv.org/pdf/2201.12086

### 🔥 **Blog**  
https://www.salesforce.com/blog/blip-bootstrapping-language-image-pretraining/

### 🔥 **Code**  
https://github.com/salesforce/BLIP

## Abstract
BLIP effectively utilizes the noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones.


## Background and Motivation
+ Existing methods have two major limitations
  + **Model perspective**: most methods either adopt an encoder-based model or an encoder-decoder. encoder-based models are less straightforward to directly transfer to text generation tasks (e.g. image captioning), whereas encoder-decoder models have not been successfully adopted for image-text retrieval tasks.
  + **Data perspective**: the noisy web text is suboptimal for vision-language learning.
+ Two contributions:
  + **Multimodal mixture of Encoder-Decoder (MED)**: a new model architecture for effective multi-task pre-training and flexible transfer learning. An MED can operate either as a unimodal encoder, or an image-grounded text encoder, or an image-grounded text decoder. The model is jointly pre-trained with three vision-language objectives: imagetext contrastive learning, image-text matching, and imageconditioned language modeling.
  + **Captioning and Filtering (CapFilt)**: a new dataset boostrapping method for learning from noisy image-text pairs. We finetune a pre-trained MED into two modules: a captioner to produce synthetic captions given web images, and a filter to remove noisy captions from both the original web texts and the synthetic texts.

## Medel Architecture
+ Unimodal encoder
  + image encoder
  + text encoder is the same as BERT
+ Image-grounded text encoder
  + add CA between SA and FFN
  + A task-specific [Encode] token is appended to the text
+ Image-grounded text decoder
  + causal self-attention layers in replace of bidirectional self-attention in encoder

## Pre-training Objectives
+ **Image-Text Contrastive Loss (ITC)**
  + align the feature space of the visual transformer and the text transformer
  + where a momentum encoder is introduced to produce features, and soft labels are created from the momentum encoder as training targets to account for the potential positives in the negative pairs.
+ **Image-Text Matching Loss (ITM)**
  + where the model uses an ITM head (a linear layer) to predict whether an image-text pair is positive (matched) or negative (unmatched) given their multimodal feature. 
  + adopt the hard negative mining strategy, where negatives pairs with higher contrastive similarity in a batch are more likely to be selected to compute the loss
+ **Language Modeling Loss (LM)**
  + It optimizes a cross entropy loss which trains the model to maximize the likelihood of the text in an autoregressive manner.
  + We apply a label smoothing of 0.1 when computing the loss    

**Totally,**
1) the text encoder and text decoder share all parameters except for the SA layers(CA, FFN), the reason is that the differences between the encoding and decoding tasks are best captured by the SA layers.
2) the encoder employs **bi-directional self-attention** to build representations
for the current input tokens, while the decoder employs causal self-attention to predict next tokens.

## CapFilt
Both the captioner and the filter are initialized from the same pre-trained MED
model, and finetuned individually on the COCO dataset.
- The Captioner is an **image-grounded text decoder** generate captions given web images
- The filter is an **image-grounded text encoder**. It is finetuned with the ITC and ITM objectives to learn whether a text matches an image

