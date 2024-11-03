# CLIP: Learning Transferable Visual Models From Natural Language Supervision

### 🔥 **Article**  
https://arxiv.org/pdf/2103.00020  

### 🔥 **Blog**  
https://openai.com/index/clip/  

### 🔥 **Code**  
https://github.com/openai/CLIP  

## Introduction:
In this work, we close
this gap and study the behaviors of image classifiers trained with natural language supervision at large scale. Enabled by the large amounts of publicly available data of this form on the internet, we create a new dataset of 400 million (image, text) pairs and demonstrate that a simplified version of ConVIRT trained from scratch, which we call CLIP, for Contrastive Language-Image Pre-training, is an efficient method of learning from natural language supervision.

## Contributions
- We find that CLIP, similar to the GPT family, learns to perform a wide set of tasks during pre-training including OCR, geo-localization, action recognition, and many others .
- We also confirm these findings with linear-probe representation learning analysis and show that CLIP outperforms the best publicly available ImageNet model while also being more computationally efficient.
- We additionally find that zero-shot CLIP models are much more robust than equivalent accuracy supervised ImageNet models which
suggests that zero-shot evaluation of task-agnostic models is much more representative of a model’s capability.

## Approach
**1. Natural Language Supervision**  

**2. Creating a Sufficiently Large Dataset**  

**3. Selecting an Efficient Pre-Training Method**  
   - Our initial approach, similar to VirTex, jointly trained an image CNN and text transformer from scratch to predict the caption of an image.
       - We explored training a system to solve the potentially easier proxy task of predicting only which text as a whole is paired with which image and not the exact words of that text.
       - CLIP learns a multi-modal embedding space by jointly training an image encoder and text encoder to maximize the cosine similarity of the image and text embeddings of the N real pairs in the batch while minimizing the cosine similarity of the embeddings of the N2 − N incorrect pairings.
       - We train CLIP from scratch without initializing the image encoder with ImageNet weights or the text encoder with pre-trained weights.
       - We instead use only a linear projection to map from each encoder’s representation to the multi-modal embedding space.
       - A random square crop from resized images is the only data augmentation used during training.  

**4. Choosing and Scaling a Model**  
   - Two different architectures for the image encoder:
       - ***ResNet-50***
           - ResNetD + antialiased rect-2 blur pooling + global average pooling layer with attention pooling.
           - For the ResNet image encoders, we adapt the approach of Tan & Le (2019), which found that allocating additional compute across width, depth, and resolution outperforms allocating it to only one dimension of the model.
           - While Tan & Le (2019) tune the ratio of compute allocated to each dimension for their EfficientNet architecture, we use **a simple baseline of allocating additional compute** equally to increasing the width, depth, and resolution of the model.
       - ***Vision Transformer***
           - Adding an additional layer normalization to the combined patch and position embeddings before the transformer and using a slightly different initialization scheme.
   - Text encoder with Transformer:
       - As a base size, we use a 63M-parameter 12-layer 512-wide model with 8 attention heads.
       - The transformer operates on a lower-cased byte pair encoding (BPE) representation of the text with a vocab size of 49,152.
       - For the text encoder, we only **scale the width** of the model to be proportional to the calculated increase in width of the ResNet and do not scale the depth at all.

## Backup
+ blog: https://zhuanlan.zhihu.com/p/493489688