# BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

### 🔥 **Article**  
https://arxiv.org/abs/2301.12597

### 🔥 **Code**  
https://github.com/salesforce/LAVIS/tree/main/projects/blip2

## Abstract:
BLIP-2 bridges the modality gap with a lightweight Querying Transformer, which is pretrained in two stages:
+ Bootstraps `vision-language representation learning` from a frozen image encoder.
  + we perform vision-language representation learning which enforces the Q-Former to learn visual representation most relevant to the text.
+ Bootstraps vision-to-language generative learning from a frozen language model.
  + we perform `vision-to-language generative learning` by connecting the output of the Q-Former to a frozen LLM, and trains the Q-Former such that its output visual representation can be interpreted by the LLM.


## Method
### Model Structure
+ The queries interact with each other through self-attention layers, and interact with frozen image features through cross-attention layers (inserted every other transformer block). The queries can additionally interact with the text through the same self-attention layers. Depending on the pre-training task, we apply different self-attention masks to control query-text interaction. We initialize QFormer with the pre-trained weights of BERTbase, whereas the cross-attention layers are randomly initialized. In total, Q-Former contains 188M parameters.
+ The size of Z (32 × 768) is much smaller than the size of frozen image features (e.g. 257 × 1024 for ViT-L/14). This bottleneck architecture works together with our pre-training objectives into forcing the queries to extract visual information that is most relevant to the text.

### Bootstrap Vision-Language Representation Learning from a Frozen Image Encoder
+ Image-Text Contrastive Learning (ITC)
+ Image-grounded Text Generation (ITG)
+ Image-Text Matching (ITM)

### Bootstrap Vision-to-Language Generative Learning from a Frozen LLM
+ We use a fully-connected (FC) layer to linearly project the output query embeddings Z into the same dimension as the text embedding of the LLM.
+ We experiment with two types of LLMs: decoder-based LLMs and encoder-decoder-based LLMs. For decoderbased LLMs, we pre-train with the language modeling loss, where the frozen LLM is tasked to generate the text conditioned on the visual representation from Q-Former. For encoder-decoder-based LLMs, we pre-train with the prefix language modeling loss, where we split a text into two parts. The prefix text is concatenated with the visual representation as input to the LLM’s encoder. The suffix text is used as the generation target for the LLM’s decoder.

### Pre-training data
+ 129M images in total, including COCO, Visual Genome, CC12M, CC3M, SBU.
+ 115M images from the LAION400M dataset.

### Model Pre-training
+ `ViT-L/14` from CLIP (Radford et al., 2021) and (2) `ViT-g/14` from EVA-CLIP (Fang
et al., 2022). We remove the last layer of the ViT and uses the second last layer’s output features
+ we explore the unsupervised-trained `OPT` model family for decoder-based LLMs, and
the `instruction-trained FlanT5` model family for encoder-decoder-based LLMs.

### Pre-training setting
+ We use a batch size of 2320/1680 for ViT-L/ViT-g in the first stage and
a batch size of 1920/1520 for OPT/FlanT5 in the second stage. 
+ We convert the frozen ViTs’ and LLMs’ parameters into FP16, except for FlanT5 where we use BFloat16.
+ Using a single 16-A100(40G) machine, our largest model with ViT-g and FlanT5-XXL requires less than 6 days for the first stage and less than 3 days for the second stage