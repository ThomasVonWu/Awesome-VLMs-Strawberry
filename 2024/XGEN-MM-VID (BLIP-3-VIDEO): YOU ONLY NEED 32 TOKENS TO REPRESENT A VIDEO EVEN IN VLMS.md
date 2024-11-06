# XGEN-MM-VID (BLIP-3-VIDEO): YOU ONLY NEED 32 TOKENS TO REPRESENT A VIDEO EVEN IN VLMS

### 🔥 **Article**  
- https://arxiv.org/abs/2410.16267
### 🔥 **Blog**  
- https://www.salesforceairesearch.com/opensource/xGen-MM-Vid/index.html

## Introduction
We introduce xGen-MMVid (BLIP-3-Video 4B), which is an efficient compact vision-language model with an `explicit temporal encode`r, designed particularly for videos,  the model can abstract each video into much fewer visual tokens (e.g., 16).

## Background
+ Models like `Video-ChatGPT` (Maaz et al., 2024) and `PLLaVA` (Xu et al., 2024a) rely on a simple `spatial/temporal pooling on top of image frame-level tokens` to represent the entire video.
+ Some models `use additional convolutional layers (or Transformer layers)` over frames to reduce their representation size (e.g., `Video-LLaMA` (Zhang et al., 2023), Kangaroo (Liu et al., 2024)).
+ Approaches that simply collect all the visual tokens from all the frames (e.g., `MiniGPT4-video` (Ataallah et al., 2024), `LLaVA-NeXT` (Li et al., 2024b), `Tarsier `Wang et al., 2024a) and `LLaVA-OneVision` (Wang et al., 2024a)) also have been very popular recently.

## Model Structure
The model architecture is composed of the following four components: 
- (1) The vision encoder (ViT) taking each frame input 
  - `SigLIP`, `Perceiver-Resampler` is then applied to map such visual tokens into N to 128 visual tokens per frame. The model takes uniformly sampled 8 frames per video. As a result, in our model, ViT first maps a video into 8 ∗ 729 visual tokens, which is then mapped to 8 ∗ 128 visual tokens using Perceiver-Resampler, and then to 16 ∼ 128 video tokens using the temporal encoder.
- (2) The frame-level tokenizer to reduce the number of tokens
  - which is then mapped to 8 ∗ 128 visual tokens using Perceiver-Resampler, and then to 16 ∼ 128 video tokens using the temporal encoder.
- (3) The temporal encoder to build video-level token representations
  - *TEMPORAL ENCODERS*  is a function of tokens, taking N · T tokens as an input and returning M tokens as an output.
    -  Temporal pooling : summating per-frame tokens over time
    -  Transformer-based :  modeling the entire token sequence and selecting the last m tokens similar to `Mirasol3B` 
    -  Spatio-temporal attentional pooling : In our model, we use TokenLearner (Ryoo et al., 2021), making it explicitly serve as our space-time aware temporal encoder.
    -  Sequential Model: We also deploy `Token Turing Machines (TTM)` (Ryoo et al., 2023) as a temporal encoder, which is a sequential model capable of taking any number of frames to generate a video-level token representation, we also further extend TTM by adding time-stamped positional encodings to embded the frame index of each token in the latent space.
- (4) The autoregressive LLM generating output text captions based on such video tokens and text prompt tokens
  - We use `Phi-3` (Abdin et al., 2024) as our LLM backbone taking such video tokens in addition to the text prompt tokens. This enables the model to **take text+video** as an input and generate **text sentences** as an output.


## Training Recipe
- (1) Image caption pretraining: First, we directly use the pretrained weights from BLIP-3 (Xue et al., 2024). BLIP-3 is for images and it does not contain weights for the temporal encoder, so we randomly initialize those weights.
- (2) Video caption pretraining, the model is then trained on LLaVA-Hound-DPO’s video caption data (Zhang et al., 2024b), featuring over 900k video captions. Instead of directly using the text captions provided in `LLaVA-Hound-DPO`, we used `GPT-4` to rephrase such text captions so that they become more GPT-style captions.
- (3) Video instruction tuning, we tuned the model using a mix of video question-answering datasets, including `VideoChatGPT’s 99k-sample` video instruction tuning data (Maaz et al., 2024), along with the training splits of the `MSVD-QA` (Xu et al., 2017), `MSRVTT-QA` (Xu et al., 2017), `ActivityNet-QA` (Yu et al., 2019), `TGIF-QA` (Jang et al., 2017), and `NExT-QA` (Xiao et al., 2021) datasets, which contain 30k, 149k, 32k, 71k, and 34k samples, respectively.

We trained our model with 8 × H100 GPUs. For the video caption pretraining, we use the batch size of 16 per GPU, 500 warmup steps, and the learning rate of 2e-5 with the cosine decay. We trained the model for 1 epoch. The video QA sft (i.e., instruction tuning) was done with the batch size of 4 per gpu, 500 warmup steps, and the learning rate of 1e-5 with the cosine decay. We trained the model for 1 epoch in this case as well. **The entire training (combining both video pretraining and the sft) takes around 12 hours**, confirming the efficiency of our model.

## Conclusion
We introduce BLIP-3-Video, which is an efficient, compact vision-language model for videos with 4B parameters. BLIP-3-Video incorporates a temporal encoder in its architecture, which allows the model to abstract the entire video with as few as 16 or 32 tokens. In contrast to many state-of-the-art video VLMs taking advantage of thousands of visual tokens to represent a video (e.g., 4608), BLIP-3-Video shows a competitive performance while utilizing much fewer visual tokens (e.g., 32).