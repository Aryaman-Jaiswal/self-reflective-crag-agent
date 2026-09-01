# Attention Is All You Need: Architecture & Mechanics

## 1. Executive Summary
The Transformer is the first sequence transduction model based entirely on attention, replacing recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

## 2. Model Architecture
The Transformer follows an encoder-decoder structure:
- **Encoder**: Composed of a stack of $N = 6$ identical layers. Each layer has two sub-layers:
  1. Multi-head self-attention mechanism
  2. Position-wise fully connected feed-forward network
  Residual connections are employed around each of the two sub-layers, followed by layer normalization:
  $$\text{LayerNorm}(x + \text{Sublayer}(x))$$
  To facilitate these residual connections, all sub-layers in the model produce outputs of dimension $d_{\text{model}} = 512$.

- **Decoder**: Also composed of a stack of $N = 6$ identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Masked self-attention is applied in the decoder to prevent positions from attending to subsequent positions.

## 3. Scaled Dot-Product Attention
The attention mechanism computes a mapping from a query and a set of key-value pairs to an output:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
Where:
- $Q$ is the matrix of queries of dimension $d_k$
- $K$ is the matrix of keys of dimension $d_k$
- $V$ is the matrix of values of dimension $d_v$

The scaling factor $\frac{1}{\sqrt{d_k}}$ is critical: for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients. Dividing by $\sqrt{d_k}$ counteracts this effect.

## 4. Multi-Head Attention
Instead of performing a single attention function with $d_{\text{model}}$-dimensional queries, keys, and values, Multi-Head Attention linearly projects queries, keys, and values $h = 8$ times with different learned linear projections to $d_k = 64$ and $d_v = 64$:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$
Where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$.

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

## 5. Positional Encoding
Since the model contains no recurrence and no convolution, positional encodings are injected into the input embeddings to inject positional order:
$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$
