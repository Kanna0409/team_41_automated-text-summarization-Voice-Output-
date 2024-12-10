# **team_41_automated-text-summarization**

## **Project Overview**
The goal of this project is to build an **automated text summarizer** that processes human speech into input texts and generates corresponding summaries. After deploying the model, it processes text from speech and creates concise summaries.

The summarizer is based on a **Transformer architecture** consisting of custom multi-layered encoders and decoders. These components:
1. Work on the output of the preceding layers.
2. Use **self-attention** and **cross-attention** mechanisms to process the data.

---

## **Workflow Summary**

1. **Dataset Loading and Preprocessing**:
   - **Input Format**: A `.csv` file containing columns `id`, `dialogues`, `summaries`, and `topics`.
   - Columns `id` and `topics` were excluded during preprocessing.
   - Text length analysis:
     - 97% of dialogue texts have a length ≤ 1584.
     - 97% of summary texts have a length ≤ 283.
   - Only texts meeting these length criteria were processed further.
   - A new dataset was created with dialogue texts and their corresponding summaries.

2. **Batch Loading**:
   - Data was loaded into batches of size 2 using `DataLoader`.

3. **Tokenization**:
   - A **T5Tokenizer** was used to tokenize texts:
     - Adds start and end tokens.
     - Pads the sequences to maintain consistent lengths.
   - Maximum token lengths:
     - Dialogue: 573 tokens.
     - Summary: 129 tokens.

4. **Padding and Masks**:
   - Padding ensures constant input sizes for the encoder and decoder.
   - Padding masks:
     - **Encoder Mask**: Ignores padding tokens during attention computation.
     - **Decoder Masks**:
       - Suppresses padded tokens.
       - Masks future tokens to prevent the model from seeing ahead.

5. **Tensor Creation**:
   - Input (`input_ids`) and output (`output_ids`) tensors were created for tokenized texts.
   - All batches were concatenated into large tensors:
     - `tensor_tokenized_inputs` (size = 12085573).
     - `tensor_tokenized_outputs` (size = 12085150).

---

## **Transformer Architecture**

### **Encoder**
1. **Multi-Head Attention**:
   - Calculates self-attention using the formula:  
     \[
     \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
     \]
   - `Q`, `K`, and `V` matrices are derived from input embeddings.
   - Outputs from all heads are concatenated and passed through a linear layer.

2. **Dropout and Residual Connections**:
   - Drops some neurons (probability = 0.1) for regularization.
   - Residual connections add the original input to the output of the attention layer to address vanishing gradients.

3. **Layer Normalization**:
   - Stabilizes training by normalizing inputs to subsequent layers.

4. **Feedforward Layers**:
   - Sequence of linear transformations, ReLU activations, and dropout.
   - Residual connections and normalization are applied again.

5. **Layer Stacking**:
   - Multiple encoder layers are stacked.
   - The output of the last encoder layer is sent to the decoder.

---

### **Decoder**
1. **Masked Self-Attention**:
   - Prevents the decoder from "seeing" future tokens by masking.
   - Ensures the model predicts tokens sequentially.

2. **Cross-Attention**:
   - Combines the decoder’s queries with the encoder’s outputs to refine predictions.

3. **Layer Configuration**:
   - Similar to the encoder, with additional cross-attention layers.

---

## **Embedding and Positional Encoding**
- Texts are converted into **word embeddings** (dense vector representations).
- Positional encodings (using sine and cosine functions) are added to embeddings to retain word order.

---

## **Training**
1. **Loss Function**:
   - Cross-entropy loss is used to compute the difference between predicted and actual summaries.

2. **Backpropagation**:
   - Loss gradients are propagated back to update model parameters.

3. **Batch Processing**:
   - Training occurs in batches of size 2, with parameters updated after each batch.

---

## **Model Results**
- **Encoder-Decoder Synchronization**:
   - Multi-layered architecture ensures efficient processing of long texts and accurate summary generation.
- **Loss Convergence**:
   - The loss reduced significantly during training, demonstrating successful learning.
   - Final Loss for model trained with smaller dataset = 0.0031
   - Final loss for model trained with Larger dataset = 0.0029

---

## **Future Enhancements**
- Real-time text-to-speech integration.
- Support for multiple languages.
- Fine-tuning to improve summary coherence and fluency.

This project showcases the power of Transformers in natural language understanding and generation, providing an accessible solution for summarizing speech-based texts.
Hi
