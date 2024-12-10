# Final Project Report

## Abstract and Introduction
This notebook implements a pipeline for natural language processing (NLP) and data analysis using popular Python libraries such as `numpy`, `pandas`, `torch`, and `spacy`. The primary goal is to preprocess textual data, utilize GPU acceleration where available, and train our Basic Custom Transformer model that was built from scratch on datasets for text summarization tasks. Two datasets are employed to evaluate the pipeline's performance on different scales, demonstrating its effectiveness in handling both small and large datasets.

---

## Prior Related Work
This project leverages well-established libraries and methodologies:
- **SpaCy**: A widely-used NLP library for tokenization and text cleaning.
- **PyTorch**: Integrated for potential GPU-accelerated machine learning workflows.
- **Pandas**: Utilized for efficient data manipulation and preprocessing.

The use of these tools aligns with best practices in natural language processing and machine learning.

---

## Dataset
Two datasets were used to train and evaluate the summarization model:
1. **DialogSum Dataset**:
   - **Source**: [DialogSum on Kaggle](https://www.kaggle.com/datasets/marawanxmamdouh/dialogsum)
   - **Description**: Contains human-written summaries of dialogues for text summarization tasks.
2. **CNN/Daily Mail Text Summarization Dataset**:
   - **Source**: [CNN/Daily Mail on Kaggle](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)
   - **Description**: Includes news articles and their summaries for summarization tasks.

The datasets vary in size, allowing for evaluation of the pipeline's performance on small and large datasets.

---

## Methodology / Model
1. **Preprocessing**:
   - **Function**: `preprocess_text(text)`
   - **Logic**:
     - Utilizes SpaCy to parse input text.
     - Removes punctuation and whitespace.
     - Joins cleaned tokens into a single string.
   - **Purpose**: Ensures text is clean and prepared for summarization tasks.
2. **Device Configuration**:
   - Detects GPU availability using PyTorch.
   - Defaults to CPU for computations if GPU is unavailable.
3. **Model**:
   - A text summarization model( Basic Transformer) that was built from scratch was trained separately on the small(dialogue summarization) and large datasets(CNN dailymail news summarization).
   - The training process minimized the loss function to optimize the summaries generated.



## Experiments
- **Preprocessing Validation**:
  - Text data from both datasets was processed through the `preprocess_text` function to verify proper tokenization and cleaning.
- **Device Verification**:
  - Confirms the runtime environment (CUDA or CPU) to ensure hardware optimization.
- **Model Training**:
  - Two same custom model(Transformer) were trained seperately using the smaller and larger datasets to compare results specifically losses.



## Results
### Smaller Dataset Output:
**Summary**:  
*Person1 invites Anna to come to a get together. Anna agrees after she knows Jack won't come because they are on bad terms.*  
This output(cleaned_text) was of the last dialogue that was in the tensor_tokenized_inputs(training dataset) from the training loop.From the training loop the predicted_text of the last dialogue of tensor_tokenized_inputs of last epoch was taken out and cleaned and saved to the variable cleaned_text. This output can be found in **`Small.ipynb`** cell number **73**

### Larger Dataset Output:
**Summary**:  
*Phillipe Mexes dismissed for grabbing the throat of Stefano Mauri. AC Milan defender also appeared to clutch the neck of Lazio player Lorik Cana. Lazio beat AC Milan 3–1 at the Stadio Olimpico on Saturday in Serie A.*  
This output(cleaned_text) was of the last article that was in the tensor_tokenized_inputs(training_dataset) from the training loop. From the training loop the predicted_text of the last dialogue of tensor_tokenized_inputs of last epoch was taken out and cleaned and saved to the variable cleaned_text. This output can be found in **`codebig.ipynb`** cell number **190**.

### Final Loss:
- **Model trained on smaller dataset**: `0.0031` (Small.ipynb)
- **Model trained on larger dataset**: `0.0029` (codebig.ipynb)



## Analysis & Conclusion
The pipeline successfully handled both small and large datasets, showcasing its flexibility and robustness. Key findings include:
1. **Smaller Dataset Performance**:
   - Generated concise summaries with good context but limited diversity due to smaller data size.
   - Final loss: `0.0031`
2. **Larger Dataset Performance**:
   - Produced more detailed summaries and improved overall performance.
   - Final loss: `0.0029`

NO EVALUATION METRICS/TECHNIQUES WERE USED. THE MODEL WAS ONLY TRAINED AND SUMMARY CORRECTNESS CHECKED ON THE PREDICTED_TEXT OF THE LAST INPUT OF THE LAST EPOCH FOR THE TWO DATASETS:

FOR 1ST DATASET(Dialogue summarization dataset) IN "Small.ipynb" THE PREDICTED_TEXT FOR LAST INPUT IN TRAINING DATASET WAS : Person1 invites Anna to come to a get together. Anna agrees after she knows Jack won't come because they are on bad terms. This output can be found in **`Small.ipynb`** cell number **73**.

SIMILARLY FOR THE 2ND DATASET(CNN dailymail news dataset) IN "codebig.ipynb" THE PREDICTED_TEXT FOR LAST INPUT IN TRAINING DATASET WAS : Phillipe Mexes dismissed for grabbing the throat of Stefano Mauri AC Milan defender also appeared to clutch neck of Lazio player Lorik Cana Lazio beat AC Milan 3 1 at the Stadio Olimpico on Saturday in Serie A. This output can be found in **`codebig.ipynb`** cell number **190**.

### Improvements for Future Work:
1. Enhance preprocessing with advanced NLP techniques like lemmatization and stop-word removal.
2. Perform exploratory data analysis (EDA) for deeper insights into dataset structure.
3. Extend experiments to include more datasets and compare models trained on different architectures.
4. Could build a "Generate function" for the model to generate the summary for the text of user input.
This pipeline provides a strong foundation for text summarization tasks, effectively balancing performance and scalability.
