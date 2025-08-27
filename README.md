# text_classification_using-_bert
# 📝 Author Classification with BERT & MLP

This project implements an **author classification** model using **BERT embeddings** and a simple **MLP classifier**. The goal is to predict the author of a given text (EAP, HPL, MWS).  

## 📌 Project Overview
- **Dataset:** Texts from multiple authors (`train.csv` and `test.csv`)  
- **Goal:** Predict the author of each text  
- **Model:** BERT embeddings + MLP classifier  
- **Output:** CSV file with author predictions  

## ⚙️ Pipeline Steps
1. **Text Preprocessing**
   - Remove numbers and unnecessary punctuation  
   - Convert text to lowercase  

2. **Author Encoding**
   - One-hot encode the `author` column  
   - Convert to binary values for each author (EAP, HPL, MWS)  

3. **Feature Extraction**
   - Use **BERT base uncased** to obtain embeddings for each text  

4. **Dataset Construction**
   - Build `torch.utils.data.Dataset` for training and testing  

5. **MLP Model**
   - Input: 768-dimensional BERT embeddings  
   - Hidden layer + ReLU + Dropout  
   - Output: 3 dimensions (one for each author)  

6. **Training & Validation**
   - Use **5-Fold Cross-Validation**  
   - Loss: `BCEWithLogitsLoss`  
   - LogLoss reported for each fold  

7. **Prediction & Submission**
   - Generate predictions on the test set  
   - Save results to `submission1.csv`  

## 📌 Notes
- GPU is used for faster training  
- Random seed is fixed for reproducibility  
- Batch size and number of epochs are configurable  

## 🔗 Files
- `train.csv` : Training dataset  
- `test.csv` : Test dataset  
- `submission1.csv` : Model predictions
