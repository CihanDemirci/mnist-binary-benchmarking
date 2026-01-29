# Binary Image Classification: Linear vs. Convolutional Models

## Project Overview
This project benchmarks two different machine learning approaches to solve a classic Computer Vision problem: distinguishing between handwritten digits **2** and **3** from the MNIST dataset. 

The goal was to demonstrate the transition from a statistical baseline (**Logistic Regression**) to a spatial-aware Deep Learning model (**CNN**), comparing their accuracy and learning efficiency.

## Tech Stack
- **Language:** Python 3.12
- **Deep Learning:** PyTorch
- **Machine Learning:** Scikit-Learn
- **Data Manipulation:** NumPy, Pandas
- **Visualization:** Matplotlib

## Key Findings
- **Logistic Regression Baseline:** Achieved a steady accuracy of ~97% but struggled with rotated or highly stylized digits.
- **CNN Performance:** Surpassed the baseline within 3 epochs, eventually reaching ~98-99% accuracy by capturing spatial hierarchies in pixel data.

## Project Structure
- `data_utils.py`: Modular data pipeline (fetching, normalization, and tensor reshaping).
- `log_reg_model.py`: Implementation of the Scikit-Learn baseline.
- `cnn_model.py`: PyTorch implementation of the Convolutional Neural Network.
- `compare_models.py`: Main execution script that benchmarks both models and generates performance plots.

## Conclusion
Although the CNN model achieved a higher accuracy score than the logistic regression model, the latter one already had a pretty good score (~97%). Since the CNN model requires much more computational power, it is best practice to stick with the logistic regression for such simple decision tasks, i.e. the CNN is a bit of overkill in this case. 