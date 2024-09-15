
# Time Series Forecasting For Business Sales

This project explores the process of predicting future store sales using time series forecasting techniques. It involves a comprehensive approach including data preprocessing, feature engineering, and the development of a multiple regression model.

## Key Highlights

### Transformation of Sales and Onpromotion Features

A column-wise transformation of the sales and onpromotion features is performed for each product family, ensuring that training can be done in parallel using a multiple regression model. This eliminates the need to handle units across different product families. Transformations are applied at the store level to capture unique store characteristics, such as size and product offerings. 

Additionally, right-skewness in the data is addressed using a **PowerTransformer**, which enhances model performance and reduces training time. While experiments with differencing and lagged features were conducted, these did not significantly improve the model's predictions.

### Feature Engineering

The feature engineering process is central to the model's performance. Key features include:

- **Seasonality**: Fourier and date-based features are incorporated to capture weekly, monthly, and yearly seasonality patterns.
- **Oil Price**: Missing values in the daily oil price data are imputed.
- **Store Size**: A feature representing store size is derived from the number of transactions.
- **Geolocation**: City and state population data are extracted from external Census data and integrated as features.
- **Outlier Detection**: Stores that were closed or exhibited abnormal behavior are identified and managed.
- **Feature Selection**: Techniques like **Mutual Information Index** and **Spearman’s rho** are used to select features most relevant to sales performance across multiple product families.

### PyTorch Multiple Regression Model

The forecasting model is built using **PyTorch Lightning** and incorporates several advanced techniques to improve performance:

- **Store Embeddings**: Embeddings are derived for each store based on categorical features such as store number, store type, geolocation, and size.
- **Autoregressive Dataset**: The autoregressive dataset structure allows the model to predict one day of sales across all stores and product families. These predictions are then fed back into the model for subsequent prediction steps. During training, actual sales values are used.
- **Model Architecture**: The model consists of an embedding layer for store-related features, an autoregressive **LSTM** layer that processes several weeks of historical sales data, and a final dense layer that combines the outputs from the embeddings and LSTM layers.
  
### Hyperparameter Tuning and Model Evaluation

Hyperparameters were optimized manually using the final month of the training dataset as validation. The model’s performance was evaluated using a self-implemented **RMSLE (Root Mean Squared Logarithmic Error)** loss function. The final model was trained on the complete dataset, and predictions were visually inspected to ensure their quality.

### Learned Embeddings

The trained model provides learned vector embeddings for each store, which can be used as features in future machine learning tasks. These embeddings complement the existing store cluster feature and help enhance the model’s ability to generalize to different store environments.

---

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

To run this project, follow these steps:

1. Clone or download the repository.
2. Install dependencies with the above command.
3. Launch Jupyter Notebook:
4. Open the relevant notebook (`Time_Series_Business_Sales.ipynb`) and run the cells to preprocess data, train the model, and evaluate results.

## Results

The project produces a time series forecasting model capable of accurately predicting sales across multiple stores and product families, with extensive use of feature engineering, PyTorch embeddings, and autoregressive modeling.
