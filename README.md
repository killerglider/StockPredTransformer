# StockPredTransformer

## Overview
StockPredTransformer is a deep learning model designed for stock price prediction using a Transformer-inspired architecture. It utilizes historical stock prices to predict future values, leveraging LSTMs for sequential data processing.

## Features
- Fetches stock data from Yahoo Finance using `yfinance`.
- Preprocesses data with Min-Max scaling.
- Uses a Transformer-inspired deep learning model for stock price prediction.
- Evaluates model performance using MSE, RMSE, R2 Score, and MAE.
- Visualizes predictions against actual stock prices.

## Installation
Ensure you have Python installed along with the required dependencies:

```sh
pip install numpy pandas matplotlib tensorflow scikit-learn yfinance
```

## Usage
Run the main script to train and evaluate the model:

```sh
python transformer_stock_prediction.py
```

### Sample Output Graph
![image](https://github.com/user-attachments/assets/e22e0e24-2fc4-40b2-9493-6d25acef64b6)


## Model Architecture
The model consists of:
- 3 LSTM layers (128, 64, and 32 neurons respectively).
- Dense layers for final predictions.
- Adam optimizer with Mean Squared Error (MSE) loss function.

## Performance Metrics
The model outputs the following performance metrics:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R2 Score**
- **Mean Absolute Error (MAE)**
- **Prediction Accuracy**

## Results
After training for 50 epochs, the model produces predictions that are visualized in a plot comparing actual vs. predicted prices.

## Contributions
Feel free to contribute by opening an issue or submitting a pull request.

## License
This project is licensed under the MIT License.

---

### Future Improvements
- Implement attention mechanisms for a true Transformer model.
- Support multiple stock predictions simultaneously.
- Optimize hyperparameters for better performance.

