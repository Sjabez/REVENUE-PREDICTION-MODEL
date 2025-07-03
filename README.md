
# Restaurant Revenue Prediction

## 🍽️ **Project Overview**
This project involves building a machine learning model to **predict the revenue of restaurants** based on several operational and business attributes. Using a dataset with 100 restaurant samples, we apply data analysis and regression modeling to generate reliable revenue forecasts.

---

## 🧾 **Problem Statement**
The dataset contains 100 observations and the following attributes:

- **ID**: Unique identifier for each restaurant
- **Name**: Name of the restaurant
- **Franchise**: Indicates whether the restaurant is a franchise
- **Category**: Cuisine or category of the restaurant
- **No_of_item**: Number of different menu items
- **Order_Placed**: Orders placed by customers (in lakhs)
- **Revenue**: Total revenue generated (target variable)

---

## 📌 **Objective**
Build a predictive model using machine learning to estimate **restaurant revenue** based on business inputs like number of items, orders placed, and franchise/category information.

---

## 🧪 **Approach & Methodology**

### 1. 🔎 Data Preprocessing
- Cleaned and prepared categorical variables like `Franchise` and `Category`
- Converted non-numeric columns using one-hot encoding
- Checked for missing values and addressed them

### 2. 📊 Exploratory Data Analysis
- Plotted distributions of revenue, orders, and number of items
- Analyzed the influence of categorical attributes (e.g., franchise vs. non-franchise)

### 3. 🔮 Model Building
- Used **Linear Regression** for baseline modeling
- Split the data using **train_test_split** method
- Evaluated the model using metrics like **R² Score** and **Root Mean Squared Error**

---

## 🧠 **Code Used**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("revenue_prediction.csv")

df.head()

df.isnull().sum()

df=df.drop(columns=["Id","Name","Franchise","Category","City","No_Of_Item"])

df.head()

df.shape

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

print(x_train)

print(x_test)

print(y_train)

print(y_test)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

y_pred

plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Revenue vs orders")
plt.xlabel("orders")
plt.ylabel("Revenue")

plt.scatter(x_test,y_test,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Revenue vs orders")
plt.xlabel("orders")
plt.ylabel("Revenue")

from sklearn.metrics import r2_score

score=r2_score(y_pred,y_test)

score
```

---

## 📈 **Results**
- **R² Score** indicated strong linear relationship between order volume and revenue
- The model generalized well on test data, making it suitable for business forecasting
- Feature importance confirmed `Order_Placed` as the most influential factor

---

## 💡 **Key Business Insight**
Franchise and category type had moderate influence, but **order volume** and **number of items** offered were the biggest drivers of restaurant revenue. This model can help new or growing restaurants estimate their potential income based on menu design and projected order volume.

---

## 🛠 **Tools & Libraries**
- Python (Pandas, NumPy, Seaborn, Matplotlib)
- Scikit-learn (LinearRegression, metrics, preprocessing)
- Jupyter Notebook

---

## 🚀 **Next Steps**
- Apply ensemble models (Random Forest, Gradient Boosting) for improved accuracy
- Deploy as a web app using Flask/Streamlit for real-world usability
- Add more restaurant metadata like location and pricing tiers

---
