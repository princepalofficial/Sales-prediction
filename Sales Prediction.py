import numpy as np
import pandas as pd
# 1. Create Sales Dataset
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    "Advertising": np.random.uniform(1000, 5000, n),
    "Price": np.random.uniform(100, 1000, n),
    "Discount": np.random.uniform(0, 30, n),
    "Customer": np.random.uniform(50,1000,n)
})

# 2. Generate Sales
noise = np.random.normal(0, 5000,n)
df["Sales"] = (
    2.5 * df["Advertising"] - 20 * df["Price"] + 500 * df["Discount"] + 100 * df["Customer"] + noise
)
df["Sales"] = df["Sales"].clip(0)

# 3. Dataset Information
print("******Dataset*****")
print(df.head())
print("\nShape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Sales Analysis
print("\n ******Sales Analysis *******")
print("Average Sales:",round(df["Sales"].mean(),2))
print("Maximum Sales:", round(df["Sales"].max(),2))
print("Minimum Sales:", round(df["Sales"].min(),2))

# 5. New product prediction
new_product = pd.DataFrame({
    "Adertising": [20000],
    "Price": [500],
    "Discount": [10],
    "Customers": [500]
})

predicted_sales = (
    2.5 * new_product["Adertising"] - 20 * new_product["Price"] + 500 * new_product["Discount"] 
    + 100 * new_product["Customers"]
)

print("\n ****new Product *****")
print(new_product)
print("\nPredicted Sales:", round(predicted_sales.iloc[0],2))
