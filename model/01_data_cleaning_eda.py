import pandas as pd

df = pd.read_excel(r"C:\Users\anujp\OneDrive\Desktop\coustomer churn predicition model\model\Telco_customer_churn.xlsx")



cols_to_drop = ["CustomerID", "Count", "Country", "State", "City",
                "Zip Code", "Lat Long", "Latitude", "Longitude",
                "Churn Label", "Churn Score", "CLTV", "Churn Reason"]

df = df.drop(columns=cols_to_drop)

df = df.rename(columns={"Churn Value": "Churn"})

# Step 3a: force conversion to numeric. Any value that CAN'T become a number
# (like an empty string) becomes NaN automatically because of errors="coerce"
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

# Step 3b: check how many are now missing
#print(df["Total Charges"].isnull().sum())   # -> 11

# Step 3c: fill those 11 with 0 (logical: brand-new customers, tenure=0, no bill yet)
df["Total Charges"] = df["Total Charges"].fillna(0.0)

#print(df["Churn"].value_counts())            # raw counts
#print(df["Churn"].value_counts(normalize=True) * 100)   # as percentages

categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

for col in categorical_cols:
    print(col, df[col].unique())
    
print(df["Total Charges"].isnull().sum())   # should print 0 (after fillna)
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True) * 100)  

df.to_csv(r"C:\Users\anujp\OneDrive\Desktop\coustomer churn predicition model\model\telco_churn_clean.csv", index=False)
print("Saved cleaned file. Final shape:", df.shape)