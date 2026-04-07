import pandas as pd
import os

pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)

base = r"d:\school\uni\sem 6\llm\week8\dataset\Indonesian Skincare Sample Dataset"

# 1. ingredients_category
print("=" * 80)
print("1. ingredients_category.xlsx")
print("=" * 80)
df1 = pd.read_excel(os.path.join(base, "ingredients_category.xlsx"))
df1.to_csv(os.path.join(base, "ingredients_category.csv"), index=False)
print(f"Shape: {df1.shape}")
print(f"Columns: {list(df1.columns)}")
print(df1.head(3).to_string())
print(f"\nNulls:\n{df1.isnull().sum()}")
print(f"\nSample values per column:")
for col in df1.columns:
    print(f"  {col}: {df1[col].dropna().unique()[:5]}")

print("\n\n")

# 2. product
print("=" * 80)
print("2. product.xlsx")
print("=" * 80)
df2 = pd.read_excel(os.path.join(base, "product.xlsx"))
df2.to_csv(os.path.join(base, "product.csv"), index=False)
print(f"Shape: {df2.shape}")
print(f"Columns: {list(df2.columns)}")
print(df2.head(3).to_string())
print(f"\nNulls:\n{df2.isnull().sum()}")

print("\n\n")

# 3. product_claim_category
print("=" * 80)
print("3. product_claim_category.xlsx")
print("=" * 80)
df3 = pd.read_excel(os.path.join(base, "product_claim_category.xlsx"))
df3.to_csv(os.path.join(base, "product_claim_category.csv"), index=False)
print(f"Shape: {df3.shape}")
print(f"Columns: {list(df3.columns)}")
print(df3.head(5).to_string())
print(f"\nNulls:\n{df3.isnull().sum()}")
print(f"\nUnique claim categories:")
for col in df3.columns:
    uniq = df3[col].dropna().unique()
    print(f"  {col} ({len(uniq)} unique): {uniq[:10]}")

print("\n\n")

# 4. ingredientsList (INCI)
print("=" * 80)
print("4. ingredientsList.csv (INCI List)")
print("=" * 80)
df4 = pd.read_csv(r"d:\school\uni\sem 6\llm\week8\dataset\Skin care product ingredients - INCI List\ingredientsList.csv")
print(f"Shape: {df4.shape}")
print(f"Columns: {list(df4.columns)}")
print(df4.head(3).to_string())
print(f"\nNulls:\n{df4.isnull().sum()}")
