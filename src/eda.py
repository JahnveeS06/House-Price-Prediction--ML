import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df= pd.read_csv('data/train.csv', encoding='utf-8')

#1. Distribution of House Prices (Target Variable)
plt.figure(figsize=(8,5))

sns.histplot(df["SalePrice"], bins=30, kde=True)

plt.title("Distribution of Sale Price")
plt.xlabel("Sale Price")
plt.ylabel("Number of Houses")

plt.savefig('plots/Distribution of Sale Price.png', dpi=300, bbox_inches='tight')
plt.show()


#2. Missing Values
missing = df.isnull().sum()

missing = missing[missing > 0].sort_values(ascending=False)

plt.figure(figsize=(12,6))

sns.barplot(
    x=missing.index,
    y=missing.values
)

plt.xticks(rotation=90)

plt.title("Missing Values in Each Column")
plt.xlabel("Columns")
plt.ylabel("Missing Count")

plt.savefig('plots/Missing Values in Each Column.png', dpi=300, bbox_inches='tight')
plt.show()

#3. Correlation Heatmap
plt.figure(figsize=(18,14))

corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig('plots/Correlation Heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

#4. Living Area vs Sale Price
plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="GrLivArea",
    y="SalePrice"
)

plt.title("Ground Living Area vs Sale Price")

plt.xlabel("Ground Living Area")
plt.ylabel("Sale Price")

plt.savefig('plots/Living Area vs Sale Price', dpi=300, bbox_inches='tight')
plt.show()

#5. Overall Quality vs Sale Price
plt.figure(figsize=(10,6))

sns.boxplot(
    data=df,
    x="OverallQual",
    y="SalePrice"
)

plt.title("Overall Quality vs Sale Price")

plt.xlabel("Overall Quality")
plt.ylabel("Sale Price")

plt.savefig('plots/Overall Quality vs Sale Price', dpi=300, bbox_inches='tight')
plt.show()

#Correlation Matrix
corr = df.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)

top_corr = corr[1:11]  # Exclude SalePrice itself

plt.figure(figsize=(8,6))

sns.barplot(
    x=top_corr.values,
    y=top_corr.index
)

plt.title("Top 10 Features Correlated with SalePrice")

plt.xlabel("Correlation")
plt.ylabel("Features")

plt.savefig('plots/Correlation matrix', dpi=300, bbox_inches='tight')
plt.show()