import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# BEFORE EDA
# remove duplicate columns or rows if any

# AFTER EDA
# remove unnecessary columns such as ID that do not contribute in finding result(house price)
# replace null values is all coloumns
# remove the 2 outliers found in lving area vs saleprice graph
# preform logarithmic transformation on saleprice
# encoding


df= pd.read_csv('data/train.csv', encoding= 'utf-8')

# drop ID column    --  df.drop(columns=['column_name1','column_name2'], inplace=True)
df.drop("Id", axis=1, inplace=True)

print("Id" in df.columns)       #chechking -- since the chnage is only made in memory and not the csv file

# Handling Null values
df['PoolQC'] = df['PoolQC'].fillna('None')
df['MiscFeature'] = df['MiscFeature'].fillna('None')
df['Alley'] = df['Alley'].fillna('None')
df['Fence'] = df['Fence'].fillna('None')
df['MasVnrType'] = df['MasVnrType'].fillna('None')
df['FireplaceQu'] = df['FireplaceQu'].fillna('None')

df['GarageType'] = df['GarageType'].fillna('None')
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].median())
df['GarageFinish'] = df['GarageFinish'].fillna('None')
df['GarageQual'] = df['GarageQual'].fillna('None')
df['GarageCond'] = df['GarageCond'].fillna('None')

df['BsmtQual'] = df['BsmtQual'].fillna('None')
df['BsmtCond'] = df['BsmtCond'].fillna('None')
df['BsmtExposure'] = df['BsmtExposure'].fillna('None')
df['BsmtFinType1'] = df['BsmtFinType1'].fillna('None')
df['BsmtFinType2'] = df['BsmtFinType2'].fillna('None')

df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median())
)

print(df.isnull().sum()[df.isnull().sum() > 0])

# Remove outliers
print("Before:", df.shape)

outliers= df[(df['SalePrice']<300000) & (df['GrLivArea']>4000)]
df= df.drop(outliers.index)

print("After:", df.shape)

#logarithmic transformation on SalePrice -- feature engineering
df["SalePrice"] = np.log1p(df["SalePrice"])

#checking
print(df["SalePrice"].head())

plt.figure(figsize=(8,5))
sns.histplot(df["SalePrice"], bins=30, kde=True)
plt.title("SalePrice After Log Transformation")
plt.xlabel("Sale Price")
plt.ylabel("Number of Houses")
plt.show()

df.to_csv("cleaned_house_prices.csv", index=False)

# Encoding

# Label encoding
#le = LabelEncoder()
#df['ExterQual_Encoded'] = le.fit_transform(df['ExterQual'])
#cannot use labelencoder because it encode alphabetically

quality_map1 = {
    "None": 0,
    "Po": 1,
    "Fa": 2,
    "TA": 3,
    "Gd": 4,
    "Ex": 5
}

df["ExterQual"] = df["ExterQual"].map(quality_map1)
df["ExterCond"] = df["ExterCond"].map(quality_map1)
df["BsmtQual"] = df["BsmtQual"].map(quality_map1)
df["BsmtCond"] = df["BsmtCond"].map(quality_map1)
df["HeatingQC"] = df["HeatingQC"].map(quality_map1)
df["KitchenQual"] = df["KitchenQual"].map(quality_map1)
df["FireplaceQu"] = df["FireplaceQu"].map(quality_map1)
df["GarageQual"] = df["GarageQual"].map(quality_map1)
df["GarageCond"] = df["GarageCond"].map(quality_map1)
df["PoolQC"] = df["PoolQC"].map(quality_map1)

quality_map2 = {
    "None": 0,
    "No": 1,
    "Mn": 2,
    "Av": 3,
    "Gd": 4
}

df['BsmtExposure']= df['BsmtExposure'].map(quality_map2)

Bsmt_map3 = {
    "None": 0,
    "Unf": 1,
    "LwQ": 2,
    "Rec": 3,
    "BLQ": 4,
    "ALQ": 5,
    "GLQ": 6
}

df['BsmtFinType1'] = df['BsmtFinType1'].map(Bsmt_map3)
df['BsmtFinType2'] = df['BsmtFinType2'].map(Bsmt_map3)

CentralAir_map4 = {
    "N": 0,
    "Y": 1
}
df['CentralAir'] = df['CentralAir'].map(CentralAir_map4)

Garage_map5 = {
    "None": 0,
    "Unf": 1,
    "RFn": 2,
    "Fin": 3
}

df['GarageFinish'] = df['GarageFinish'].map(Garage_map5)

functional_map6 = {
    "Sal": 0,
    "Sev": 1,
    "Maj2": 2,
    "Maj1": 3,
    "Mod": 4,
    "Min2": 5,
    "Min1": 6,
    "Typ": 7
}

df["Functional"] = df["Functional"].map(functional_map6)

landslope_map7 = {
    "Sev": 0,
    "Mod": 1,
    "Gtl": 2
}

df["LandSlope"] = df["LandSlope"].map(landslope_map7)

# One-Hot encoding
nominal_cols = [
    "MSZoning",
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "Fence",
    "MiscFeature",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "Foundation",
    "Heating",
    "Electrical",
    "GarageType",
    "PavedDrive",
    "SaleType",
    "SaleCondition"
]

df = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=int)

print(df.select_dtypes(include="object").columns)

df.to_csv("encoded_house_prices.csv", index=False)



