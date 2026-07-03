import pandas as pd

df= pd.read_csv('data/train.csv', encoding='utf-8')

#Number of rows and columns : tuple (number of rows, number of columns)
print(df.shape)

#Column names: 
print(df.columns)

#Check duplicate columns 
print(df.columns[df.columns.duplicated()])       #same name -- if the output is Index([], dtype='object') -- no duplicate column names.

#same data -- if the output is [] -- no duplicate columns
duplicate_columns = []

for i in range(len(df.columns)):
    col1 = df.columns[i]

    for j in range(i + 1, len(df.columns)):
        col2 = df.columns[j]

        if df[col1].equals(df[col2]):
            duplicate_columns.append((col1, col2))

print(duplicate_columns)

#same row
print("Duplicate rows:", df.duplicated().sum())

#Comprehensive summary: index range, total rows & columns, non-null counts per column, data types and memory usage
df.info()

#Displays first n rows (default 5)
print(df.head(10))      #print(df.head(n))
#Displays last n rows (default 5).
print(df.tail())       #print(df.tail(n))


#To check specific columns:
#Single column: df [' column_name' ] returns a Series.
print(df['LotShape'])
#Multiple columns: df [['col1', 'col2']] returns a DataFrame.
print(df[['LotShape', 'LotArea']])

#FILTERING

#single condition: 
filtered= df[df['LotShape']=='Reg']

#Multiple conditions: AND (&) and OR (I) 
filtered= df[(df['LotShape']=='Reg') | (df['SalePrice']> 200000)]

#Descriptive statistics for al numerical columns:
print(df.describe())

"""
Outputs include:
count: Number of non-null entries.
mean : Arithmetic average.
std (standard deviation): Measures spread or variability of the data around the mean.
min: Minimum value.
25% (1st quartile): Value below which 25% of the data lie.
50% (median): Middle value when data sorted.
75% (3rd quartile): Value below which 75% of data lie.
max: Maximum value
"""