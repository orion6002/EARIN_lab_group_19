# Python dependencies, please execute:
# pip install pandas seaborn matplotlib


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load data from csv file
training = pd.read_csv("./datasets/train.csv")

print("Dataset shape:", training.shape)
print("\nColumn info:")
training.info()  # meaning of each collumn of the csv file
print("\nDescriptive statistics:")
print(training.describe())  # statistics about datas


# Classification of the data-type
df_num = training[["Age", "SibSp", "Parch", "Fare"]]  # which are quantitative
df_cat = training[
    ["Survived", "Pclass", "Sex", "Ticket", "Cabin", "Embarked"]
]  # which are qualitative


# Distribution plots for numeric variables
for col in df_num.columns:
    plt.hist(df_num[col], bins=20)
    plt.title(col)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"dist_{col}.png")
    plt.close()

# Correlation heatmap
print("\nCorrelation matrix:")
print(df_num.corr())
sns.heatmap(df_num.corr(), annot=True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "correlation_heatmap.png")
plt.close()

# Survival rates by boarding class
print("\nSurvival by Pclass:")
print(
    pd.pivot_table(
        training, index="Survived", columns="Pclass", values="Ticket", aggfunc="count"
    )
)

# Survival rates by passenger sex
print("\nSurvival by Sex:")
print(
    pd.pivot_table(
        training, index="Survived", columns="Sex", values="Ticket", aggfunc="count"
    )
)