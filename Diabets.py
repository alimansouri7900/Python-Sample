import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Read CSV File
data012 = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")
data_binary = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")

# Showing numbers of pre-diabetic patients to show data_binary is actually better
plt.figure(figsize=(10, 6))
labels = ['Non-Diabetic', 'Pre-Diabetic', 'Diabetic']
colors = ['blue', 'yellow', 'black']

for i in range(3):
    plt.hist(data012[data012['Diabetes_012'] == i]['Diabetes_012'], bins=1, color=colors[i], edgecolor='black', label=labels[i], alpha=0.7)

plt.title('Histogram of Diabetes_012')
plt.xlabel('Diabetes_012')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Separate 2-category variables and numeric variables
target = "Diabetes_binary"
bool_vars = [col for col in data_binary.columns if data_binary[col].nunique() == 2 and col != target]
num_vars = [col for col in data_binary.columns if col not in bool_vars + [target]]

# Define colors for binary variables
colors = ["#33ff33", "#ff0000"]

# Define function to analyze categorical variables
def analyse_cat(var):
    df = data_binary.copy()
    df['Diabetes_binary'] = df['Diabetes_binary'].map({0: "Non-Diabetic", 1: "Diabetic"})
    df[var] = df[var].map({0: "No", 1: "Yes"})

    plot_data = df.groupby(['Diabetes_binary', var]).size().reset_index(name='count')
    total_counts = df['Diabetes_binary'].value_counts().to_dict()
    plot_data['prop'] = plot_data.apply(lambda row: row['count'] / total_counts[row['Diabetes_binary']], axis=1)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='prop', y='Diabetes_binary', hue=var, data=plot_data, palette=colors)
    plt.xlim(0, 1)
    plt.xlabel('Proportion')
    plt.ylabel('Diabetes')
    plt.title(f'Distribution of {var} by Diabetes Status')
    plt.legend(title=var, loc='upper right')
    plt.show()

# Analyze all boolean variables
for var in bool_vars:
    analyse_cat(var)

# Plot histograms for numeric variables one by one
for x in num_vars:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data_binary, x=x, hue='Diabetes_binary', multiple='stack', palette=colors, kde=True)
    plt.title(f"Histogram of {x}")
    plt.xlabel(x)
    plt.ylabel('Frequency')
    plt.legend(title='Diabetes')
    plt.tight_layout()
    plt.show()

# Correlation Heatmap
cor_matrix = data_binary.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(cor_matrix, annot=True, fmt=".2f", cmap=sns.diverging_palette(220, 20, as_cmap=True), center=0)
plt.title('Correlation Heatmap')
plt.show()

#This code was written to visualize data
#This code was written to answer the question in data analysis practice,
#to se whether which data set to choose for analysis
#and which parameters are used for hypothesis testing.
