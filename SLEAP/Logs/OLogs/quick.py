import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data (replace with your actual data loading method)
data = pd.read_csv('Logs\OLogs\Foebonacci_Log.csv')

# Create box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Name', y='F1', data=data, showmeans=True, meanprops={'marker':'o', 'markerfacecolor':'white'})
plt.title('F1 Score Distribution by Model Architecture')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()