# Import seaborn
import seaborn as sns
import pandas as pd
# Apply the default theme
sns.set_theme()

# Load an example dataset
tips = pd.read_csv('tips.csv')

# Create a visualization
sns.relplot(
    data=tips,
    x="total_bill", y="tip", col="time",
    hue="smoker", style="smoker", size="size",
)