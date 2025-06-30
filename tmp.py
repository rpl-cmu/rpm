import seaborn as sns
import matplotlib.pyplot as plt

# Load a sample dataset
fmri = sns.load_dataset("fmri")

# Create a line plot with markers
sns.lineplot(data=fmri, x="timepoint", y="signal", hue="event", markers=True)

# Customize marker appearance (optional)
sns.lineplot(data=fmri, x="timepoint", y="signal", marker='o', markersize=8, markerfacecolor='red')

plt.title("Line Plot with Markers")
plt.show()