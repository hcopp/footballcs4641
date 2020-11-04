import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

df = pd.read_csv('scaled.csv')
corr = df.corr()
cmap = sn.diverging_palette(230, 20, as_cmap=True)
# We prepare the plot  
fig, ax = plt.subplots()

# We change the fontsize of minor ticks label 
ax.tick_params(axis='both', which='major', labelsize=8)
ax.tick_params(axis='both', which='minor', labelsize=8)

sn.heatmap(corr, cmap=cmap, center=0, cbar_kws={'shrink': .5}, annot=False, xticklabels=True, yticklabels=True)
plt.show()
plt.yticks(rotation=45)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig('corr.png')