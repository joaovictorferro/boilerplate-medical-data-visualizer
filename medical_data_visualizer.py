import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("/workspace/boilerplate-medical-data-visualizer/medical_examination.csv")

# 2
df['overweight'] = df['overweight'] = ((df['weight']/ (df['height']/100) **2) >25).astype(int)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4
def draw_cat_plot():
    # 5
    df0 = df[df['cardio'] == 0]
    df0long = pd.melt(df0, value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    df1 = df[df['cardio'] == 1]
    df1long = pd.melt(df1, value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    fig, axs = plt.subplots(ncols=2, figsize=(15, 5))
    
    plot0 = sns.countplot(data=df0long, x='variable', hue='value', ax=axs[0]).set(title='cadio = 0', ylabel="total")
    
    axs[0].legend([],[], frameon=False)
    
    plot1 = sns.countplot(data=df1long, x='variable', hue='value', ax=axs[1]).set(title='cadio = 1', ylabel="total")
    
    sns.move_legend(axs[1], "right", bbox_to_anchor=(1.15, 0.5))

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &  # Pressão diastólica não pode ser maior que a sistólica
        (df['height'] >= df['height'].quantile(0.025)) &  # Altura acima do percentil 2.5
        (df['height'] <= df['height'].quantile(0.975)) &  # Altura abaixo do percentil 97.5
        (df['weight'] >= df['weight'].quantile(0.025)) &  # Peso acima do percentil 2.5
        (df['weight'] <= df['weight'].quantile(0.975))    # Peso abaixo do percentil 97.5
    ]

    # 12
    corr = df.corr().round(decimals=1)

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(ncols=1,figsize=(12, 10))

    # 15

    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f")

    # 16
    fig.savefig('heatmap.png')
    return fig
