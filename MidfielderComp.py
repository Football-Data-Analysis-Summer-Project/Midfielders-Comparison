import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


st.set_page_config(page_title="Midfielder Comparison", layout="centered")

st.title("🧠 Midfielder Radar Comparison")
st.markdown("Compare the performance of two Midfielders based on multiple stats.")


df = pd.read_csv("Midfielders.csv")





players = df['Name'].unique()
player1 = st.selectbox("Select First Midfielder", players, index=0)
player2 = st.selectbox("Select Second Midfielder", players, index=1)


features = [
    'Age', 'Club Level', 'Minutes Played', 'Goals',
    'Assists', 'xG', 'xA',
    'Progressive Passes', 'Pass Completion', 'SCA', 'Interceptions', 'Games Missed', 'Transfer Value'
]


inverse_features = ['Games Missed']


df_scaled = df.copy()
for feature in features:
    values = df_scaled[feature]
    if feature in inverse_features:
        values = values.max() - values
    if values.max() > 1000:
        values = np.log1p(values)  # Log scale if very large
    df_scaled[feature] = MinMaxScaler().fit_transform(values.values.reshape(-1, 1))


p1_stats = df_scaled[df_scaled['Name'] == player1][features].values.flatten()
p2_stats = df_scaled[df_scaled['Name'] == player2][features].values.flatten()


labels = features
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]
p1_stats = np.concatenate((p1_stats, [p1_stats[0]]))
p2_stats = np.concatenate((p2_stats, [p2_stats[0]]))


fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.plot(angles, p1_stats, color='blue', linewidth=2, label=player1)
ax.fill(angles, p1_stats, color='blue', alpha=0.2)

ax.plot(angles, p2_stats, color='red', linewidth=2, label=player2)
ax.fill(angles, p2_stats, color='red', alpha=0.2)

ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_facecolor('#f9f9f9')
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20", "40", "60", "80", "100"], color="gray", size=8)
ax.set_title("Midfielders Player Comparison", size=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))


st.pyplot(fig)
