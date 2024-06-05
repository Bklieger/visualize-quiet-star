import json
import streamlit as st
import altair as alt
import pandas as pd
from matplotlib.colors import Normalize, to_hex
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str, default="generated_data.json", help="File name for the input data")
args = parser.parse_args()

# Load data from JSON file
with open(args.data_file, "r") as f:
    data = json.load(f)

generated_text = data["generated_text"]
probabilities = data["probabilities"]
distances_from_top = data["distances_from_top"]
token_texts = data["token_texts"]

# Apply logarithmic transformation to distances_from_top for color mapping
log_distances_from_top = np.log(np.array(distances_from_top) + 1e-10)  # Adding a small constant to avoid log(0)
min_log_dist = log_distances_from_top.min()
max_log_dist = log_distances_from_top.max()

# Convert data to a pandas DataFrame
prob_df = pd.DataFrame({
    'Token Index': list(range(len(probabilities))),
    'Probability': probabilities,
    'More Likely Tokens Count': distances_from_top,
    'Token': [token.replace('<', '&lt;').replace('>', '&gt;') for token in token_texts]  # Escape < and >
})

# Streamlit app
st.title("Generated Text Visualization with Probabilities and More Likely Tokens Count")

st.markdown("##### Probability of <|startthought|> Token Over Time")
# Plot probabilities using Altair
prob_chart = alt.Chart(prob_df).mark_line(point=True).encode(
    x=alt.X('Token Index:Q', title='Token Index'),
    y=alt.Y('Probability:Q', title='Probability'),
    tooltip=['Token Index:Q', 'Token:N', 'Probability:Q', 'More Likely Tokens Count:Q']
).properties(
    title=''
)

st.altair_chart(prob_chart, use_container_width=True)

st.markdown("##### Position of <|startthought|> (Log Scale)")
# Plot more likely tokens count using Altair (flipped log scale)
more_likely_chart = alt.Chart(prob_df).mark_line(point=True, color='red').encode(
    x=alt.X('Token Index:Q', title='Token Index'),
    y=alt.Y('More Likely Tokens Count:Q', title='Number of More Likely Tokens', scale=alt.Scale(type="log", domain=[np.max(distances_from_top) + 1, 1])),
    tooltip=['Token Index:Q', 'Token:N', 'Probability:Q', 'More Likely Tokens Count:Q']
).properties(
    title=''
)

st.altair_chart(more_likely_chart, use_container_width=True)

# Highlight text based on token_texts with a legend description
color_map = plt.get_cmap('coolwarm_r')
norm = Normalize(vmin=min_log_dist, vmax=max_log_dist)
colors = [to_hex(color_map(norm(dist))) for dist in log_distances_from_top]

highlighted_text = ""
prev_char = ""

# Iterate over each token and its corresponding color
for token, color in zip(token_texts, colors):
    token = token.replace('<', '&lt;').replace('>', '&gt;')  # Escape < and >

    # If the previous character was not a space and the current token is not a space, add a space
    if prev_char != " " and token != " " and prev_char != "":
        highlighted_text += f'<span style="background-color:{color}"> </span>'

    if token == " ":
        highlighted_text += f'<span style="background-color:{color}">&nbsp;</span>'  # Handle spaces
    elif token == "\n":
        highlighted_text += f'<span style="background-color:{color}">\\n</span><br>'  # Handle newlines
    else:
        highlighted_text += f'<span style="background-color:{color}">{token}</span>'
    
    prev_char = token

st.markdown("""
##### Probability Map of <|startthought|>

**Color Legend:**
- **Red**: More likely
- **Grey**: Moderately likely
- **Blue**: Less likely
""")

# Display the highlighted text with a legend description
st.markdown(f'<div style="font-family: monospace; font-size: 16px; white-space: pre-wrap;">{highlighted_text}</div>', unsafe_allow_html=True)
