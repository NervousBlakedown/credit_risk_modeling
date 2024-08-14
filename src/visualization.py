# src/visualization.py

import pandas as pd
import plotly.graph_objs as go

# Load the processed data
df = pd.read_csv('../data/processed/credit_risk_data.csv')

# Calculate the percentages of people who defaulted on loans and who didn't
percentage_default = (df['default'].sum() / len(df)) * 100
percentage_no_default = 100 - percentage_default

# Separate the data into two dataframes based on the 'default' column
df_0 = df[df['default'] == 0]
df_1 = df[df['default'] == 1]

# Create a scatter3d trace for each dataframe
trace0 = go.Scatter3d(
    x=df_0['age'],
    y=df_0['income'],
    z=df_0['loan_amount'],
    mode='markers',
    marker=dict(size=5, color='blue', symbol='circle'),
    name='Repaid',
    hovertext=df_0.index
)

trace1 = go.Scatter3d(
    x=df_1['age'],
    y=df_1['income'],
    z=df_1['loan_amount'],
    mode='markers',
    marker=dict(size=5, color='red', symbol='diamond'),
    name='Default',
    hovertext=df_1.index
)

# Combine traces into a list
data = [trace0, trace1]

# Create layout and add the calculated percentages
layout = go.Layout(
    title={
        'text': '3D Scatter plot of Credit Risk Data',
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    scene=dict(
        xaxis_title='Age',
        yaxis_title='Income',
        zaxis_title='Loan Amount'
    ),
    autosize=True,
    margin=dict(l=0, r=0, b=0, t=0),
    title_x=0.5,
    annotations=[
        dict(
            text=f"Percentage who defaulted: {percentage_default:.2f}%<br>"
                 f"Percentage who didn't default: {percentage_no_default:.2f}%",
            showarrow=False,
            align='right',
            x=1,
            y=1,
            xref='paper',
            yref='paper',
            xanchor='center',
            yanchor='bottom',
            font=dict(size=12)
        )
    ]
)

# Show 3D scatterplot
fig = go.Figure(data=data, layout=layout)
fig.show()
