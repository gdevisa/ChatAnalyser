import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from collections import Counter
import emoji
import base64
import datetime
import io
import plotly.graph_objs as go
import regex

from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json

import langid
import dostoevsky
import spacy

external_stylesheets = [dbc.themes.LITERA]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    "graphBackground": "#F5F5F5",
    "background": "#ffffff",
    "text": "#000000"
}

app.layout = html.Div([
    dbc.Container([
        dcc.Upload(
                id='upload-file',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=False
            ),
        html.Div(id='output-file-upload')
    ])
])


@app.callback(Output('output-file-upload', 'children'), [Input('upload-file', 'contents'), Input('upload-file', 'filename')])
def update_output(contents, filename):
    if contents is not None:
        df = parse_data(contents, filename)

        # Total messages and emojis pie chart
        total_emojis_list = list([a for b in df.emoji for a in b])
        emoji_dict = dict(Counter(total_emojis_list))
        emoji_dict = sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True)
        emoji_df = pd.DataFrame(emoji_dict, columns=['emoji', 'count'])
        emoji_df = emoji_df[:8]

        labels = df['from'].unique()
        values = df.groupby(by='from', axis=0).size()

        words_total = df.groupby(by='from', axis=0).sum()
        words_per_message = words_total['Words'] / values

        fig_pie = px.pie(emoji_df, names='emoji', values='count')
        fig_pie.update_traces(textposition='inside', textinfo='percent+label', showlegend=False)
        fig_pie.update_layout(
            title={
                'text': 'Emojis Used',
                'y': 0.92,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

        #Chronology line chart
        date_df = df.groupby('Date').size().rename_axis('Date').to_frame('Messages')
        date_df.reset_index(inplace=True)
        fig_line = px.line(date_df, x='Date', y='Messages')
        fig_line.update_layout(
            title={
                'text': "Chronology",
                'y':0.92,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

        # Activity by day area plot
        df['DayOfWeek'] = df['date'].dt.dayofweek
        date_df = df.groupby('DayOfWeek').size().rename_axis('DayOfWeek').to_frame('Messages')
        date_df.reset_index(inplace=True)
        date_df['DayOfWeek'] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        fig_days = px.line_polar(date_df, r='Messages', theta='DayOfWeek', line_close=True)
        fig_days.update_traces(fill='toself')
        fig_days.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 4000]
                )),
            showlegend=False,
            title={
                'text': "Activity by Day",
                'y': 0.96,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )

        #Activity by hour area plot
        df['Hour'] = df['date'].dt.hour
        time_df = df.groupby('Hour').size().rename_axis('Hour').to_frame('MessageCount')
        time_df.reset_index(inplace=True)

        fig_hour = go.Figure()
        fig_hour.add_trace(go.Scatter(x=time_df.Hour, y=time_df.MessageCount,
                    mode='lines', fill='tozeroy'))
        fig_hour.update_layout(
            showlegend=False,
            title={
                'text': "Activity by Hour",
                'y': 0.92,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )

        #Total Stats
        jumbotron = dbc.Jumbotron(
            [
                html.H1("Results", className="display-3"),
                html.Hr(className="my-2"),
                html.P(),
                html.H5('Total Messages', className="card-title"),
                html.P(
                    f"{len(df):,}",
                    className="card-text",
                ),
                html.H5('Total Words', className="card-title"),
                html.P(
                    f"{words_total.sum().values[0]:,}",
                    className="card-text",
                ),
                html.H5('Time Period', className="card-title"),
                html.P(
                    'from {0} to {1}'.format(str(df.Date[0]), str(df.Date[len(df)])),
                    className="card-text",
                ),

            ], style={'textAlign': 'center'}
        )


        cards = html.Div(
            [
                jumbotron,
                dbc.Row(
                    [
                        dbc.Col(dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5(labels[0], className="card-title"),
                                                html.P(
                                                    'Messages Sent: ' + f'{values[0]:,}',
                                                    className="card-text",
                                                ),
                                                html.P(
                                                    'Words per Message: ' + str(round(words_per_message[0], 3)),
                                                    className="card-text",
                                                )
                                            ]
                                        )
                                    )),
                        dbc.Col(dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5(labels[1], className="card-title"),
                                                html.P(
                                                    'Messages Sent: ' + f'{values[1]:,}',
                                                    className="card-text",
                                                ),
                                                html.P(
                                                    'Words per message: ' + str(round(words_per_message[1], 3)),
                                                    className="card-text",
                                                )
                                            ]
                                        )
                                    ))
                    ],
                    className="mb-4", style={'textAlign': 'center'}
                ),
                dbc.Row(
                    [
                        dbc.Col(dbc.Card(
                             [dbc.Row([dbc.Col(dcc.Graph(figure = fig_pie))]),
                              dbc.Row([dbc.Col(dcc.Graph(figure=fig_days)), dbc.Col(dcc.Graph(figure=fig_hour))]),
                              dbc.Row([dbc.Col(dcc.Graph(figure = fig_line))])
                             ]
                         )),

                    ]
                ),
            ]
        )

        return [html.Div(cards)]

def parse_data(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'json' in filename:
        # Assume that the user uploaded a JSON file
            data = json.loads(decoded)

            df = pd.DataFrame.from_records(data['messages'])
            df = df.loc[df['type'] == 'message']
            df = df[df['forwarded_from'].isnull()]
            df = df.drop(
                ['photo', 'width', 'height', 'edited', 'file', 'media_type', 'duration_seconds', 'actor', 'actor_id',
                 'forwarded_from',
                 'reply_to_message_id', 'discard_reason', 'mime_type', 'thumbnail', 'action', 'sticker_emoji', 'type',
                 'id'], axis=1)
            df['date'] = pd.to_datetime(df['date'])
            df['Date'] = df['date'].dt.date
            df['Day'] = df['date'].dt.day
            df['Time'] = df['date'].dt.time
            df['text'] = df['text'].apply(concat)
            df['emoji'] = df['text'].apply(split_count)
            df['Words'] = df['text'].apply(lambda s: len(s.split(' ')))
            df.to_csv('data_saved.csv')
            return df

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

def split_count(text):
    emoji_list = []
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_list.append(word)

    return emoji_list

def concat(text):
    if isinstance(text, list):
        text = ''
    return text

def sentiment_analysis(data):
    langid.set_languages(['en', 'ru'])
    lang = langid.classify(data['text'][0])[0]

if __name__ == '__main__':
    app.run_server(debug=True)