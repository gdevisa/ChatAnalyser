import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    "graphBackground": "#F5F5F5",
    "background": "#ffffff",
    "text": "#000000"
}

app.layout = html.Div([
    dcc.Upload(id='upload-file', children=html.Button('Upload File (JSON)'),
                multiple=False, style= {'textAlign': 'center'}),
    html.Div(id='output-file-upload')
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

        labels = df['from'].unique()
        values = df.groupby(by='from', axis=0).size()
        fig_pie = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
        fig_pie.add_trace(go.Pie(labels=labels, values=values,
                                 name='Total Messages', hole=.3), row=1, col=1)
        fig_pie.add_trace(go.Pie(labels=emoji_df.emoji[:10], values=emoji_df['count'][:10],
                                 name='Total Emojis', hole=.3), row=1, col=2)


        # Activity by day area plot
        df['DayOfWeek'] = df['date'].dt.dayofweek
        date_df = df.groupby('DayOfWeek').size().rename_axis('DayOfWeek').to_frame('MessageCount')
        date_df.reset_index(inplace=True)
        date_df['DayOfWeek'] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fig_area_byday = px.area(date_df, x="DayOfWeek", y="MessageCount")

        #MessagesHistory line chart
        date_df = df.groupby('Date').size().rename_axis('Date').to_frame('MessageCount')
        date_df.reset_index(inplace=True)
        fig_line = px.line(date_df, x='Date', y='MessageCount')

        #Activity by hour area plot
        df['Hour'] = df['date'].dt.hour
        time_df = df.groupby('Hour').size().rename_axis('Hour').to_frame('MessageCount')
        time_df.reset_index(inplace=True)
        fig_area_byhour = px.area(time_df, x="Hour", y="MessageCount")

        return [dcc.Graph(figure =fig_pie), dcc.Graph(figure=fig_line), dcc.Graph(figure=fig_area_byday),
                dcc.Graph(figure=fig_area_byhour)]

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
            df['emoji'] = df['text'].apply(split_count)
            df.to_csv('data_saved.csv')
            return df

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

def split_count(text):
    emoji_list = []
    if isinstance(text, list):
        text = ','.join([str(i) for i in text])
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_list.append(word)

    return emoji_list

if __name__ == '__main__':
    app.run_server(debug=True)