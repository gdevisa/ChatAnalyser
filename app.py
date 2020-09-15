import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import base64
import datetime
import io
import plotly.graph_objs as go


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
    dcc.Upload(id='upload-file', children=html.Button('Upload File'),
                multiple=False),
    html.Div(id='output-file-upload')
])


@app.callback(Output('output-file-upload', 'children'), [Input('upload-file', 'contents'), Input('upload-file', 'filename')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        df = parse_data(contents, filename)
        labels = df['from'].unique()
        values = df.groupby(by='from', axis=0).size()

        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

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

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df





if __name__ == '__main__':
    app.run_server(debug=True)