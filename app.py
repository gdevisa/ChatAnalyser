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

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from skimage import io

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    "graphBackground": "#F5F5F5",
    "background": "#ffffff",
    "text": "#000000"
}

app.layout = html.Div([
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
        fig_pie = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                                subplot_titles=('Total Messages', 'Total Emojis'))
        fig_pie.add_trace(go.Pie(labels=labels, values=values,
                                 textinfo='label+value+percent'), row=1, col=1)
        fig_pie.add_trace(go.Pie(labels=emoji_df.emoji[:8], values=emoji_df['count'][:8],
                                 textinfo='label+percent'), row=1, col=2)
        fig_pie.update_layout(showlegend=False)


        #MessagesHistory line chart
        date_df = df.groupby('Date').size().rename_axis('Date').to_frame('MessageCount')
        date_df.reset_index(inplace=True)
        fig_line = px.line(date_df, x='Date', y='MessageCount')
        fig_line.update_layout(
            title={
                'text': "Message History",
                'y':0.92,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

        # Activity by day area plot
        df['DayOfWeek'] = df['date'].dt.dayofweek
        date_df = df.groupby('DayOfWeek').size().rename_axis('DayOfWeek').to_frame('MessageCount')
        date_df.reset_index(inplace=True)
        date_df['DayOfWeek'] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        #Activity by hour area plot
        df['Hour'] = df['date'].dt.hour
        time_df = df.groupby('Hour').size().rename_axis('Hour').to_frame('MessageCount')
        time_df.reset_index(inplace=True)

        fig = make_subplots(rows=1, cols=2, subplot_titles=('Activity by Hour', 'Activity by Day'))
        fig.add_trace(go.Scatter(x=time_df.Hour, y=time_df.MessageCount,
                    mode='lines', fill='tozeroy'), 1, 1)
        fig.add_trace(go.Scatter(x=date_df.DayOfWeek, y=date_df.MessageCount,
                    mode='lines', fill='tozeroy'), 1, 2)
        fig.update_layout(showlegend=False)

        #Generating a wordcloud
        texts = df['text']
        #exec(open("stopwords.txt").read(), globals())
        stopwords = ["типа", "просто", "вроде", "ну", "вообще", "там", "короче", "потом", "раз", "тоже", "че", "давай",
                     "тут", "сейчас", "бля", "надо", "шо", "много", "сегодня", "завтра", "вчера", "понял", "очень",
                     "хочу", "думаю", "кстати", "ща", "знаю", "Оо", "Ооо", "Окей", "наверное", "думал", "пару", "чисто",
                     "думаю", "бо", "хочу", "хотел", "какие", "понимаю", "норм", "мб", "c", "а", "алло", "без", "белый",
                     "близко", "более", "больше", "большой", "будем", "будет", "будете", "будешь", "будто", "буду",
                     "будут", "будь", "бы", "бывает", "бывь", "был", "была", "были", "было", "быть", "в", "важная",
                     "важное", "важные", "важный", "вам", "вами", "вас", "ваш", "ваша", "ваше", "ваши", "вверх",
                     "вдали", "вдруг", "ведь", "везде", "вернуться", "весь", "вечер", "взгляд", "взять", "вид", "видел",
                     "видеть", "вместе", "вне", "вниз", "внизу", "во", "вода", "война", "вокруг", "вон", "вообще",
                     "вопрос", "восемнадцатый", "восемнадцать", "восемь", "восьмой", "вот", "впрочем", "времени",
                     "время", "все", "все еще", "всегда", "всего", "всем", "всеми", "всему", "всех", "всею", "всю",
                     "всюду", "вся", "всё", "второй", "вы", "выйти", "г", "где", "главный", "глаз", "говорил",
                     "говорит", "говорить", "год", "года", "году", "голова", "голос", "город", "да", "давать", "давно",
                     "даже", "далекий", "далеко", "дальше", "даром", "дать", "два", "двадцатый", "двадцать", "две",
                     "двенадцатый", "двенадцать", "дверь", "двух", "девятнадцатый", "девятнадцать", "девятый", "девять",
                     "действительно", "дел", "делал", "делать", "делаю", "дело", "день", "деньги", "десятый", "десять",
                     "для", "до", "довольно", "долго", "должен", "должно", "должный", "дом", "дорога", "друг", "другая",
                     "другие", "других", "друго", "другое", "другой", "думать", "душа", "е", "его", "ее", "ей", "ему",
                     "если", "есть", "еще", "ещё", "ею", "её", "ж", "ждать", "же", "жена", "женщина", "жизнь", "жить",
                     "за", "занят", "занята", "занято", "заняты", "затем", "зато", "зачем", "здесь", "земля", "знать",
                     "значит", "значить", "и", "иди", "идти", "из", "или", "им", "имеет", "имел", "именно", "иметь",
                     "ими", "имя", "иногда", "их", "к", "каждая", "каждое", "каждые", "каждый", "кажется", "казаться",
                     "как", "какая", "какой", "кем", "книга", "когда", "кого", "ком", "комната", "кому", "конец",
                     "конечно", "которая", "которого", "которой", "которые", "который", "которых", "кроме", "кругом",
                     "кто", "куда", "лежать", "лет", "ли", "лицо", "лишь", "лучше", "любить", "люди", "м", "маленький",
                     "мало", "мать", "машина", "между", "меля", "менее", "меньше", "меня", "место", "миллионов", "мимо",
                     "минута", "мир", "мира", "мне", "много", "многочисленная", "многочисленное", "многочисленные",
                     "многочисленный", "мной", "мною", "мог", "могу", "могут", "мож", "может", "может быть", "можно",
                     "можхо", "мои", "мой", "мор", "москва", "мочь", "моя", "моё", "мы", "на", "наверху", "над", "надо",
                     "назад", "наиболее", "найти", "наконец", "нам", "нами", "народ", "нас", "начала", "начать", "наш",
                     "наша", "наше", "наши", "не", "него", "недавно", "недалеко", "нее", "ней", "некоторый", "нельзя",
                     "нем", "немного", "нему", "непрерывно", "нередко", "несколько", "нет", "нею", "неё", "ни",
                     "нибудь", "ниже", "низко", "никакой", "никогда", "никто", "никуда", "ним", "ними", "них", "ничего",
                     "ничто", "но", "новый", "нога", "ночь", "ну", "нужно", "нужный", "нх", "о", "об", "оба", "обычно",
                     "один", "одиннадцатый", "одиннадцать", "однажды", "однако", "одного", "одной", "оказаться", "окно",
                     "около", "он", "она", "они", "оно", "опять", "особенно", "остаться", "от", "ответить", "отец",
                     "откуда", "отовсюду", "отсюда", "очень", "первый", "перед", "писать", "плечо", "по", "под",
                     "подойди", "подумать", "пожалуйста", "позже", "пойти", "пока", "пол", "получить", "помнить",
                     "понимать", "понять", "пор", "пора", "после", "последний", "посмотреть", "посреди", "потом",
                     "потому", "почему", "почти", "правда", "прекрасно", "при", "про", "просто", "против", "процентов",
                     "путь", "пятнадцатый", "пятнадцать", "пятый", "пять", "работа", "работать", "раз", "разве", "рано",
                     "раньше", "ребенок", "решить", "россия", "рука", "русский", "ряд", "рядом", "с", "с кем", "сам",
                     "сама", "сами", "самим", "самими", "самих", "само", "самого", "самой", "самом", "самому", "саму",
                     "самый", "свет", "свое", "своего", "своей", "свои", "своих", "свой", "свою", "сделать", "сеаой",
                     "себе", "себя", "сегодня", "седьмой", "сейчас", "семнадцатый", "семнадцать", "семь", "сидеть",
                     "сила", "сих", "сказал", "сказала", "сказать", "сколько", "слишком", "слово", "случай", "смотреть",
                     "сначала", "снова", "со", "собой", "собою", "советский", "совсем", "спасибо", "спросить", "сразу",
                     "стал", "старый", "стать", "стол", "сторона", "стоять", "страна", "суть", "считать", "т", "та",
                     "так", "такая", "также", "таки", "такие", "такое", "такой", "там", "твои", "твой", "твоя", "твоё",
                     "те", "тебе", "тебя", "тем", "теми", "теперь", "тех", "то", "тобой", "тобою", "товарищ", "тогда",
                     "того", "тоже", "только", "том", "тому", "тот", "тою", "третий", "три", "тринадцатый",
                     "тринадцать", "ту", "туда", "тут", "ты", "тысяч", "у", "увидеть", "уж", "уже", "улица", "уметь",
                     "утро", "хороший", "хорошо", "хотел бы", "хотеть", "хоть", "хотя", "хочешь", "час", "часто",
                     "часть", "чаще", "чего", "человек", "чем", "чему", "через", "четвертый", "четыре", "четырнадцатый",
                     "четырнадцать", "что", "чтоб", "чтобы", "чуть", "шестнадцатый", "шестнадцать", "шестой", "шесть",
                     "эта", "эти", "этим", "этими", "этих", "это", "этого", "этой", "этом", "этому", "этот", "эту", "я",
                     "являюсь"]
        stopwords = set(stopwords)
        wordcloud = WordCloud(stopwords=stopwords,
                              width=1000, height=500, random_state=1, background_color='white', colormap='Set2',
                              collocations=False,
                              ).generate(' '.join(texts))
        plt.imshow(wordcloud)  # image show
        plt.axis('off')  # to off the axis of x and y
        plt.savefig('Plotly-World_Cloud.png')

        img = io.imread(
            'Plotly-World_Cloud.png')
        fig_wordcloud = px.imshow(img)
        fig_wordcloud.update_layout(coloraxis_showscale=False)
        fig_wordcloud.update_xaxes(showticklabels=False)
        fig_wordcloud.update_yaxes(showticklabels=False)

        return [html.H1('Total Messages: ' + str(len(df)), style={'textAlign': 'center'}),
                html.H1('History: from {0} to {1}'.format(str(df.Date[0]), str(df.Date[len(df)])), style={'textAlign': 'center'}),
                html.H1('Group Members: ' + str(len(df['from'].unique())), style={'textAlign': 'center'}),
                dcc.Graph(figure = fig_wordcloud),
                dcc.Graph(figure = fig_pie),
                dcc.Graph(figure = fig_line), dcc.Graph(figure=fig)]

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

if __name__ == '__main__':
    app.run_server(debug=True)