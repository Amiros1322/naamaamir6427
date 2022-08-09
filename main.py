# from jupyter_dash import JupyterDash
from urllib.request import urlopen
import dash as dash
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, ctx
from dash.dependencies import Input, Output
# from geopy.geocoders import Nominatim  # to find city latitudes and longitudes
import plotly.graph_objects as go
import json
import statsmodels
import plotly.express as px
from pandas import json_normalize
from IPython.display import Image, display
from datetime import datetime as dt


# preproc
isr_json = None

df_daily = pd.read_csv("final_with_sozio_with_rario.csv")
old_daily = pd.read_csv("final_with_sozio_with_rario (1).csv")
df_daily = df_daily.iloc[::3, :]
df = pd.read_csv("corona_data_24.csv")

df["Year"] = df["Date"].apply(lambda d: int(d[-4:]))
df["DateStr"] = df["Date"]
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

num_cols = ['Cumulative_verified_cases', 'Cumulated_recovered', 'Cumulated_deaths', 'Cumulated_number_of_tests',
            'Cumulated_number_of_diagnostic_tests',
            "Tests", "Deaths", "Recovered", "Verified_cases"]

for col in num_cols:
    df[col] = df[col].apply(lambda x: float(x) if x != '<15' else 7.5)

df_droppedNa = df.copy()
df_droppedNa.dropna(inplace=True)

min_year = min(df["Year"])
max_year = max(df["Year"])

cum_df = df[['Cumulative_verified_cases', 'Cumulated_recovered', 'Cumulated_deaths', 'Cumulated_number_of_tests',
             'Cumulated_number_of_diagnostic_tests', "City_Name"]]

# map preprocess
state_tups = list(zip(list(df_droppedNa["Longitude"].unique()), list(df_droppedNa["Latitude"].unique())))


# Figures and dash
# IMPORTANT: average of cities and not population
line_df = df[["Date", "Ecolour", "final_score"]].groupby(by=["Date", "Ecolour"], as_index=False).agg('mean')

# graphs
fig = px.line(line_df, x=line_df["Date"], y=line_df["final_score"], line_group=line_df["Ecolour"],
              color="Ecolour", template="simple_white", color_discrete_sequence=["green", "orange", "red", "yellow"])

# Custom function for the markers

styyling = False


# Globals for drawing
def add_markers():
    ret = {str(i): f"{date[3:5]}" for i, date in enumerate(df["DateStr"].unique())}

    ret['0'] = '2020'
    ret['9'] = '2021'
    ret['21'] = '2022'

    return ret


def lines_animation(df2):
    df2 = df2.copy()
    df2['Date'] = pd.to_datetime(df2['Date']).dt.strftime('%Y-%m-%d')
    df2 = df2[(df2['Date'] > '2020-03-16') & (df2['Date'] < '2022-06-20')]
    eil = df2[df2['City_Name'].isin(['באר שבע'])]
    bash = df2[df2['City_Name'].isin(['הרצליה'])]
    ashk = df2[df2['City_Name'].isin(['נתניה'])]
    ashd = df2[df2['City_Name'].isin(['חדרה'])]
    eilat = go.Scatter(x=eil['Date'][:2],
                       y=eil['Cumulative_verified_cases'][:2],
                       mode='lines',
                       line=dict(width=1.5),
                       name="Beersheba")
    beer_sheva = go.Scatter(x=bash['Date'][:2],
                            y=bash['Cumulative_verified_cases'][:2],
                            mode='lines',
                            line=dict(width=1.5),
                            name="Herzilya")
    ashkelon = go.Scatter(x=ashk['Date'][:2],
                          y=ashk['Cumulative_verified_cases'][:2],
                          mode='lines',
                          line=dict(width=1.5),
                          name="Netanya")
    ashdod = go.Scatter(x=ashd['Date'][:2],
                        y=ashd['Cumulative_verified_cases'][:2],
                        mode='lines',
                        line=dict(width=1.5),
                        name="Hadera")
    frames = [dict(data=[dict(type='scatter',
                              x=eil['Date'][:k + 1],
                              y=eil['Cumulative_verified_cases'][:k + 1]),
                         dict(type='scatter',
                              x=bash['Date'][:k + 1],
                              y=bash['Cumulative_verified_cases'][:k + 1]),
                         dict(type='scatter',
                              x=ashk['Date'][:k + 1],
                              y=ashk['Cumulative_verified_cases'][:k + 1]),
                         dict(type='scatter',
                              x=ashd['Date'][:k + 1],
                              y=ashd['Cumulative_verified_cases'][:k + 1]),
                         ],
                   traces=[0, 1, 2, 3],
                   ) for k in range(1, len(eil) - 1)]
    layout = go.Layout(width=700,
                       height=600,
                       showlegend=False,
                       hovermode='x unified',
                       updatemenus=[
                           dict(
                               type='buttons', showactive=False,
                               y=1.05,
                               x=1.15,
                               xanchor='right',
                               yanchor='top',
                               pad=dict(t=0, r=10),
                               buttons=[dict(label='Play',
                                             method='animate',
                                             args=[None,
                                                   dict(frame=dict(duration=3,
                                                                   redraw=False),
                                                        transition=dict(duration=0),
                                                        fromcurrent=True,
                                                        mode='immediate')]
                                             )]
                           )
                       ]
                       )
    layout.update(xaxis=dict(range=['2020-7-18', '2022-06-20'], autorange=False),
                  yaxis=dict(range=[0, 110000], autorange=False, title="Cumulative Verified Cases"),
                  title="Verified Cases of interesting cities");

    fig = go.Figure(data=[eilat, beer_sheva, ashkelon, ashdod], frames=frames, layout=layout)
    fig.update_layout(title_x=0.5)
    return fig


def scatter_animation(df2):
    df2["avg_salery"] = pd.to_numeric(df2["avg_salery"], errors='coerce')

    d1 = df2[df2['City_Name'] == 'אילת']
    d2 = df2[df2['City_Name'] == 'באר שבע']
    d3 = df2[df2['City_Name'] == 'אשקלון']
    d4 = df2[df2['City_Name'] == 'אשדוד']

    m1 = df2[df2['City_Name'] == 'ראשון לציון']
    m2 = df2[df2['City_Name'] == 'רחובות']
    m3 = df2[df2['City_Name'] == 'תל אביב - יפו']

    s1 = df2[df2['City_Name'] == 'הרצליה']
    s2 = df2[df2['City_Name'] == 'נתניה']
    s3 = df2[df2['City_Name'] == 'חדרה']

    t1 = df2[df2['City_Name'] == 'חיפה']
    t2 = df2[df2['City_Name'] == 'טבריה']
    t3 = df2[df2['City_Name'] == 'קרית שמונה']

    df = pd.concat([t3, t2, t1, s3, s2, s1, m3, m2, m1, d4, d3, d2, d1], axis=0)

    fig = px.scatter(df, x="NewAllCumulative_verified_cases_Prs2", y="avg_salery",
                     color="City_Name", color_continuous_scale=px.colors.sequential.YlOrRd,
                     size="NewAllCumulative_verified_cases_Prs2",
                     animation_frame="Date", animation_group="City_Name", range_y=[5000, df['avg_salery'].max() + 668],
                     range_x=[0, df['Verfied Cases Proportion'].max()])

    fig.update_layout(title_x=0.5)
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1
    return fig


def animation_plot(df2):
    d1 = df2[df2['City_Name'] == 'אילת']
    d2 = df2[df2['City_Name'] == 'באר שבע']
    d3 = df2[df2['City_Name'] == 'אשקלון']
    d4 = df2[df2['City_Name'] == 'אשדוד']

    m1 = df2[df2['City_Name'] == 'ראשון לציון']
    m2 = df2[df2['City_Name'] == 'רחובות']
    m3 = df2[df2['City_Name'] == 'חולון']
    m4 = df2[df2['City_Name'] == 'תל אביב']
    m5 = df2[df2['City_Name'] == 'בני ברק']
    m6 = df2[df2['City_Name'] == 'פתח תקווה']

    s1 = df2[df2['City_Name'] == 'הרצליה']
    s2 = df2[df2['City_Name'] == 'כפר סבא']
    s3 = df2[df2['City_Name'] == 'נתניה']
    s4 = df2[df2['City_Name'] == 'חדרה']

    t1 = df2[df2['City_Name'] == 'חיפה']
    t2 = df2[df2['City_Name'] == 'טבריה']
    t3 = df2[df2['City_Name'] == 'קרית שמונה']

    df = pd.concat([t3, t2, t1, s4, s3, s2, s1, m6, m5, m4, m3, m2, m1, d4, d3, d2, d1], axis=0)
    # Watch as bars chart covid cases changes
    fig = px.bar(df, x="City_Name", y="Cumulative_verified_cases",
                 color="final_score", color_continuous_scale=px.colors.sequential.YlOrRd,
                 animation_frame="DateStr", animation_group="City_Name",
                 range_y=[0, df['Cumulative_verified_cases'].max()])

    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1
    # Image("bp.jpg", width=1515, height=145)
    return fig


def salary_over_time(df2):
    df2["avg_salery"] = pd.to_numeric(df2["avg_salery"], errors='coerce')

    d1 = df2[df2['City_Name'] == 'אילת']
    d2 = df2[df2['City_Name'] == 'באר שבע']
    d3 = df2[df2['City_Name'] == 'אשקלון']
    d4 = df2[df2['City_Name'] == 'אשדוד']

    m1 = df2[df2['City_Name'] == 'ראשון לציון']
    m2 = df2[df2['City_Name'] == 'רחובות']
    m3 = df2[df2['City_Name'] == 'תל אביב - יפו']

    s1 = df2[df2['City_Name'] == 'הרצליה']
    s2 = df2[df2['City_Name'] == 'נתניה']
    s3 = df2[df2['City_Name'] == 'חדרה']

    t1 = df2[df2['City_Name'] == 'חיפה']
    t2 = df2[df2['City_Name'] == 'טבריה']
    t3 = df2[df2['City_Name'] == 'קרית שמונה']

    df = pd.concat([t3, t2, t1, s3, s2, s1, m3, m2, m1, d4, d3, d2, d1], axis=0)
    df['avg_salery'] = df["avg_salery"].dropna()

    # Watch as bars chart covid cases changes
    fig = px.scatter(df, x="Cumulative Verified Cases Proportion", y="avg_salery",
                     color="City_Name", color_continuous_scale=px.colors.sequential.YlOrRd,
                     size="Cumulative Verified Cases Proportion",
                     animation_frame="Date", animation_group="City_Name", range_y=[5000, df['avg_salery'].max() + 668],
                     range_x=[0, df['Cumulative Verified Cases Proportion'].max()])
    # ,title="Total proportion infected by average salary")
    fig.update_yaxes(title="Average Salary")
    fig.update_layout(title="Total proportion infected by average salary")
    fig.update_layout(title_x=0.5)

    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1

    return fig


def old_salary_over_time(df2):
    df2["avg_salery"] = df2["avg_salery"].apply(lambda x: x.replace(",", "") if type(x) is str else x)
    df2["avg_salery"] = pd.to_numeric(df2["avg_salery"])

    d1 = df2[df2['City_Name'] == 'אילת']
    d2 = df2[df2['City_Name'] == 'באר שבע']
    d3 = df2[df2['City_Name'] == 'אשקלון']
    d4 = df2[df2['City_Name'] == 'אשדוד']

    m1 = df2[df2['City_Name'] == 'ראשון לציון']
    m2 = df2[df2['City_Name'] == 'רחובות']
    m3 = df2[df2['City_Name'] == 'תל אביב - יפו']

    s1 = df2[df2['City_Name'] == 'הרצליה']
    s2 = df2[df2['City_Name'] == 'נתניה']
    s3 = df2[df2['City_Name'] == 'חדרה']

    t1 = df2[df2['City_Name'] == 'חיפה']
    t2 = df2[df2['City_Name'] == 'טבריה']
    t3 = df2[df2['City_Name'] == 'קרית שמונה']

    df = pd.concat([t3, t2, t1, s3, s2, s1, m3, m2, m1, d4, d3, d2, d1], axis=0)

    # Watch as bars chart covid cases changes
    fig = px.scatter(df, x="NewAllCumulative_verified_cases_Prs2", y="avg_salery",
                     color="City_Name", color_continuous_scale=px.colors.sequential.YlOrRd,
                     size="NewAllCumulative_verified_cases_Prs2",

                     animation_frame="Date", animation_group="City_Name", range_y=[5000, df['avg_salery'].max() + 668],
                     range_x=[0, df['NewAllCumulative_verified_cases_Prs2'].max()])

    fig.update_xaxes(title="Part of total Population Infected")
    fig.update_yaxes(title="Average Salary")
    fig.update_layout(title="Total proportion infected by average salary")
    fig.update_layout(title_x=0.5)

    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1

    return fig


radio_labels_to_col_names = {"Cases": "Verified_cases", "Deaths": "Deaths", "Recovered": "Recovered", "Tests": "Tests",
                             "Date": "Date"}

app = dash.Dash()
server = app.server

app.layout = html.Div([

    html.H1("Corona through Ramzor - Over Time", style={'textAlign': 'center'}),

    html.Div([dcc.Graph(figure=lines_animation(df_daily))]),

    html.Div(children=[dcc.Graph(figure=old_salary_over_time(old_daily), id="bar")], id="bottom",
             style={'width': '100%', 'display': 'inline-block'}),

    html.H2("Interactive Analysis", style={'textAlign': 'center'}),

    dcc.RangeSlider(
        id="DateSlider",
        min=0,
        max=27,
        value=[0, 27],
        marks=add_markers(),
        step=1,
        allowCross=False),

    html.Div([html.H4("X value"),
              dcc.RadioItems(options=["Cases", "Tests", "Deaths", "Recovered", "Date"], value="Deaths",
                             labelStyle={'display': 'block'}, id="X_rad"),
              html.H4("Y value"),
              dcc.RadioItems(options=["Cases", "Tests", "Deaths", "Recovered"], value="Cases",
                             labelStyle={'display': 'block'}, id="Y_rad")],
             style={'position': 'absolute', 'right': '10px', 'background': '#eee'}),

    html.Div([dcc.Checklist(options=["Show Regression Line"], id="checklist")],
             style={'float': 'up-left', 'margin-left': '20px'}),

    html.Div([dcc.Graph(figure={}, id="trellis"),  # html.H4("Loading",  id="plot_title")
              ], style={'float': 'left', 'margin-left': '20px'}),

    html.Div([dcc.Graph(figure={}, id="scatter")], id="scatter_html",
             style={'padding-left': '40%', "padding-right": "40%", 'float': 'right'}),

    #

])


@app.callback(
    Output(component_id='scatter', component_property='figure'),
    Output(component_id='trellis', component_property="figure"),
    Input(component_id='DateSlider', component_property='value'),
    Input(component_id='X_rad', component_property='value'),
    Input(component_id='Y_rad', component_property='value'),
    Input(component_id='checklist', component_property='value'),
)
def update_output_div(input_value, x_val_inp, y_val_inp, check_val_inp):
    # check_val_inp inserted as list of all checked values. (we only have one)
    use_regression = ""
    if check_val_inp is not None and len(check_val_inp) > 0:
        use_regression = "ols"

    # input value is [min_val, max_val]
    x_val, y_val = radio_labels_to_col_names[x_val_inp], radio_labels_to_col_names[y_val_inp]
    min_val = input_value[0]
    max_val = input_value[1]
    curr_min_date = slider_to_date(min_val)
    curr_max_date = slider_to_date(max_val)

    # get updated dates
    dates = dates_between(min_val=curr_min_date, max_val=curr_max_date)
    selected_date_df = df[df["DateStr"].isin(dates)]

    # Location scatterplot
    locations_df = selected_date_df[["Ecolour", "Latitude", "Longitude", "City_Name"]] \
        .groupby(by=["City_Name"], as_index=False).agg(pd.Series.mode)

    fig_trell = None
    if x_val != "Date":
        # trellis plot (A city can be in 2 places if it was there during that time period.
        trellis_df = selected_date_df[["Ecolour", x_val, y_val, "City_Name"]].groupby(by=["City_Name", "Ecolour"],
                                                                                      as_index=False).agg('mean')
        # max_trell_x, max_trell_y = trellis_df[x_val].max(), trellis_df[y_val].max()
        # min_trell_x, min_trell_y = trellis_df[x_val].min(), trellis_df[y_val].min()

        if use_regression == 'ols':
            fig_trell = px.scatter(trellis_df, x=x_val, y=y_val, facet_col="Ecolour", color="Ecolour",
                                   template="simple_white",
                                   color_discrete_map={"Green": "green", "Red": "red", "Yellow": "yellow",
                                                       "Orange": "orange"},
                                   facet_col_wrap=2, hover_name="City_Name", trendline=use_regression,
                                   trendline_color_override="#99cfe0",
                                   title="Dynamic Graph Creator")
        else:
            fig_trell = px.scatter(trellis_df, x=x_val, y=y_val, facet_col="Ecolour", color="Ecolour",
                                   template="simple_white",
                                   color_discrete_map={"Green": "green", "Red": "red", "Yellow": "yellow",
                                                       "Orange": "orange"},
                                   facet_col_wrap=2, hover_name="City_Name", title="Dynamic Graph Creator")

        # fig_trell.add_trace(go.Scatter(x=[min_trell_x, max_trell_x], y=[max_trell_y, max_trell_y], fill='tozeroy'), row=1, col=1)
    else:
        trellis_df = selected_date_df[["Ecolour", x_val, y_val]].groupby(by=["Ecolour", x_val], as_index=False).agg(
            'mean')

        fig_trell = px.line(trellis_df, x=x_val, y=y_val, line_group="Ecolour",
                            color="Ecolour", template="simple_white", title="Dynamic Graph Creator",
                            color_discrete_map={"Green": "green", "Red": "red", "Yellow": "yellow", "Orange": "orange"})

    fig_trell.update_layout(showlegend=False, title_x=0.5)
    fig_trell.for_each_annotation(lambda x: x.update(text=" "))

    fig_trell = go.Figure(fig_trell)
    fig_trell.update_traces(
        opacity=0.8
    )

    def clean_mode_agg(x):
        if type(x) == np.ndarray and len(x) > 0:
            return x[1]
        elif type(x) == np.ndarray:
            return ""
        return x

    locations_df["Ecolour"] = locations_df["Ecolour"].apply(lambda x: clean_mode_agg(x))
    fig_locations = px.scatter(locations_df, x="Longitude", y="Latitude", color="Ecolour", template="simple_white",
                               color_discrete_map={"Green": "green", "Red": "red", "Yellow": "yellow",
                                                   "Orange": "orange"},
                               title="Physical Distribution Map", hover_name="City_Name")
    fig_locations.update_yaxes(range=[29, 33.5], automargin=True, showline=True, linewidth=2, linecolor='black',
                               mirror=True)
    fig_locations.update_xaxes(range=[34.5, 35.7], automargin=True, showline=True, linewidth=2, linecolor='black',
                               mirror=True)
    fig_locations.update_layout(legend_title_text=' ', width=300)

    return fig_locations, fig_trell


def slider_to_date(slider_val):
    return f"01/0{(slider_val + 4) % 12}/202{(slider_val + 4) // 12}"


# Gets dates between the min_val and max_val (they are strings formatted dd/mm/yyyy
def dates_between(min_val=None, max_val=None):
    if min_val is None or max_val is None:
        raise Exception("dates_between got a null")

    min_day, max_day = min_val[:2], max_val[:2]
    min_month, max_month = min_val[3:5], max_val[3:5]
    min_year, max_year = min_val[6:], max_val[6:]

    max_year = max_year.replace('/', '')
    min_year = min_year.replace('/', '')

    dates = []
    if min_year == max_year:
        return dates_between_same_year(min_month, max_month, min_year)
    elif int(min_year) < int(max_year):
        # get all years until max_year
        curr_year = int(min_year)
        while curr_year < int(max_year):
            dates.extend(dates_between_same_year(min_month, 13, curr_year))
            min_month = 1
            curr_year += 1

        # get last year
        dates.extend(dates_between_same_year(1, max_month, max_year))

        return dates


def dates_between_same_year(min_month, max_month, year):
    dates = []
    for m in range(int(min_month), int(max_month)):
        d_str = f"01/{m}/{year}"
        if m < 10:
            d_str = f"01/0{m}/{year}"
        dates.append(d_str)
    return dates

if __name__ == "__main__":
    app.run_server(mode="external", debug=False,dev_tools_ui=False,dev_tools_props_check=False)