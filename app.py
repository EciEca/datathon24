# MOST UPDATED AS OF 6:46PM
import dash
import plotly.express as px
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.figure_factory as ff
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY,'/assets/style.css'])

wine_df = pd.read_csv("https://raw.githubusercontent.com/EciEca/datathon24/main/winequality-red.csv")

target_column = 'quality'
def plot_feature_importance(data, target_column, random_state=1, max_depth=12):
    model = RandomForestRegressor(random_state=random_state, max_depth=max_depth)
    x = data.drop([target_column], axis=1)
    data_encoded = pd.get_dummies(data)
    model.fit(x, data_encoded[target_column])
    features = data_encoded.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[:]
        fig = go.Figure(go.Bar(
        x=importances[indices],
        y=[features[i] for i in indices],
        orientation='h',
        marker=dict(color='#A65C6A'),
    ))

    fig.update_layout(
        title='Component Relative Weight on the Quality of Wine',
        xaxis_title='Relative Weights',
        yaxis_title='Components',
        plot_bgcolor='#ecdbc7',  
        paper_bgcolor='#ecdbc7'
    )
    return fig



app.layout = dbc.Container(
    fluid=True,
    style={'backgroundColor': '#ecdbc7'},
    children=[
        dbc.Row([
            dbc.Col(
                width=3,
                style={'padding': '8px'},  
                children=[
                    html.H2("Welcome to the Vineyard Ventures Dashboard", style={'margin-bottom': '20px', 'font-family': 'Raleway', 'color': '#6b0f1a'}),
                    html.Hr(),
                    html.Div(
                        id="intro",
                        children=[
                            "Discover the world of Portuguese red wines through our dashboard, curated from this ",
                            html.A("dataset", href="https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009", target="_blank",style={'color':'#A65C6A'}),
                            ". This dashboard offers a relaxed yet informative experience, serving as a great tool to delve into the distinctive qualities of Portuguese reds. Cheers to exploring the diverse flavors and characteristics that make these wines truly special!"
                        ],
                        style={'margin-bottom': '20px','margin-left': '3px'}
                    ),
                    html.Hr(),
                    html.H6("Select an Attribute to Start Exploring:", style={'font-family': 'Raleway','font-size': '24px','margin-bottom': '20px','margin-left': '3px', 'color': '#6b0f1a'}),
                    dcc.Dropdown(
                        id='variable-dropdown',
                        options=[{'label': col, 'value': col} for idx, col in enumerate(wine_df.columns[:-1])],
                        value='fixed acidity',  # default variable
                        clearable = False,
                        style={'width': '100%', 'margin-bottom': '20px','margin-left': '3px', 'font-size': '16px','background-color': '#e7cfb7'},
                    ),
                    html.Div(id='description-div',style={'margin': '0 20px', 'font-size': '16px'})       
                ]
            ),
            dbc.Col(
                width=5,
                children=[
                    dcc.Graph(id='distplot-graph', style={'margin-top': '10px'}),
                    html.Div(style={'height': '20px'}),
                    dcc.Graph(id='violinplot-graph'),
                ]
            ),
            dbc.Col(
                width=4,  
                style={'padding': '8px'},
                children=[
                    html.H3("Wine Quality Predictor",style={'font-family': 'Raleway', 'font-size': '34px','color': '#6b0f1a'}),
                    html.Div("Predicts the quality of wine based on customizable attributes. Drag the sliders to test your wine quality!", style={'margin-bottom': '20px'}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("fixed acidity"),
                            dcc.Slider(id="fixed acidity", min=4, max=18, value=11, marks={4: '4', 18: '18'},
                                       tooltip={"placement": "top", "always_visible": False},),
                            html.Label("citric acid"),
                            dcc.Slider(id="citric acid", min=0, max=1, value=0.5, marks={0: '0', 1: '1'},
                                       tooltip={"placement": "top", "always_visible": False}),
                            html.Label("chlorides"),
                            dcc.Slider(id="chlorides", min=0, max=1, value=0.5, marks={0: '0', 1: '1'},
                                       tooltip={"placement": "top", "always_visible": False}),
                            html.Label("total sulfur dioxide"),
                            dcc.Slider(id="total sulfur dioxide", min=0, max=300, value=150, marks={0: '0', 300: '300'},
                                       tooltip={"placement": "top", "always_visible": False}),
                            html.Label("pH"),
                            dcc.Slider(id="pH", min=2, max=5, value=3.5, marks={2: '2', 5: '5'},
                                       tooltip={"placement": "top", "always_visible": False}),
                            html.Label("alcohol"),
                            dcc.Slider(id="alcohol", min=8, max=16, value=12, marks={8: '8', 16: '16'},
                                       tooltip={"placement": "top", "always_visible": False}),
                        ]),
                        dbc.Col([
                            html.Label("volatile acidity"),
                            dcc.Slider(id="volatile acidity", min=0, max=2, value=1, marks={0: '0', 2: '2'},
                                       tooltip={"placement": "top", "always_visible": False}),
                            html.Label("residual sugar"),
                            dcc.Slider(id="residual sugar", min=0, max=16, value=8, marks={0: '0', 16: '16'},
                                       tooltip={"placement": "top", "always_visible": False}),
                            html.Label("free sulfur dioxide"),
                            dcc.Slider(id="free sulfur dioxide", min=0, max=80, value=40, marks={0: '0', 80: '80'},
                                       tooltip={"placement": "top", "always_visible": False}),
                            html.Label("density"),
                            dcc.Slider(id="density", min=0.98, max=1, value=0.99, marks={0.98: '0.98', 1: '1'},
                                       tooltip={"placement": "top", "always_visible": False}),
                            html.Label("sulphates"),
                            dcc.Slider(id="sulphates", min=0, max=2, value=1, marks={0: '0', 2: '2'},
                                       tooltip={"placement": "top", "always_visible": False}),
                            html.Div(id='output-container', style={'text-align': 'center'}),      
                        ]),
                    ]),
                    dcc.Graph(id='feature-importance-graph') 
                ]
            ),
        ]),
    ]
)



bin_sizes = {
    'fixed acidity': 0.2,
    'citric acid': 0.02,
    'chlorides': 0.01,
    'total sulfur dioxide': 4.5,
    'pH': 0.02,
    'sulphates': 0.025,
    'volatile acidity': 0.025,
    'residual sugar': 0.2,
    'free sulfur dioxide': 1,
    'density': 0.00025,
    'alcohol': 0.1
}



@app.callback(
    Output('distplot-graph', 'figure'),
    [Input('variable-dropdown', 'value')]
)
def update_distplot(selected_variable):
    colors = ['#1c4e80']
    fig = ff.create_distplot([wine_df[selected_variable]], [selected_variable], bin_size=bin_sizes[selected_variable], colors=colors)

    for trace in fig['data']:
        trace.showlegend = False

    fig.update_layout(
        title=f"{selected_variable} distribution",
        xaxis_title=selected_variable,
        yaxis_title="Density",
        font=dict(size=12),
        plot_bgcolor='#e7cfb7', 
        paper_bgcolor='#e7cfb7'  
    )
    return fig



@app.callback(
    Output('violinplot-graph', 'figure'),
    [Input('variable-dropdown', 'value')]
)
def update_violinplot(selected_variable):

    color_discrete_map = {
        3: '#003f5c',
        4: '#58508d',
        5: '#bc5090',
        6: '#1496bb',
        7: '#b9375e',
        8: '#a46d87'
    }

    fig = px.violin(wine_df, x='quality', y=selected_variable, box=True, points=False, color="quality",
                    color_discrete_map=color_discrete_map,
                    category_orders={"quality": sorted(wine_df['quality'].unique())})

    for trace in fig.data:
        trace.update(showlegend=False)

    fig.update_layout(
        title=f"{selected_variable} vs quality",
        xaxis_title="Quality",
        yaxis_title=selected_variable,
        legend_title="Quality",
        font=dict(size=12),
        plot_bgcolor='#e7cfb7', 
        paper_bgcolor='#e7cfb7' 
    )
    return fig



@app.callback(
    Output('description-div', 'children'),
    [Input('variable-dropdown', 'value')]
)
def update_description(selected_value):
    description = {
        'fixed acidity': "The intricate tapestry of wine chemistry, includes a symphony of mainly fixed acids: tartaric, malic, citric, and succinic. These acids contribute to the overall taste profile, with variations in intensity. Notably, higher fixed acidity levels are often associated with higher quality wines. Analysis in the boxplots suggests that premium wines typically exhibit fixed acidity levels ranging from 8g/L to 13g/L. This range ensures a balanced acidity that enhances the wine's flavor complexity and longevity, a hallmark of superior quality.",
        'volatile acidity': "The tangy, acidic taste that gives wine its liveliness is volatile acidity. Among these, acetic acid is the most dominant, its familiar association with the pungent tang of vinegar marks it as a less desirable trait. Acetic acid stands as the primary culprit, typically resulting from bacterial invasion during the winemaking process. Within the range of 0.2 to 0.4 g/L, volatile acidity poses minimal threat to a wine's quality. However, beyond these thresholds, the boxplots present that volatile acidity manifests as a glaring flaw. Optimal quality wines maintain VA levels averaging between 0.3 to 0.5g/L, ensuring a harmonious balance free from excessive sourness.",
        'citric acid': "In the spectrum of wine acids, citric acid emerges as a relatively uncommon presence compared to its counterparts. This weak organic compound, abundant in citrus fruits like oranges and limes, holds a minor presence in grapes yet contributes significantly to their total acid content, comprising about 5%. During fermentation, winemakers often employ citric acid as a supplement to enhance acidity, particularly in grapes cultivated in warmer climates. In premium wines, citric acid adds a fresh feel, with levels typically hovering between 0.3 to 0.6 g/L, ensuring a balanced acidity conducive to quality and complexity.",
        'residual sugar': "In a finished wine, residual sugar refers to the sugars left unfermented. This residual sugar directly impacts the wine's sweetness and determines specific labeling terms. For instance, a wine with more than 45 g/L is categorized as ‘sweet wine’. Residual sugar can also enhance the flavor profile of a plain wine and mellow the sharpness of acidic ones. However, it poses a potential risk to wine stability, potentially leading to re-fermentation in the bottle. This phenomenon can result in unwanted flavors and gas production. While very low-quality wines often have high residual sugar levels, it is generally advisable to maintain levels below 2.5g/L to preserve overall quality.",
        'chlorides':"Chlorides, found in wine as a measure of salt content, contribute to both acidity and flavor modulation. Winemakers carefully monitor chloride levels to ensure optimal taste profiles and acidity balances in their wines. Analysis of the dataset shows an inverse correlation between chlorides and red wine quality, implying that better wines tend to have lower chloride levels. In addition, chlorides play a crucial part in preserving the wine's stability over time, which is something winemakers carefully consider. Based on this analysis, an optimal chloride concentration ranging from 75 mg/L to 80 mg/L emerges as favorable for top-tier wines, ensuring a delicate balance that enhances overall quality and taste.",
        'free sulfur dioxide': "Free sulfur dioxide (SO2) in wine refers to the portion of sulfur dioxide that is not bound to other compounds in the wine, such as sugars or proteins. Sulfur dioxide is commonly added to wine as a preservative to prevent oxidation and microbial spoilage. The free form of sulfur dioxide, on the other hand, is the active component responsible for these protective effects. Winemakers measure and manage free SO2 levels very carefully, as they directly impact the wine's stability, shelf life, and sensory characteristics. Proper management of free SO2 ensures the wine remains fresh, vibrant, and free from off-flavors or aromas caused by oxidation or microbial contamination.",
        'total sulfur dioxide': "Total sulfur dioxide (SO2) in wine refers to the sum of both the free and bound forms of sulfur dioxide present in the wine. Free sulfur dioxide is the portion that is not chemically bound to other compounds, while bound sulfur dioxide is combined with various wine components such as sugars, pigments, or proteins. Both forms contribute to the overall sulfur dioxide content, which serves as a preservative in wine, protecting it from oxidation and microbial spoilage. Winemakers monitor total SO2 levels to ensure they comply with regulatory limits and to maintain the wine's stability and quality throughout its production and aging process.",
        'density':"The density of wine serves as a key indicator of its quality, with a noticeable inverse correlation that is often observed. This relationship is primarily attributed to the deliberate addition of sugar, alcohol, and other ingredients during the winemaking process, intended to elevate the wine's flavor profile and complexity. Consequently, these supplementary components typically lead to a reduction in the wine's density. Thus, aiming for a mean density around 0.996435 g/cc not only signifies meticulous craftsmanship but also ensures a high-quality product that embodies balance and refinement, captivating discerning palates with its depth and character.",
        'pH': "A measure of acidity or alkalinity, influences a wine’s quality and aging potential. High acidity, as revealed in our analysis, is a defining trait of premium wine, enhancing its character for better aging. Through the boxplots we see that optimal pH levels typically lie between 3.20-3.30. This range ensures a balance that accentuates crispness and tartness over a smoother, rounder profile which is associated with lower acidity. This carefully calibrated pH not only enriches flavor complexity but also fosters microbial stability, vital for preserving the wine's integrity and enhancing its overall quality.",
        'sulphates':"Also known as sulfur dioxide (SO2), sulphates serve as essential preservatives in winemaking, safeguarding against oxidation and microbial spoilage to maintain freshness and flavor. Their presence must be carefully managed, as excessive levels can impart undesirable aromas and flavors causing consumers to experience adverse reactions such as headaches or allergic responses. Wine usually contains sulfites within a range of 5 mg/L to 200 mg/L, with well-crafted dry red wines typically hovering around 50 mg/L. From analysis shown to the right, it's evident that high-quality wines strike a delicate balance, ensuring sulfite levels enhance preservation without compromising taste or consumer health.",
        'alcohol':"As a key feature in wine, alcohol content plays a significant role in determining wine quality. Optimal alcohol levels contribute to a wine's balance, enhancing its flavor complexity and structure. However, excessively high alcohol content can overwhelm the wine's other characteristics, leading to a lack of harmony and finesse. Conversely, wines with too low alcohol levels may lack depth and intensity. Therefore, achieving the right balance of alcohol is crucial for producing wines of exceptional quality, ensuring that it complements rather than dominates the overall sensory profile. Therefore, considering this dataset, the graphs show that alcohol level should be at 10.4% to 13.4% range for a high quality wine.",
        'quality':"Quality assessment of red wine typically involves evaluating various sensory attributes, structural components, and overall balance on a scale of 1 to 10, considering factors such as aroma complexity, flavor intensity, and potential for aging. A high-quality red wine boasts a rich bouquet, harmonious flavors, well-integrated tannins, a lingering finish, and precise texture, offering a memorable drinking experience worthy of appreciation."
    }
    return html.P(description.get(selected_value, "Select a variable to see its description."))

@app.callback(
    Output('feature-importance-graph', 'figure'),
    [Input('feature-importance-graph', 'id')]
)
def update_graph(dummy_input):
    fig = plot_feature_importance(wine_df, target_column)
    return fig



@app.callback(
    Output('output-container', 'children'),
    [Input('fixed acidity', 'value'),
     Input('citric acid', 'value'),
     Input('chlorides', 'value'),
     Input('total sulfur dioxide', 'value'),
     Input('pH', 'value'),
     Input('alcohol', 'value'),
     Input('volatile acidity', 'value'),
     Input('residual sugar', 'value'),
     Input('free sulfur dioxide', 'value'),
     Input('density', 'value'),
     Input('sulphates', 'value')]
)
def evaluate_wine_quality(fixed_acidity, citric_acid, chlorides, total_sulfur_dioxide, pH, alcohol, volatile_acidity,
                          residual_sugar, free_sulfur_dioxide, density, sulphates):
    input_data = pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol]
    }) 


                              
wine = pd.read_csv("https://raw.githubusercontent.com/EciEca/datathon24/main/winequality-red.csv")
bins = (2, 5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=group_names)
label_quality = LabelEncoder()
wine['quality'] = label_quality.fit_transform(wine['quality'])
X = wine.drop('quality', axis=1)
y = wine['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

input_data = sc.transform(input_data)
prediction = rfc.predict(input_data)[0]

if prediction == 1: 
    return html.Div("good wine", style={'color': '#6a994e', 'text-align': 'center', 'padding': '15px', 'font-size': '20px', 'border': '1px solid #6a994e', 'display': 'inline-block', 'width': 'fit-content', 'margin': 'auto'})
else: 
    return html.Div("poor wine", style={'color': '#b9375e', 'text-align': 'center', 'padding': '15px', 'font-size': '20px', 'border': '2px solid #b9375e', 'display': 'inline-block', 'width': 'fit-content', 'margin': 'auto'})



if __name__ == '__main__':
    app.run_server(debug=True, port=2345)
