# importy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import statsmodels.api as sm
from scipy.stats import pearsonr, f_oneway, stats, shapiro, ttest_ind, levene, mannwhitneyu, spearmanr
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from statsmodels.formula.api import ols

# wyłączenie domyślnego szablonu w Plotly
pio.templates.default = None

# ustawienie szerokości strony
st.set_page_config(layout="wide")

# wczytanie danych
df = pd.read_csv('heart_disease_uci.csv')

# zastępowanie wartości NaN medianą (tylko dla kolumn numerycznych)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# przekształcenie kolumny 'sex' na wartości numeryczne (binarne)
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

# przekształcenie kolumn kategorycznych na numeryczne
df = pd.get_dummies(df, columns=['restecg', 'thal'], drop_first=True)

# utworzenie zmiennej docelowej na potrzeby modeli predykcyjnych
df['target'] = (df['num'] > 0).astype(int)  # 1 dla choroby, 0 dla braku choroby

# tytuł całej analizy (dashboardu)
st.title("Analiza danych dotyczących chorób serca")

# instrukcja do obsługi menu
st.write("Wybierz sekcję z menu, aby zobaczyć szczegóły.")

# przyciski nawigacyjne
menu = st.radio("Menu", ["Start", "Statystyki", "Modele predykcyjne"], horizontal=True, label_visibility="collapsed")

# opis kolumn do wyświetlenia w dataframe
column_descriptions = {
    "id": "Identyfikator pacjenta",
    "age": "Wiek pacjenta (w latach)",
    "sex": "Płeć pacjenta",
    "dataset": "Zbiór, z którego pochodzą dane",
    "cp": "Typ bólu w klatce piersiowej",
    "trestbps": "Spoczynkowe ciśnienie krwi (w mm Hg)",
    "chol": "Stężenie cholesterolu w surowicy (mg/dl)",
    "fbs": "Poziom cukru we krwi na czczo (> 120 mg/dl)",
    "restecg": "Wyniki elektrokardiografii spoczynkowej",
    "thalch": "Maksymalne osiągnięte tętno",
    "exang": "Wystąpienie dławicy piersiowej podczas wysiłku",
    "oldpeak": "Depresja odcinka ST w porównaniu do spoczynkowego EKG",
    "slope": "Kształt krzywej ST podczas wysiłku",
    "ca": "Liczba głównych naczyń krwionośnych widocznych w fluoroskopii",
    "thal": "Wyniki testu talasemiowego",
    "num": "Wskaźnik obecności choroby serca"
}

# stworzenie dataframe z opisami kolumn
columns_df = pd.DataFrame(list(column_descriptions.items()), columns=["Kolumna", "Opis"])

# tekst objaśniający całą analizę
description_text = """
Analiza obejmuje różne statystyki, które mogą pomóc w lepszym zrozumieniu oraz przewidywaniach dotyczących chorób serca. Wnioski mogą być przydatne w zrozumieniu struktury demograficznej badanej grupy, czy zbadaniu potencjalnego związku między wiekiem czy płcią a występowaniem choroby serca. 
Analiza została przeprowadzona na zbiorze UCI Heart Disease, który obejmuje 920 rekordów i jest popularnym, darmowym zbiorem ćwiczeniowym.

W pierwszej części analizy przedstawiono podstawowe statystyki, takie jak m.in. udział kobiet i mężczyzn, jak również zaprezentowano rozkład wieku dla poszczególnych płci. Ponadto przedstawiono wykresy skumulowane gęstości rozkładu wartości ciśnienia tętniczego skurczowego i stężenia cholesterolu z podziałem na płeć.

Następnie przeprowadzono także testy normalności i jednorodności wariancji (Shapiro-Wilka i Levene'a) wobec zmiennych trestbps (spoczynkowe ciśnienie tętnicze) i chol (cholesterol), które wykazały, że obie zmienne nie spełniają założeń normalności rozkładu, a zmienna chol nie spełnia również jednorodności wariancji.
Wobec tego zdecydowano się przeprowadzić test nieparametryczny Manna-Whitneya, który wykazał istotną różnicę w poziomie cholesterolu między płciami, ale brak istotnej różnicy dla ciśnienia tętniczego.
Po zidentyfikowaniu istotnych różnic w poziomie cholesterolu między płciami za pomocą testu Manna-Whitneya, przeprowadzono analizę korelacji Spearmana celem sprawdzenia czy istnieje monotoniczna zależność między cholesterolem a ciśnieniem (czy wzrost lub spadek cholesterolu wpływa na wzrost lub spadek ciśnienie krwi) dla poszczególnych grup (płci). 
Wyniki tej analizy pokazały, że zależność mięzy stężeniem cholesterolu a ciśnieniem krwi jest bardzo słaba (współczynnik korelacji Spearmana wynosi 0.0986).
Ponadto p-wartość na poziomie 0.0027 pokazuje, że istnieje istotny statystycznie, ale słaby związek między obiema zmiennymi. Oznacza to, że związek między tymi zmiennymi od strony praktycznej nie ma dużego znaczenia. 

W drugiej części niniejszej analizy porównano 5 modeli predykcyjnych, które reprezentują różne podejścia do klasyfikacji. Każdy ma unikalne właściwości, które moją wpływać na skuteczność predykcji w zależności od rodzaju danych. Zwłaszcza modele regresji logistycznej, Random Forest czy Gradient Boosting są powszechnie stosowane w klasyfikacji. Modele różnią się między sobą pod względem optymalizacji czy interpretowalności. Dzięki ich porównaniu można lepiej zrozumieć, który model najlepiej radzi sobie z przewidywaniem wystąpienia choroby serca na podstawie wykorzystanego zbioru danych.
W tym porównaniu najlepszym modelem okazał się Gradient Boosting, osiągając najwyższe wyniki we wszystkich kluczowych metrykach, zwłaszcza pod względem dokładności, czułości i F1-score dla klasy 1. Model ten najlepiej identyfikuje przypadki chorobowe, jest skuteczny w przewidywaniu i rzadko pomija przypadki choroby.
Drugim najlepszym modelem okazał się Random Forest, który również osiągnął wysokie wyniki i jest zbliżony do Gradient Boosting. Random Forest jest bardzo skuteczny w wykrywaniu przypadków chorobowych, z dobrą precyzją i czułością, choć jego dokładność jest nieco niższa niż Gradient Boosting.
Umiarkowaną skuteczność wykazał model regresji logistycznej (Logistic Regression), a najsłabsze okazały się SVM i K-Nearest Neighbors, zwłaszcza pod kątem dokładności i czułości, przez co są mniej odpowiednie dla analizy tego zbioru danych bez odpowiedniej optymalizacji parametrów.
Na końcu zamieszczono również wykresy przedstawiające istotność cech dla dwóch najlepszych modeli - Gradient Boosting i Random Forest, które pozwalają zrozumieć, które zmienne mają największy wpływ na przewidywania modelu.

Wykresy wykorzystane w analizie:

1. histogram 
2. wykres rozrzutu (scatter plot, wykres punktowy) 
3. wykres słupkowy (bar plot) 
4. skumulowany wykres gęstości

Ponadto do modelu predykcyjnego:

5. wykres radarowy
6. macierz błędów
7. raport klasyfikacji (tabela tekstowa)

Dzięki zastosowaniu tych różnych typów wykresów oraz tabeli można uzyskać pełny obraz analizowanych danych, zarówno w aspekcie statystycznym, jak i wizualnym, co ułatwia wyciąganie wniosków i podejmowanie decyzji w ramach projektu.

Wykorzystane biblioteki i funkcje:

a)	obliczenia

1.	pandas 
2.	statsmodels
3.	scipy

b)	wizualizacje

4.	matplotlib
5.	seaborn
6.	plotly

c)	modele predykcyjne

7.	scikit-learn

Aplikacja webowa utworzona za pomocą frameworka Streamlit.

"""

# wyświetlanie poszczególnych części analizy na podstawie wyboru w menu
if menu == "Start":
    st.subheader("Raport został stworzony na podstawie zbioru danych UCI Heart Disease, zawierającego 920 rekordów. Analiza składa się z podstawowych statystyk oraz modeli predykcyjnych.")
    # wyświetlenie mojego tekstu (opisu analizy)
    st.write(description_text)
    # wyświetlenie kolumn z objaśnieniami w dataframe
    st.write("Opis kolumn w zbiorze UCI Heart Disease")
    st.dataframe(columns_df, use_container_width=True)
    # wyświetlenie infomacji o charakterze pracy
    st.write("Analiza ma charekter edukacyjny i została stworzona wyłącznie na potrzeby budowy portfolio, a uzyskane wyniki nie zastępują profesjonalnej diagnozy medycznej.")

elif menu == "Statystyki":
    # sekcja z podstawowymi statystykami
    st.subheader("Statystyki")

    # obliczenia do podstawowych statystyk
    total_records = df.shape[0]
    num_men = df['sex'].sum()
    num_women = total_records - num_men
    percent_men = (num_men / total_records) * 100
    percent_women = (num_women / total_records) * 100

    # CSS do tworzenia ramek dla statystyk
    st.markdown(
        """
        <style>
        .stat-box {
            border: 2px solid #ddd;
            padding: 20px;
            margin: 10px;
            border-radius: 10px;
            background-color: #f9f9f9;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .stat-title {
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }
        .stat-value {
            font-size: 24px;
            color: #e6005c;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ustawienia globalne dla czcionki
    pio.templates["my_custom_template"] = pio.templates["plotly"]
    pio.templates["my_custom_template"].layout.font.family = "Arial"
    pio.templates["my_custom_template"].layout.font.size = 14
    pio.templates["my_custom_template"].layout.font.color = "black"

    # ustawienie tego szablonu jako domyślnego
    pio.templates.default = "my_custom_template"

    # wyświetlanie statystyk w formie ramek
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class="stat-box">
                <div class="stat-title">Liczba przeanalizowanych rekordów</div>
                <div class="stat-value">{total_records}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="stat-box">
                <div class="stat-title">Liczba kobiet</div>
                <div class="stat-value">{num_women} ({percent_women:.2f}%)</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="stat-box">
                <div class="stat-title">Liczba mężczyzn</div>
                <div class="stat-value">{num_men} ({percent_men:.2f}%)</div>
            </div>
        """, unsafe_allow_html=True)

    # wykresy dla podstawowych statystyk
    # obliacznie mediany wieku dla każdej płci
    median_age_men = df[df['sex'] == 1]['age'].median()
    median_age_women = df[df['sex'] == 0]['age'].median()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Rozkład wieku mężczyzn")
        fig1 = px.histogram(df[df['sex'] == 1], x='age', nbins=20, title="Histogram rozkładu wieku mężczyzn", labels={'age': 'Wiek'}, color_discrete_sequence=['#ff99bb'])
        fig1.update_layout(bargap=0.1, yaxis_title='Liczba')
        fig1.add_vline(x=median_age_men, line=dict(color='blue', dash='dash'),
                       annotation_text=f"Mediana: {median_age_men}", annotation_position="top right")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Rozkład wieku kobiet")
        fig2 = px.histogram(df[df['sex'] == 0], x='age', nbins=20, title="Histogram rozkładu wieku kobiet", labels={'age': 'Wiek'}, color_discrete_sequence=['#ff80aa'])
        fig2.update_layout(bargap=0.1, yaxis_title='Liczba')
        fig2.add_vline(x=median_age_women, line=dict(color='purple', dash='dash'),
                       annotation_text=f"Mediana: {median_age_women}", annotation_position="top right")
        st.plotly_chart(fig2, use_container_width=True)

    # ustawienie kolorków dla badanych grup (kobiet i mężczyzn)
    colors = {
        'male': '#ff99bb',  # Jasny róż dla mężczyzn
        'female': '#ff80aa'  # Ciemny róż dla kobiet
    }

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Rozkład ciśnienia tętniczego z podziałem na płeć")
        fig_trestbps = go.Figure()

        fig_trestbps.add_trace(go.Histogram(
            x=df[df['sex'] == 1]['trestbps'],
            histnorm='density',
            name='Mężczyźni',
            opacity=0.6,
            marker_color=colors['male'],
            cumulative_enabled=True
        ))

        fig_trestbps.add_trace(go.Histogram(
            x=df[df['sex'] == 0]['trestbps'],
            histnorm='density',
            name='Kobiety',
            opacity=0.6,
            marker_color=colors['female'],
            cumulative_enabled=True
        ))

        fig_trestbps.update_layout(
            title='Skumulowany rozkład ciśnienia tętniczego z podziałem na płeć',
            xaxis_title='Spoczynkowe ciśnienie krwi (trestbps)',
            yaxis_title='Gęstość',
            barmode='overlay'
        )

        st.plotly_chart(fig_trestbps, use_container_width=True)

    with col4:
        st.subheader("Rozkład poziomu cholesterolu z podziałem na płeć")
        fig_chol = go.Figure()

        fig_chol.add_trace(go.Histogram(
            x=df[df['sex'] == 1]['chol'],
            histnorm='density',
            name='Mężczyźni',
            opacity=0.6,
            marker_color=colors['male'],
            cumulative_enabled=True
        ))

        fig_chol.add_trace(go.Histogram(
            x=df[df['sex'] == 0]['chol'],
            histnorm='density',
            name='Kobiety',
            opacity=0.6,
            marker_color=colors['female'],
            cumulative_enabled=True
        ))

        fig_chol.update_layout(
            title='Skumulowany rozkład poziomu cholesterolu z podziałem na płeć',
            xaxis_title='Poziom cholesterolu (chol)',
            yaxis_title='Gęstość',
            barmode='overlay'
        )

        st.plotly_chart(fig_chol, use_container_width=True)

    # testy statystyczne
    st.subheader(
        "Testy normalności i jednorodności wariancji dla zmiennych cholesterol oraz ciśnienie w odniesieniu do grup według płci")

    # testy Shapiro-Wilka
    chol_men = df[df['sex'] == 1]['chol']
    chol_women = df[df['sex'] == 0]['chol']
    trestbps_men = df[df['sex'] == 1]['trestbps']
    trestbps_women = df[df['sex'] == 0]['trestbps']

    shapiro_chol_men = shapiro(chol_men)
    shapiro_chol_women = shapiro(chol_women)
    shapiro_trestbps_men = shapiro(trestbps_men)
    shapiro_trestbps_women = shapiro(trestbps_women)

    # test Levene’a
    levene_chol = levene(chol_men, chol_women)
    levene_trestbps = levene(trestbps_men, trestbps_women)

    # wyświetlanie wyników testów statystycznych
    st.subheader("Testy normalności Shapiro-Wilka")
    st.write(
        f"Cholesterol - Mężczyźni: statystyka = {shapiro_chol_men.statistic:.4f}, p-wartość = {shapiro_chol_men.pvalue:.4f}")
    st.write(
        f"Cholesterol - Kobiety: statystyka = {shapiro_chol_women.statistic:.4f}, p-wartość = {shapiro_chol_women.pvalue:.4f}")
    st.write(
        f"Ciśnienie tętnicze - Mężczyźni: statystyka = {shapiro_trestbps_men.statistic:.4f}, p-wartość = {shapiro_trestbps_men.pvalue:.4f}")
    st.write(
        f"Ciśnienie tętnicze - Kobiety: statystyka = {shapiro_trestbps_women.statistic:.4f}, p-wartość = {shapiro_trestbps_women.pvalue:.4f}")

    st.subheader("Test jednorodności wariancji Levene’a")
    st.write(f"Cholesterol: statystyka = {levene_chol.statistic:.4f}, p-wartość = {levene_chol.pvalue:.4f}")
    st.write(
        f"Ciśnienie tętnicze: statystyka = {levene_trestbps.statistic:.4f}, p-wartość = {levene_trestbps.pvalue:.4f}")

    st.subheader("Test Manna-Whitneya")
    mannwhitney_chol = mannwhitneyu(chol_men, chol_women, alternative='two-sided')
    mannwhitney_trestbps = mannwhitneyu(trestbps_men, trestbps_women, alternative='two-sided')
    st.write(f"Cholesterol: statystyka = {mannwhitney_chol.statistic:.4f}, p-wartość = {mannwhitney_chol.pvalue:.4f}")
    st.write(
        f"Ciśnienie tętnicze: statystyka = {mannwhitney_trestbps.statistic:.4f}, p-wartość = {mannwhitney_trestbps.pvalue:.4f}")

    # analiza korelacji Spearmana
    st.subheader("Analiza korelacji Spearmana między stężeniem cholesterolu a ciśnieniem krwi")
    st.subheader("Analiza korelacji bez podziału na grupy")
    spearman_corr, spearman_p_value = spearmanr(df['chol'], df['trestbps'])
    st.write(f"Współczynnik korelacji Spearmana: {spearman_corr:.4f}")
    st.write(f"p-wartość: {spearman_p_value:.4f}")

    # wykres punktowy dla korelacji Spearmana
    fig = px.scatter(df, x='chol', y='trestbps', trendline="ols",
                     labels={'chol': 'Cholesterol', 'trestbps': 'Spoczynkowe ciśnienie krwi'},
                     title='Interaktywny wykres cholesterolu i ciśnienia krwi z linią regresji')

    fig.update_layout(width=600, height=400)
    st.plotly_chart(fig, use_container_width=True)

    # podział danych na grupy według płci
    df_men = df[df['sex'] == 1]  # Mężczyźni
    df_women = df[df['sex'] == 0]  # Kobiety

    spearman_corr_men, p_value_men = spearmanr(df_men['chol'], df_men['trestbps'])
    spearman_corr_women, p_value_women = spearmanr(df_women['chol'], df_women['trestbps'])

    # wyświetlenie wyników
    st.subheader("Analiza korelacji z podziałem na płeć")

    st.write(f"**Mężczyźni** - Współczynnik korelacji Spearmana: {spearman_corr_men:.4f}, p-wartość: {p_value_men:.4f}")
    st.write(
        f"**Kobiety** - Współczynnik korelacji Spearmana: {spearman_corr_women:.4f}, p-wartość: {p_value_women:.4f}")


    # wykres dla mężczyzn
    fig_men = px.scatter(df_men, x='chol', y='trestbps', trendline="ols",
                         labels={'chol': 'Cholesterol', 'trestbps': 'Spoczynkowe ciśnienie krwi'},
                         title='Korelacja cholesterolu i ciśnienia krwi - Mężczyźni')
    fig_men.update_traces(marker=dict(color='blue'), selector=dict(mode='markers'))
    fig_men.update_layout(width=600, height=400)

    # wykres dla kobiet
    fig_women = px.scatter(df_women, x='chol', y='trestbps', trendline="ols",
                           labels={'chol': 'Cholesterol', 'trestbps': 'Spoczynkowe ciśnienie krwi'},
                           title='Korelacja cholesterolu i ciśnienia krwi - Kobiety')
    fig_women.update_traces(marker=dict(color='pink'), selector=dict(mode='markers'))
    fig_women.update_layout(width=600, height=400)
    st.plotly_chart(fig_men, use_container_width=True)
    st.plotly_chart(fig_women, use_container_width=True)


# modele predykcyjne
elif menu == "Modele predykcyjne":

    # tworzenie zmiennej docelowej (0 - brak choroby, 1 - wystąpienie choroby)
    df['target'] = (df['num'] > 0).astype(int)

    # wybór odpowiednich kolumn do modelu
    features = ['age', 'sex', 'trestbps', 'chol', 'fbs'] + df.columns[df.columns.str.startswith('restecg')].tolist() + df.columns[df.columns.str.startswith('thal')].tolist()
    X = df[features]
    y = df['target']

    # wypełnianie braków (NaN) medianą dla kolumn numerycznych w X
    X = X.fillna(X.median())

    # podział na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # lista modeli do przetestowania
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
    }

    # przechowywanie wyników modeli
    results = []

    # trenowanie i ocena każdego modelu
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # metryki dla modeli
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc
        })

    # wyświetlanie wyników
    results_df = pd.DataFrame(results).set_index("Model")
    st.subheader("Porównanie modeli: Random Forest, Logistic Regression, Gradient Boosting, SVM, K-Nearest Neighbors")
    st.write(results_df)

    # lista metryk do wykresu słupkowego
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    metrics_data = results_df[metrics].reset_index()

    # wykres słupkowy dla porównania metryk modeli
    st.subheader("Porównanie modeli na podstawie różnych metryk")
    fig_bar = px.bar(metrics_data.melt(id_vars="Model"),
                     x="variable", y="value", color="Model",
                     barmode="group", labels={"variable": "Metryka", "value": "Wartość"},
                     title="Porównanie modeli na podstawie metryk")
    st.plotly_chart(fig_bar, use_container_width=True)

    # wykres radarowy (spider plot)
    st.subheader("Porównanie modeli na wykresie radarowym")
    fig_radar = go.Figure()

    for model_name in results_df.index:
        fig_radar.add_trace(go.Scatterpolar(
            r=results_df.loc[model_name, metrics].values,
            theta=metrics,
            fill='toself',
            name=model_name
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Wykres radarowy dla metryk modeli"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # szczegółowe raporty klasyfikacyjne i macierze konfuzji dla każdego modelu
    st.subheader("Szczegółowe raporty klasyfikacyjne dla każdego modelu")

    for model_name, model in models.items():
        st.write(f"### Model: {model_name}")

        # predykcja dla danego modelu
        y_pred = model.predict(X_test)

        # raport klasyfikacji
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write(f"Raport klasyfikacji dla modelu {model_name}:")
        st.write(report_df)

        # macierz błędów
        cm = confusion_matrix(y_test, y_pred)
        cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="magenta",
                           labels=dict(x="Predykcje", y="Prawdziwe", color="Liczba"),
                           x=['Negatyw', 'Pozytyw'], y=['Negatyw', 'Pozytyw'])
        cm_fig.update_layout(title=f"Macierz błędów - {model_name}")
        st.plotly_chart(cm_fig, use_container_width=True)

    # przedstawienie istotności cech w dwóch najlepszych modelach
    # tworzenie zmiennej docelowej i wybór cech
    df['target'] = (df['num'] > 0).astype(int)
    features = ['age', 'sex', 'trestbps', 'chol', 'fbs'] + df.columns[df.columns.str.startswith('restecg')].tolist() + \
               df.columns[df.columns.str.startswith('thal')].tolist()
    X = df[features]
    y = df['target']

    # wypełnianie braków medianą i podział na zbiory
    X = X.fillna(X.median())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # trenowanie modelu Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_importances = rf_model.feature_importances_

    # trenowanie modelu Gradient Boosting
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train, y_train)
    gb_importances = gb_model.feature_importances_

    # tworzenie data frame dla istotności cech
    rf_features_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_importances
    }).sort_values(by='Importance', ascending=False)

    gb_features_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': gb_importances
    }).sort_values(by='Importance', ascending=False)

    # ustawienie palety kolorów
    magenta_color = '#cc6699'

    # ustawienia czcionki
    font_family = "Arial"
    font_size = 12

    # wykres istotności cech dla Random Forest
    st.subheader("Istotność cech - Random Forest")
    fig_rf = go.Figure(go.Bar(
        x=rf_features_df['Importance'],
        y=rf_features_df['Feature'],
        orientation='h',
        marker_color=magenta_color
    ))
    fig_rf.update_layout(
        title="Istotność cech w modelu Random Forest",
        xaxis_title="Istotność",
        yaxis_title="Cechy",
        yaxis={'categoryorder': 'total ascending', 'automargin': True},

        font=dict(family=font_family, size=font_size)
    )
    st.plotly_chart(fig_rf, use_container_width=True)

    # wykres istotności cech dla Gradient Boosting
    st.subheader("Istotność cech - Gradient Boosting")
    fig_gb = go.Figure(go.Bar(
        x=gb_features_df['Importance'],
        y=gb_features_df['Feature'],
        orientation='h',
        marker_color=magenta_color
    ))
    fig_gb.update_layout(
        title="Istotność cech w modelu Gradient Boosting",
        xaxis_title="Istotność",
        yaxis_title="Cechy",
        yaxis={'categoryorder': 'total ascending', 'automargin': True},
        font=dict(family=font_family, size=font_size))
    st.plotly_chart(fig_gb, use_container_width=True)