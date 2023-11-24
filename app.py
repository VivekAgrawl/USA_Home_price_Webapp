import streamlit as st 
import numpy as np
import pandas as pd
import altair as alt
import joblib

st.set_page_config(page_title = "Home Price Index", page_icon = "🏘")

st.title("Factors Impacting Home Price Trends")
st.divider()
data = pd.read_csv("data_cleaned/house_price_index.csv", index_col='date')
tab1, tab2, tab3= st.tabs(['Home Price Index', 'Factors Analysis', 'About Data'])

with tab1:
    st.info("Population, Producer Price Index (Cement), Producer Price Index (Concrete Bricks), Unemployement Rate, Personal Income and Gross Domestic Product are the factors impacting Home Prices.", icon = "ℹ️")
    st.write("Utilize the slider to adjust different factors and observe their influence on the Home Price Index.")
    col1, col2 = st.columns([2, 2], gap = "large")
    
    with col1:
        st.divider()
        population = st.slider("Population", 290000, 340000, 317270)
        PPI_cement = st.slider("Producer Price index (Cement)", 149.40, 332.00, 208.00)
        PPI_concrete = st.slider("Producer Price Index (Concrete Brick)", 154.00, 337.00, 201.00)
        unemployment = st.slider("Unemployment Rate", 3.40, 14.70, 5.30)
        income = st.slider("Personal Income", 9500.00, 25000.00, 14000.00)
        GDP = st.slider("Gross Domestic Product", 11500.00, 28000.00, 17000.00)
        emp_cons = st.slider("All Employees, Construction", 5420.00, 8000.00, 7002.00)
        fed_funds = st.slider("Federal Funds Effective Rate", 0.00, 6.00, 0.40)
        houses_supply = st.slider("Monthly Supply of New Houses", 3.00, 12.50, 5.65)
        lab_force = st.slider("Labor Force Participation Rate", 60.00, 67.00, 63.30)
        median_CPI = st.slider("Median Consumer Price Index", -0.30, 8.50, 2.45)
        df = {'emp_cons' : emp_cons, 'fed_funds': fed_funds, 'GDP': GDP, 'houses_supply': houses_supply, "income": income, "lab_force": lab_force, "median_CPI": median_CPI, "population": population, "PPI_cement": PPI_cement, "PPI_concrete": PPI_concrete, "unemployment": unemployment}
        df = pd.DataFrame(df, index = [0])
        

    with col2:
        st.divider()
        st.subheader("Home Price Index")
        model = joblib.load("Model/model.pkl")
        predictions = model.predict(df)
        st.success(f"{predictions[0]:.2f}")

        # ------HPI Histogram-------------
        base = alt.Chart(data)
        bar = base.mark_bar(color = "steelblue").encode(
            alt.X('HPI:Q').bin().axis(None),
            y='count()'
        )
        rule = base.mark_rule(color='#ff4b4b').encode(
            x=alt.datum(predictions[0]),
            size=alt.value(3)
        )
        st.altair_chart(bar+rule, use_container_width=True)

with tab2:
    subtab1, subtab2, subtab3 = st.tabs(['Correlation', 'Principal Component Analysis', 'Feature Importance'])