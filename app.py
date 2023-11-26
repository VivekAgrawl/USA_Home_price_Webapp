import streamlit as st 
import numpy as np
import pandas as pd
import altair as alt
import joblib

st.set_page_config(page_title = "Home Price Index", page_icon = "🏘")

st.title("🏡 Factors Impacting Home Price Trends 📈")
st.write("")
st.divider()
data = pd.read_csv("data_cleaned/house_price_index.csv", index_col='date')
tab1, tab2, tab3= st.tabs(['🏡 Home Price Index', '🔍 Factor Analysis', '📑 About the Project'])

with tab1:
    st.write("Exploring two decades of US home price trends, our data-driven analysis unveils key influencers. Factors such as Population, Producer Price Index (Cement), Producer Price Index (Concrete Bricks), Unemployment Rate, Personal Income, and Gross Domestic Product emerge as significant drivers of home prices. ")
    st.write("Use the slider to manipulate these variables and observe their direct impact on the Home Price Index. ")
    col1, col2 = st.columns([2, 2], gap = "large")
    
    with col1:
        st.divider()
        st.subheader("User Input")
        population = st.slider("**Population (Thousands)**", 290000, 340000, 317270)
        PPI_cement = st.slider("**Producer Price index (Cement)**", 149.40, 332.00, 208.00, help="The Producer Price Index (PPI) measures the average change over time in the selling prices received by domestic producers for their output.")
        PPI_concrete = st.slider("**Producer Price Index (Concrete Brick)**", 154.00, 337.00, 201.00, help="The Producer Price Index (PPI) measures the average change over time in the selling prices received by domestic producers for their output.")
        unemployment = st.slider("**Unemployment Rate (%)**", 3.40, 14.70, 5.30, help = "The unemployment rate represents the number of unemployed as a percentage of the labor force.")
        income = st.slider("**Personal Income (Billions of Dollars)**", 9500.00, 25000.00, 14000.00, help = "Personal income is the income that persons receive in return for their provision of labor, land, and capital used in current production and the net current transfer payments that they receive from business and from government.")
        GDP = st.slider("**Gross Domestic Product (Billions of Dollars)**", 11500.00, 28000.00, 17000.00, help = "Gross domestic product (GDP) is the market value of the goods and services produced by labor and property located in the United States.")
        emp_cons = st.slider("**All Employees, Construction (Thousands of Persons)**", 5420.00, 8000.00, 7002.00, help = "Construction employees in the construction sector include: Working supervisors, qualified craft workers, mechanics, apprentices, helpers, laborers, and so forth, engaged in new work, alterations, demolition, repair, maintenance, and the like, whether working at the site of construction or in shops or yards at jobs (such as precutting and preassembling) ordinarily performed by members of the construction trades.")
        fed_funds = st.slider("**Federal Funds Effective Rate (%)**", 0.00, 6.00, 0.40, help = "The rate that the borrowing institution pays to the lending institution is determined between the two banks; the weighted average rate for all of these types of negotiations is called the effective federal funds rate.")
        houses_supply = st.slider("**Monthly Supply of New Houses**", 3.00, 12.50, 5.65, help = "The months' supply is the ratio of new houses for sale to new houses sold.")
        lab_force = st.slider("**Labor Force Participation Rate (%)**", 60.00, 67.00, 63.30, help = 'The Labor Force Participation Rate is defined by the Current Population Survey (CPS) as “the number of people in the labor force as a percentage of the civilian noninstitutional population […] the participation rate is the percentage of the population that is either working or actively looking for work."')
        median_CPI = st.slider("**Median Consumer Price Index**", -0.30, 8.50, 2.45, help = "Median Consumer Price Index (CPI) is a measure of core inflation calculated the Federal Reserve Bank of Cleveland and the Ohio State University.")
        df = {'emp_cons' : emp_cons, 'fed_funds': fed_funds, 'GDP': GDP, 'houses_supply': houses_supply, "income": income, "lab_force": lab_force, "median_CPI": median_CPI, "population": population, "PPI_cement": PPI_cement, "PPI_concrete": PPI_concrete, "unemployment": unemployment}
        df = pd.DataFrame(df, index = [0])
        

    with col2:
        st.divider()
        st.subheader("Home Price Index")
        model = joblib.load("code/model.pkl")
        predictions = model.predict(df)
        st.success(f"**Prediction:** {predictions[0]:.2f}")

        # ------HPI Histogram-------------
        base = alt.Chart(data)
        bar = base.mark_bar(color = "steelblue").encode(
            alt.X('HPI:Q', title = "Home Price Index").bin(),
            alt.Y('count()', title = 'Frequency')
        )
        rule = base.mark_rule(color='#ff4b4b').encode(
            x=alt.datum(predictions[0]),
            size=alt.value(3)
        )
        st.altair_chart(bar+rule, use_container_width=True)

with tab2:
    st.write("Implemented the Random Forest Regression Algorithm for a comprehensive feature importance analysis, achieving a notable R2 Score of 99.48%. This score indicates the model's effectiveness in explaining Home Price Index variance with selected features. Explore feature importance values to understand each factor's impact on enhancing predictive accuracy.")
    st.write(" ")
    df_feature = pd.read_csv("code/feature_importance.csv")
    corr_chart = alt.Chart(df_feature, title = 'Feature Importance in Predicting Home Prices').mark_bar().encode(
            alt.Y("Feature", title = " ").sort("-x"),
            alt.X("Importance_Value:Q", title = "Feature Importance Score"),
            color= alt.value("steelblue")
        ).properties(height = 400)
    st.altair_chart(corr_chart, use_container_width=True)

    st.write("""
        ### Key Insights
            
        ##### Population (Importance: 28.64%)

        Population emerges as the most influential factor, contributing significantly to the predictive power of the model. Changes in population dynamics evidently play a crucial role in determining home prices.
            
        ##### PPI (Cement) and PPI (Concrete)

        Producer Price Indices for Cement and Concrete hold substantial importance, collectively contributing 39.72% to the model's accuracy. This emphasizes the impact of construction material prices on home prices.
                     
        ##### Unemployment and Economic Indicators

        Unemployment, Personal Income, and GDP collectively contribute 29.49% to the model's accuracy. The model recognizes the importance of labor market dynamics and broader economic indicators in predicting home prices.
                     
        ##### Other Factors

        Employees in Construction, Labor Force, Houses Supply, Federal Funds, and Median CPI, while less influential individually, still contribute to the overall accuracy of the model.        
        """)
    
with tab3:
    st.write("""
            ### Project Overview
            Investigate the factors influencing US home prices over the last two decades through a data-driven analysis. Utilizing datasets from [fred.stlouisfed.org](https://fred.stlouisfed.org/), key factors are considered to understand their impact on the S&P Case Schiller Home Price Index.
            
             ### Key Factors
            Population, Producer Price Index (Cement & Concrete, Bricks), Unemployment Rate, Personal Income, GDP, Construction Employment, Labor Force Participation Rate, Monthly Supply of New Houses, Federal Funds Effective Rate, and Median Consumer Price Index.
            
             ### Data Science Model
            Using advanced analytics, our model uncovers patterns and relationships, shedding light on how these factors impact home prices.
            
             ### Why It Matters
            For homeowners, buyers, policymakers, and investors, this analysis offers valuable insights into the forces driving real estate dynamics. Explore our web app for interactive visualizations and detailed analyses. Uncover the story behind the numbers, empowering informed decisions in the ever-evolving US housing landscape.
             
            Explore the code implementation on [GitHub](https://github.com/VivekAgrawl/USA_Home_price_Webapp.git) to understand how our data-driven analysis unveils insights into the factors shaping the US housing market.
             """)

st.write("---")

st.write("""#### About Me""")
st.write(" ")
st.write("""
            I'm Vivek Agrawal, a passionate data enthusiast skilled in data mining, data visualization, and machine learning.
                 
            [Portfolio](https://www.vivekagrawal.space) | [Linkedin](https://www.linkedin.com/in/ivivekagrawal) | [Github](https://github.com/VivekAgrawl) | [Kaggle](https://www.kaggle.com/atom1991)
        """)