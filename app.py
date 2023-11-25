import streamlit as st 
import numpy as np
import pandas as pd
import altair as alt
import joblib

st.set_page_config(page_title = "Home Price Index", page_icon = "🏘")

st.title("Factors Impacting Home Price Trends")
st.divider()
data = pd.read_csv("data_cleaned/house_price_index.csv", index_col='date')
tab1, tab2, tab3= st.tabs(['Home Price Index', 'Factors Analysis', 'About the Project'])

with tab1:
    st.info("Population, Producer Price Index (Cement), Producer Price Index (Concrete Bricks), Unemployment Rate, Personal Income and Gross Domestic Product are the factors impacting Home Prices.", icon = "ℹ️")
    st.write("Utilize the slider to adjust different factors and observe their influence on the Home Price Index.")
    col1, col2 = st.columns([2, 2], gap = "large")
    
    with col1:
        st.divider()
        st.subheader("User Input")
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
    st.write("Utilizing the Random Forest Regression Algorithm with the Gini criterion, we conducted a comprehensive feature importance analysis, achieving an impressive R2 Score of 99.48%. The R2 Score indicates the model's ability to explain the variance in the Home Price Index using the selected features. The feature importance values showcase the significance of each factor in influencing the predictive accuracy of the model.")
    st.write(" ")
    df_feature = pd.read_csv("Model/feature_importance.csv")
    corr_chart = alt.Chart(df_feature, title = 'Feature Importance in Predicting Home Prices').mark_bar().encode(
            alt.Y("Feature", title = " ").sort("-x"),
            alt.X("Importance_Value:Q", title = "Importance Value"),
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
            Explore the factors impacting US home prices over the last two decades through our data-driven analysis. Leveraging datasets from fred.stlouisfed.org, we've considered key factors such as population, unemployment rate, GDP, and more to understand their influence on the S&P Case Schiller Home Price Index.
            
             ### Key Factors
            Population, Producer Price Index (Cement & Concrete, Bricks), Unemployment Rate, Personal Income, GDP, Construction Employment, Labor Force Participation Rate, Monthly Supply of New Houses, Federal Funds Effective Rate, and Median Consumer Price Index.
            
             ### Data Collection
            Our datasets span 20 years, sourced from fred.stlouisfed.org, ensuring a comprehensive view of long-term trends.
            
             ### Data Science Model
            Using advanced analytics, our model uncovers patterns and relationships, shedding light on how these factors impact home prices.
            
             ### Why It Matters
            For homeowners, buyers, policymakers, and investors, this analysis offers valuable insights into the forces driving real estate dynamics. Explore our web app for interactive visualizations and detailed analyses. Uncover the story behind the numbers, empowering informed decisions in the ever-evolving US housing landscape.
             
             Explore the code on [GitHub](https://github.com/VivekAgrawl/USA_Home_price_Webapp.git) to understand how our data-driven analysis unveils insights into the factors shaping the US housing market.
             """)

st.write("---")

st.write("""#### About Me""")
st.write(" ")
st.write("""
            I'm Vivek Agrawal, a passionate data enthusiast skilled in data mining, data visualization, and machine learning.
                 
            [Portfolio](https://www.vivekagrawal.space) | [Linkedin](https://www.linkedin.com/in/ivivekagrawal) | [Github](https://github.com/VivekAgrawl) | [Kaggle](https://www.kaggle.com/atom1991)
        """)