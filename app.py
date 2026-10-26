import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 House Price Predictor")
st.write( "Some house Data-set is presented and displayed")
st.divider()

st.sidebar.title("🏡 House Predictor")
st.sidebar.write("Machine Learning Project")

st.sidebar.divider()

st.sidebar.info(
    "This application uses Linear Regression "
    "to estimate house prices."
)

st.sidebar.caption("Built with Python,Streamlit,Pandas,numpy,matplotlib And Scikit-Learn")

data = {
    "Area_sqft":[800, 1000, 1200, 1500, 1800,
                2000, 2200, 2500, 2800, 3000],

    "Bedrooms":[2, 2, 3, 3, 3, 4, 4, 4, 5, 5], 

    "Bathrooms":[1, 2, 2, 2, 3, 3, 3, 4, 4, 4],

    "Age_years":[10, 8, 5, 7, 4, 6, 3, 2, 5, 1],

    "Price_PKR":[6000000, 8000000, 10000000,
        12000000, 15000000, 16000000,
        18000000, 22000000, 24000000,28000000]

}
df=pd.DataFrame(data)

st.subheader("📊 House Dataset")
st.dataframe(df,use_container_width=True)

st.subheader("Dataset infromation ")

col1,col2,col3=st.columns(3)

col1.metric("Total House",df.shape[0])
col2.metric("Input features",4)
col3.metric("Output Features",df.shape[1])

st.subheader("Basic Statistics")
st.dataframe(df.describe(),use_container_width=True)


X=df[["Area_sqft","Bedrooms","Bathrooms","Age_years"]]
y=df["Price_PKR"]


col1, col2 = st.columns(2)

with col1:
    st.subheader("Features (X)")
    st.dataframe(X, use_container_width=True)

with col2:
    st.subheader("Target (y)")
    st.dataframe(y, use_container_width=True)

st.success(" Add Data to the system  ")


# exploring the dataset 
st.divider()
st.header("Exploring the dataset and finding Patterns")

st.subheader("1.missing values")
st.dataframe(df.isnull().sum(),use_container_width=True)

st.subheader("Duplicate Rows")
duplicate_count=df.duplicated().sum()
st.write(f"the number of duplicate rows ",{duplicate_count})

st.subheader("3. feature correlation")
correlation=df.corr(numeric_only=True)
st.dataframe(correlation, use_container_width=True)


st.subheader("4. Features vs House Price")

st.bar_chart(
    correlation["Price_PKR"].drop("Price_PKR")
)

# final features and target
st.subheader("5. Final Features and Target")

X = df[["Area_sqft", "Bedrooms", "Bathrooms", "Age_years"]]
y = df["Price_PKR"]

st.write("Input features (X):")
st.write(list(X.columns))

st.write("Target (y):")
st.write(y.name)

st.success(" Data explored!")

# training the model spliting the data
st.divider()
st.header("Training And Testing Data ")

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=40
)

st.subheader("Training set")
st.write("the number of training house",len(X_train))
st.dataframe(X_train,use_container_width=True)

st.subheader("Testing set")
st.write("the number of testing house",len(X_test))
st.dataframe(X_test,use_container_width=True)

st.subheader("Actual values")
col1,col2=st.columns(2)

with col1:
    st.write("training prices")
    st.dataframe(y_train,use_container_width=True)

with col2:
    st.write("testing prices")
    st.dataframe(y_test,use_container_width=True)

st.success(" Data split successfully!")

       
# ---------------------------
# TRAINING OUR FIRST ML MODEL 
# ---------------------------

st.divider()
st.header("Training the Model")

model=LinearRegression()
model.fit(X_train,y_train)
st.success("model trained successfully")

y_predict=model.predict(X_test)

result=pd.DataFrame({
    "Actual Price (PKR) ": y_test.to_numpy(),
    "Predicited Price (PKR)":y_predict.round(0)

})

st.header("Actual VS Predicited Prices")
st.dataframe(result,use_container_width=True)

# EVALUATE MODEL PERFORMACE
mae=mean_absolute_error(y_test,y_predict)
mse=mean_squared_error(y_test,y_predict)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_predict)

st.subheader("📊 Model Evaluation")
col1,col2,col3,col4=st.columns(4)

col1.metric("MAE (PKR)",f"{mae:,.0f}")
col2.metric("MSE ",f"{mse:,.0f}")
col2.metric("RMSE",f"{rmse:,.0f}")
col4.metric("R2 SCORE",f"{r2:.3f}")

st.subheader("Learned Model Coefficients")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

st.dataframe(coefficients, use_container_width=True)
st.write("Intercept:", round(model.intercept_, 2))



#---------------------------------
#ACTUAL VS PREDICITED PRICE CHART
# -------------------------------

st.divider()
st.header("Evaluate Model Predication")

comparison=pd.DataFrame({
    "Actual Price": y_test.to_numpy(),
    "Predicated Price":y_predict

})

st.line_chart(comparison)
comparison["Absolute Error"]=(comparison["Actual Price"] - comparison["Predicated Price"]).abs()

st.subheader("Actual Predicate Price")
st.dataframe(comparison,use_container_width=True)

st.write(
    "Average absolute error (MAE):",
    f"₨{comparison['Absolute Error'].mean():,.0f}"
)


# ==========================================
#  INTERACTIVE PREDICTION INTERFACE
# ==========================================
# ==========================================
# STEP 6.3.1: HOUSE PRICE DASHBOARD
# ==========================================

st.divider()
st.title("🏡 House Price Predictor")
st.caption("Machine Learning Dashboard | Linear Regression")

prediction_tab, performance_tab, insights_tab = st.tabs([
    "🏠 Prediction",
    "📊 Model Performance",
    "🔍 Data Insights"
])

# ------------------------------------------
# TAB 1: PREDICTION
# ------------------------------------------
with prediction_tab:
    st.subheader("Predict a House Price")
    st.write("Enter the property details below.")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            "Area (square feet)",
            min_value=300,
            max_value=10000,
            value=1500,
            step=100
        )

        bedrooms = st.number_input(
            "Number of bedrooms",
            min_value=1,
            max_value=15,
            value=3
        )

    with col2:
        bathrooms = st.number_input(
            "Number of bathrooms",
            min_value=1,
            max_value=10,
            value=2
        )

        age = st.number_input(
            "House age (years)",
            min_value=0,
            max_value=100,
            value=5
        )

    if st.button("🔮 Predict House Price", use_container_width=True):
        new_house = pd.DataFrame({
            "Area_sqft": [area],
            "Bedrooms": [bedrooms],
            "Bathrooms": [bathrooms],
            "Age_years": [age]
        })

        predicted_price = model.predict(new_house)[0]

        st.success("Prediction completed!")
        st.metric(
            "Estimated House Price",
            f"₨{predicted_price:,.0f}"
        )

        st.caption(
            "Educational estimate based on a small sample dataset; "
            "not a reliable real-world market valuation."
        )

# ------------------------------------------
# TAB 2: MODEL PERFORMANCE
# ------------------------------------------
with performance_tab:
    st.subheader("Model Evaluation")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"₨{mae:,.0f}")
    col2.metric("RMSE", f"₨{rmse:,.0f}")
    col3.metric("R² Score", f"{r2:.3f}")

    comparison = pd.DataFrame({
        "Actual Price": y_test.to_numpy(),
        "Predicted Price": y_predict
    })

    comparison["Absolute Error"] = (
        comparison["Actual Price"] -
        comparison["Predicted Price"]
    ).abs()

    st.subheader("Actual vs Predicted Prices")
    st.line_chart(comparison[["Actual Price", "Predicted Price"]])

    st.subheader("Prediction Error Details")
    st.dataframe(comparison, use_container_width=True)

# ------------------------------------------
# TAB 3: DATA INSIGHTS
# ------------------------------------------
with insights_tab:
    st.subheader("Explore the House Dataset")

    st.write("Dataset size:", df.shape[0], "houses")
    st.write("Number of features:", X.shape[1])

    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

    st.subheader("House Price Distribution")
    st.bar_chart(df.set_index("Area_sqft")[["Price_PKR"]])

    st.subheader("Feature Correlations")
    correlation = df.corr(numeric_only=True)
    st.dataframe(correlation, use_container_width=True)

    st.subheader("Feature Correlation with Price")
    st.bar_chart(correlation["Price_PKR"].drop("Price_PKR"))