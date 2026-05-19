import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, r2_score
from sklearn.preprocessing import StandardScaler

# Set page title and icon
st.set_page_config(page_title="Airline Passenger Satisfaction Dataset Modelation", page_icon="✈️")

# Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis", "Model Training and Evaluation", "Make Predictions!", "Recommendations and Insights"])

# Load dataset
df = pd.read_csv('data/cleaned_airline_passenger_satisfaction.csv')

# Home Page
if page == "Home":
    st.title("📊 Airline Passenger Satisfaction Dataset Modelation")
    st.subheader("Welcome to our Airline Passenger Satisfaction dataset modelation app!")
    st.write("""
        This app provides an interactive platform to explore the Airline Passenger Satisfaction dataset. 
        You can visualize the distribution of data, explore relationships between features, and even make predictions on new data!
        Use the sidebar to navigate through the sections.
    """)
    st.image('https://framerusercontent.com/images/schDoamnA5b6R8z6M8CaosoO4sk.jpg?width=2508&height=1672', caption="Airline Customers")
    st.write("Use the sidebar to navigate between different sections.")


# Data Overview
elif page == "Data Overview":
    st.title("🔢 Data Overview")

    st.subheader("About the Data")
    st.write("""
        The Airline Passenger Satisfaction dataset contains {df.shape[0]} samples of airline passenger experience.
        For each experience, the dataset includes {df.shape[1]} columns with costumer information like age of class, experience information like inflight entertainment and on-board_service and general passenger satisfaction.
    """)
    st.image('https://s.yimg.com/lo/mysterio/api/718ed665823f0c664ed7b2a02b3d1127b4ed04e2d7e1f401657261bd87916a9a/lightyear_networkapi/resizefill_w976;quality_80;format_webp/https:%2F%2Fmedia.zenfs.com%2Fen%2Fislands_423%2F8ab0804c7094c3ffb09127e182bf48bc', caption="On-board service")

    # Dataset Display
    st.subheader("Quick Glance at the Data")
    if st.checkbox("Show DataFrame"):
        st.dataframe(df)
    




# Exploratory Data Analysis (EDA)
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.subheader("Select the type of visualization you'd like to explore:")
    eda_type = st.multiselect("Visualization Options", ['Histograms', 'Box Plots', 'Scatterplots', 'Count Plots'])

    encoding_dict = {
        "gender": {0: "Male", 1: "Female"},
        "customer_type": {0: "Disloyal Customer", 1: "Loyal Customer"},
        "type_of_travel": {0: "Business Travel", 1: "Personal Travel"},
        "class": {0: "Business", 1: "Eco", 2: "Eco Plus"},
        "satisfaction": {0: "Neutral or Dissatisfied", 1: "Satisfied"}
    }

    obj_cols = [col for col in encoding_dict.keys() if col in df.columns]
    num_cols = [
        col for col in df.select_dtypes(include='number').columns.tolist()
        if col not in obj_cols
    ]

    def show_encoding_table(column):
        if column in encoding_dict:
            encoding_df = pd.DataFrame({
                "Value": list(encoding_dict[column].keys()),
                "Meaning": list(encoding_dict[column].values())
            })

            st.markdown(f"**Encoding for `{column}`**")
            st.dataframe(encoding_df, hide_index=True, use_container_width=True)

    if 'Histograms' in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for the histogram:", num_cols)

        if h_selected_col:
            chart_title = f"Distribution of {h_selected_col.title().replace('_', ' ')}"

            if st.checkbox("Show by satisfaction"):
                st.plotly_chart(px.histogram(df, x=h_selected_col, color='satisfaction', title=chart_title, barmode='overlay'))
                show_encoding_table("satisfaction")
            else:
                st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title))

    if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        b_selected_col = st.selectbox("Select a numerical column for the box plot:", num_cols)

        if b_selected_col:
            chart_title = f"Distribution of {b_selected_col.title().replace('_', ' ')}"
            st.plotly_chart(px.box(df, x='satisfaction', y=b_selected_col, title=chart_title, color='satisfaction'))
            show_encoding_table("satisfaction")

    if 'Scatterplots' in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        selected_col_x = st.selectbox("Select x-axis variable:", num_cols)
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols)

        if selected_col_x and selected_col_y:
            chart_title = f"{selected_col_x.title().replace('_', ' ')} vs. {selected_col_y.title().replace('_', ' ')}"
            st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, color='satisfaction', title=chart_title))
            show_encoding_table("satisfaction")

    if 'Count Plots' in eda_type:
        st.subheader("Count Plots - Visualizing Categorical Distributions")
        selected_col = st.selectbox("Select a categorical variable:", obj_cols)

        if selected_col:
            chart_title = f'Distribution of {selected_col.title().replace("_", " ")}'
            st.plotly_chart(px.histogram(df, x=selected_col, color='satisfaction', title=chart_title, barmode='group'))

            show_encoding_table(selected_col)

            if selected_col != "satisfaction":
                show_encoding_table("satisfaction")

                
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Duplicate Values", value="0")

    with col2:
        st.metric(label="Encoded Categorical Columns", value="5")

    with col3:
        st.metric(label="Missing Values Replaced", value="310")

    st.markdown("""
    **Data Cleaning Summary**

    - The 5 categorical columns were encoded into numerical values for modeling and correlation analysis.
    - These columns are still treated as categorical variables in the visualizations.
    - The 310 missing values corresponded to `arrival_delay_in_minutes`.
    - Missing values were replaced with the median because the distribution was skewed.
    """)

    
# Model Training and Evaluation Page
elif page == "Model Training and Evaluation":
    st.title("🛠️ Model Training and Evaluation")

    # Sidebar for model selection
    st.sidebar.subheader("Choose a Machine Learning Model")
    model_option = st.sidebar.selectbox("Select a model", ["K-Nearest Neighbors", "Logistic Regression", "Random Forest"])

    # Prepare the data
    X = df.drop(columns = 'satisfaction')
    y = df['satisfaction']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the selected model
    if model_option == "K-Nearest Neighbors":
        k = st.sidebar.slider("Select the number of neighbors (k)", min_value=1, max_value=20, value=3)
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()

    # Train the model on the scaled data
    model.fit(X_train_scaled, y_train)

    # Display training and test accuracy
    st.write(f"**Model Selected: {model_option}**")
    st.write(f"Training Accuracy: {model.score(X_train_scaled, y_train):.2f}")
    st.write(f"Test Accuracy: {model.score(X_test_scaled, y_test):.2f}")

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, cmap='Blues')
    st.pyplot(fig)




# Make Predictions Page
elif page == "Make Predictions!":
    st.title("✈️ Make Predictions")

    st.subheader("Adjust the values below to predict airline passenger satisfaction with a RandomForest Classifier Model:")

    encoding_dict = {
        "gender": {0: "Male", 1: "Female"},
        "customer_type": {0: "Disloyal Customer", 1: "Loyal Customer"},
        "type_of_travel": {0: "Business Travel", 1: "Personal Travel"},
        "class": {0: "Business", 1: "Eco", 2: "Eco Plus"}
    }

    categorical_cols = list(encoding_dict.keys())

    numerical_cols = [
        col for col in df.select_dtypes(include='number').columns.tolist()
        if col not in categorical_cols and col != "satisfaction"
    ]

    user_input = {}

    st.markdown("### Categorical Variables")

    for col in categorical_cols:
        options = list(encoding_dict[col].values())

        selected_text = st.selectbox(
            f"Select {col.replace('_', ' ').title()}:",
            options
        )

        selected_number = [
            key for key, value in encoding_dict[col].items()
            if value == selected_text
        ][0]

        user_input[col] = selected_number

    st.markdown("### Numerical Variables")

    for col in numerical_cols:
        min_value = float(df[col].min())
        max_value = float(df[col].max())
        mean_value = float(df[col].mean())

        if df[col].dtype == 'int64':

            user_input[col] = st.slider(
                f"Select {col.replace('_', ' ').title()}:",
                min_value=int(min_value),
                max_value=int(max_value),
                value=int(round(mean_value)),
                step=1
            )
        
        else:
            user_input[col] = st.slider(
                f"Select {col.replace('_', ' ').title()}:",
                min_value=float(min_value),
                max_value=float(max_value),
                value=float(mean_value)
            )

    user_input_df = pd.DataFrame([user_input])

    user_input_df = user_input_df[df.drop(columns='satisfaction').columns]

    st.write("### Your Input Values")
    st.dataframe(user_input_df, hide_index=True, use_container_width=True)

    X = df.drop(columns='satisfaction')
    y = df['satisfaction']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42,
        stratify=y
    )

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    user_input_scaled = scaler.transform(user_input_df)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Prediction for user input
    prediction = model.predict(user_input_scaled)[0]

    satisfaction_labels = {
        0: "Neutral or Dissatisfied",
        1: "Satisfied"
    }

    st.write("### Prediction Result")
    st.write(f"The model predicts that the passenger is: **{satisfaction_labels[prediction]}**")

    # Test set evaluation
    y_test_pred = model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_test_pred)

    st.write("### Model Evaluation on Test Set")

    st.metric(label="R2 Score", value=f"{test_r2:.2f}")

    st.write("### Confusion Matrix")

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_test_pred,
        display_labels=["Neutral or Dissatisfied", "Satisfied"],
        ax=ax,
        cmap="Blues"
    )

    st.pyplot(fig)

    st.balloons()



# Recommendations and Insights Page
if page == "Recommendations and Insights":

    st.title("💡 Recommendations and Insights")

    st.subheader("1. Correlation with Passenger Satisfaction")

    corr = df.corr(numeric_only=True)["satisfaction"].sort_values(ascending=False)
    top_corr = corr.drop("satisfaction").head(5)

    corr_df = top_corr.reset_index()
    corr_df.columns = ["Variable", "Correlation with Satisfaction"]

    st.write("""
    The table below shows the variables with the strongest positive correlation with passenger satisfaction.
    Correlation helps identify which variables are most associated with satisfaction, but it does not prove causation.
    """)

    st.dataframe(corr_df, hide_index=True, use_container_width=True)

    st.subheader("2. Feature Importance from Random Forest")

    X = df.drop(columns="satisfaction")
    y = df["satisfaction"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    importance_df = pd.DataFrame({
        "Variable": X.columns,
        "Feature Importance": model.feature_importances_
    }).sort_values(by="Feature Importance", ascending=False).head(5)

    st.write("""
    The table below shows the variables that the Random Forest model used the most to predict passenger satisfaction.
    This is useful because it reflects the model’s predictive behavior, not just simple correlations.
    """)

    st.dataframe(importance_df, hide_index=True, use_container_width=True)

    st.subheader("3. Comparison and Recommendations")

    common_variables = set(corr_df["Variable"]).intersection(set(importance_df["Variable"]))

    st.write("""
    Comparing both methods helps identify variables that are not only correlated with satisfaction, 
    but also important for the predictive model.
    """)

    if len(common_variables) > 0:
        st.markdown("**Variables that appear in both analyses:**")
        for variable in common_variables:
            st.write(f"- {variable.replace('_', ' ').title()}")
    else:
        st.write("There are no variables repeated in the top 5 of both methods.")

    st.markdown("""
    **Recommendations for Airlines**

    - Prioritize improving the service areas that appear in both correlation and feature importance results.
    - Improve online boarding if it appears as a top variable, since it may strongly influence the passenger experience before the flight.
    - Improve inflight entertainment, seat comfort, and onboard service if they rank highly, because they are directly related to the passenger’s in-flight experience.
    - Use customer satisfaction surveys regularly to monitor whether improvements in these areas are reflected in better satisfaction scores.
    - Remember that correlation and feature importance show strong relationships, but they do not prove direct causation.
    """)



