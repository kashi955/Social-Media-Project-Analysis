import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    r2_score,
    mean_squared_error
)
from sklearn.decomposition import PCA


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="LIMFADD - Social Media Analysis",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("📊 LIMFADD - Social Media Project Analysis")
st.write(
    "Machine Learning analysis of social media account data "
    "using Python and Scikit-learn."
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():

    df = pd.read_csv("LIMFADD.csv")

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
    )

    # Convert numeric-looking columns to numeric
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")

        if converted.notna().sum() == df[col].notna().sum():
            df[col] = converted

    # Replace invalid values
    df = df.replace(
        ["#DIV/0!", "NaN", "nan", ""],
        np.nan
    )

    # Fill numeric missing values
    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    df[numeric_cols] = df[numeric_cols].fillna(
        df[numeric_cols].median()
    )

    # Encode categorical columns
    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns

    for col in categorical_cols:
        if col != "Labels":
            df[col] = LabelEncoder().fit_transform(
                df[col].astype(str)
            )

    # Encode target label separately
    if "Labels" in df.columns:
        df["Labels"] = LabelEncoder().fit_transform(
            df["Labels"].astype(str)
        )

    return df


df = load_data()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("Navigation")

section = st.sidebar.radio(
    "Select Section",
    [
        "Dataset Overview",
        "Data Analysis",
        "Linear Regression",
        "Logistic Regression",
        "PCA Analysis"
    ]
)


# ---------------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------------
if section == "Dataset Overview":

    st.header("📁 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Records", df.shape[0])

    with col2:
        st.metric("Total Features", df.shape[1])

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    st.write("### Columns")

    st.write(list(df.columns))

    st.write("### Data Types")

    st.dataframe(
        pd.DataFrame(df.dtypes, columns=["Data Type"]),
        use_container_width=True
    )


# ---------------------------------------------------------
# DATA ANALYSIS
# ---------------------------------------------------------
elif section == "Data Analysis":

    st.header("📈 Data Analysis")

    st.subheader("Correlation Heatmap")

    corr = df.select_dtypes(
        include=[np.number]
    ).corr()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("Pie Chart of Labels")
    label_counts = df["Labels"].value_counts()

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    ax.pie(
        label_counts.values,
        autopct="%1.1f%%",
        shadow=False,
        startangle=90,
        radius=0.70,
        pctdistance=0.60
    )
    
    

    plt.close(fig)

    st.pyplot(fig)

    st.subheader("Followers Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        df["Followers"],
        bins=40
    )

    ax.set_xlabel("Followers")
    ax.set_ylabel("Frequency")
    ax.set_title("Followers Distribution")

    st.pyplot(fig)

    st.subheader("Donut Chart of Labels")

    label_counts = df["Labels"].value_counts()

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.pie(
        label_counts.values,
        labels=label_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.45)
    )

    ax.set_title("Donut Chart of Labels")
    ax.axis("equal")

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)
    st.subheader("Bar Chart of Labels")

    label_counts = df["Labels"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(
        label_counts.index.astype(str),
        label_counts.values
    )

    ax.set_xlabel("Labels")
    ax.set_ylabel("Count")
    ax.set_title("Bar Chart of Labels")

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)
    
    st.subheader("Horizontal Bar Chart of Labels")

    label_counts = df["Labels"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.barh(
        label_counts.index.astype(str),
        label_counts.values
    )

    ax.set_xlabel("Count")
    ax.set_ylabel("Labels")
    ax.set_title("Horizontal Bar Chart of Labels")

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.subheader("Followers Line Chart")

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        df.index,
        df["Followers"]
    )

    ax.set_xlabel("Record Index")
    ax.set_ylabel("Followers")
    ax.set_title("Followers Line Chart")

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.subheader("Followers vs Following Scatter Plot")

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.scatter(
        df["Followers"],
        df["Following"],
        s=15,
        alpha=0.6
    )

    ax.set_xlabel("Followers")
    ax.set_ylabel("Following")
    ax.set_title("Followers vs Following")

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.subheader("Countplot of Labels")

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.countplot(
        x="Labels",
        data=df,
        ax=ax
    )

    ax.set_xlabel("Labels")
    ax.set_ylabel("Count")
    ax.set_title("Countplot of Labels")

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

# ---------------------------------------------------------
# LINEAR REGRESSION
# ---------------------------------------------------------
elif section == "Linear Regression":

    st.header("📊 Linear Regression")

    st.write(
        "Predicting Followers using selected social media features."
    )

    # Avoid target leakage from ratio-based features
    X = df.drop(
        [
            "Followers",
            "Following/Followers",
            "Posts/Followers"
        ],
        axis=1
    )

    y = df["Followers"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(X_test)

    r2 = r2_score(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "R² Score",
            f"{r2:.4f}"
        )

    with col2:
        st.metric(
            "Mean Squared Error",
            f"{mse:,.2f}"
        )

    st.subheader("Actual vs Predicted Followers")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        y_test,
        y_pred,
        s=15
    )

    ax.set_xlabel("Actual Followers")
    ax.set_ylabel("Predicted Followers")
    ax.set_title("Actual vs Predicted Followers")

    st.pyplot(fig)


# ---------------------------------------------------------
# LOGISTIC REGRESSION
# ---------------------------------------------------------
elif section == "Logistic Regression":

    st.header("🤖 Logistic Regression")

    st.write(
        "Classifying social media accounts into high-follower "
        "and low-follower groups."
    )

    # Create classification target
    df["Followers_high"] = (
        df["Followers"] > df["Followers"].median()
    ).astype(int)

    # Remove target-related and ratio features
    X = df.drop(
        [
            "Followers",
            "Followers_high",
            "Following/Followers",
            "Posts/Followers"
        ],
        axis=1
    )

    y = df["Followers_high"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    st.dataframe(
        pd.DataFrame(report).transpose(),
        use_container_width=True
    )


# ---------------------------------------------------------
# PCA ANALYSIS
# ---------------------------------------------------------
elif section == "PCA Analysis":

    st.header("🔍 Principal Component Analysis (PCA)")

    st.write(
        "PCA is used for dimensionality reduction and visualization."
    )

    # Create target for visualization
    df["Followers_high"] = (
        df["Followers"] > df["Followers"].median()
    ).astype(int)

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    y_num = df["Followers_high"]

    # Remove target-related and ratio features
    X = numeric_df.drop(
        [
            "Followers",
            "Followers_high",
            "Following/Followers",
            "Posts/Followers"
        ],
        axis=1
    )

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_scaled)

    explained_variance = pca.explained_variance_ratio_

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "PC1 Variance",
            f"{explained_variance[0] * 100:.2f}%"
        )

    with col2:
        st.metric(
            "PC2 Variance",
            f"{explained_variance[1] * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Combined Variance",
            f"{explained_variance.sum() * 100:.2f}%"
        )

    st.subheader("PCA Visualization")

    fig, ax = plt.subplots(figsize=(6, 4))

    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y_num,
        s=12,
        cmap="viridis",
        alpha=0.7
    )

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("PCA - Social Media Dataset")

    fig.colorbar(
        scatter,
        ax=ax,
        label="Target Classes",
        shrink=0.8
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    plt.close(fig)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.sidebar.markdown("---")

st.sidebar.write(
    "LIMFADD Social Media Project"
)

st.sidebar.write(
    "Built with Python, Pandas, "
    "Scikit-learn and Streamlit"
)