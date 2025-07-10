# streamlit_app.py — Full Production-Ready Streamlit Script

import pandas as pd
import numpy as np
import logging
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, fbeta_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# === Functions ===
def load_data(path):
    try:
        df = pd.read_csv(path)
        logging.info("✅ Dataset loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"❌ Failed to load dataset: {e}")
        return None

def fill_missing_item_price(df):
    try:
        df['Product_category'] = df['Product_category'].fillna('Missing')
        features = ['CSAT Score', 'Product_category', 'Agent Shift', 'Tenure Bucket']
        df_train = df[df['Item_price'].notnull()][features + ['Item_price']]
        df_pred = df[df['Item_price'].isnull()][features]

        encoders = {}
        for col in ['Product_category', 'Agent Shift', 'Tenure Bucket']:
            le = LabelEncoder()
            df_train[col] = le.fit_transform(df_train[col])
            df_pred[col] = le.transform(df_pred[col])
            encoders[col] = le

        X_train_price = df_train.drop(columns='Item_price')
        y_train_price = df_train['Item_price']
        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        reg.fit(X_train_price, y_train_price)

        df.loc[df['Item_price'].isnull(), 'Item_price'] = reg.predict(df_pred)
        logging.info("✅ Missing Item_price values filled using RandomForestRegressor.")
        return df
    except Exception as e:
        logging.error(f"❌ Failed to fill missing item prices: {e}")
        return df

def preprocess_data(df):
    drop_cols = ["Unique id", "Order_id", "Issue_reported at", "issue_responded", 
                 "Agent_name", "Customer Remarks", "order_date_time", "Customer_City", 
                 "Survey_response_Date", "connected_handling_time"]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    def csat_to_sentiment(score):
        if score >= 4:
            return 'positive'
        elif score <= 2:
            return 'negative'
        else:
            return 'neutral'

    df['sentiment_label'] = df['CSAT Score'].apply(csat_to_sentiment)
    return df

def encode_features(df):
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders

def train_models(X_train, Y_train):
    models = {}
    try:
        logreg = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', class_weight='balanced')
        logreg.fit(X_train, Y_train)
        models['Logistic Regression'] = logreg

        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_train, Y_train)
        models['Random Forest'] = rf
        return models
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        return models

def evaluate_model(name, model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    prec = precision_score(Y_test, Y_pred, average='macro')
    f2 = fbeta_score(Y_test, Y_pred, beta=2, average='macro')
    report = classification_report(Y_test, Y_pred, output_dict=True)
    return acc, prec, f2, report, Y_pred


# === Streamlit UI ===
st.set_page_config(page_title="Flipkart CSAT Dashboard", layout="wide")
st.title("📦 Flipkart Customer Satisfaction Prediction Dashboard")

uploaded_file = st.file_uploader("📁 Upload Flipkart Customer Support CSV", type="csv")
if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.subheader("📄 Raw Data Preview")
        st.write(df.shape)
        st.dataframe(df.head())

        st.subheader("📊 Pre-cleaning Distribution & Missingness")
        colX, colY = st.columns(2)
        with colX:
            fig_dist, ax_dist = plt.subplots(figsize=(4, 3))
            sns.countplot(x='CSAT Score', data=df, palette='Set2', ax=ax_dist)
            ax_dist.set_title('Distribution of CSAT Score')
            ax_dist.set_xlabel('Customer Satisfaction Score')
            ax_dist.set_ylabel('Count')
            st.pyplot(fig_dist)
        with colY:
            st.markdown("""
            **Why this chart?** To understand the imbalance in CSAT scores.

            **Insight:** Heavy skew toward CSAT = 5, very few CSAT = 2.

            **Impact:** Indicates need for balanced model training and weighted loss.
            """)

        colA, colB = st.columns(2)
        with colA:
            fig_null, ax_null = plt.subplots(figsize=(6, 3))
            sns.heatmap(df.isnull(), cbar=False, cmap='mako', ax=ax_null)
            ax_null.set_title('Heatmap of Missing Values in Dataset')
            st.pyplot(fig_null)
        with colB:
            st.markdown("""
            **Why this chart?** Visual representation of missing data.

            **Insight:** `Item_price`, `Customer City`, and `order_date_time` have large gaps.

            **Impact:** Must impute `Item_price`, and drop very sparse columns for reliable modeling.
            """)



        # === UBM Charts ===
        st.subheader("🔎 UBM Charts")

        colUBM1, colUBM2 = st.columns(2)
        with colUBM1:
            fig1, ax1 = plt.subplots(figsize=(4,3))
            sns.histplot(df['Item_price'], bins=30, kde=True, ax=ax1)
            ax1.set_title('Distribution of Item Price')
            st.pyplot(fig1)
        with colUBM2:
            st.markdown("""
            **Why did you pick this chart?** Univariate analysis of a key numeric feature.

            **Insight:** Prices cluster below 500, few outliers exist.

            **Impact:** Helps remove skew or apply transformations for better model accuracy.

            **Negative Growth?** ❌ No.
            """)

        colUBM3, colUBM4 = st.columns(2)
        with colUBM3:
            fig2, ax2 = plt.subplots(figsize=(4,3))
            sns.violinplot(x='Agent Shift', y='Item_price', data=df, ax=ax2)
            ax2.set_title('Item Price by Agent Shift')
            st.pyplot(fig2)
        with colUBM4:
            st.markdown("""
            **Why this chart?** Bivariate comparison of numeric vs categorical.

            **Insight:** Night shift has more price variation; morning shows consistent pricing.

            **Business Impact:** Could refine shift-wise pricing strategies.

            **Negative Growth?** ❌ No.
            """)

        colUBM5, colUBM6 = st.columns(2)
        with colUBM5:
            fig3, ax3 = plt.subplots(figsize=(4,3))
            sns.boxplot(x='Tenure Bucket', y='Item_price', hue='Agent Shift', data=df, ax=ax3)
            ax3.set_title('Item Price by Tenure and Shift')
            ax3.tick_params(axis='x', rotation=45)
            st.pyplot(fig3)
        with colUBM6:
            st.markdown("""
            **Why this chart?** Multivariate analysis: tenure + shift + price.

            **Insight:** Senior agents handle higher-priced items more consistently.

            **Impact:** Helps link agent experience with revenue-driven decisions.

            **Negative Growth?** ❌ No.
            """)



        df = fill_missing_item_price(df)
        df = preprocess_data(df)
        df, _ = encode_features(df)
        
        # === Balanced Data ===
        colA, colB = st.columns(2)
        with colA:
            fig_null, ax_null = plt.subplots(figsize=(6, 3))
            sns.heatmap(df.isnull(), cbar=False, cmap='mako', ax=ax_null)
            ax_null.set_title('Heatmap of Balanced Values in Dataset')
            st.pyplot(fig_null)
        with colB:
            st.markdown("""
            **Why this chart?** Visual representation of Balanced Dataset before training.

            **Insight:** `Item_price`  ``Customer City`, and `order_date_time` have 80% missing values . Item price was 80% missing data was added by predictions made my regression mdoel trained on avaiable 20% dataset 

            **Impact:** Must impute `Item_price`, and drop very sparse columns for reliable modeling.
            """)

        # OPTIONAL TWO CHARTS TO FILL CHART 5 & 6
        st.subheader("📌 Additional Insights")
        colR1, colR2 = st.columns(2)
        with colR1:
            fig_cat_shift, ax_cat_shift = plt.subplots(figsize=(4, 3))
            sns.countplot(x='Product_category', hue='Agent Shift', data=df, ax=ax_cat_shift)
            ax_cat_shift.set_title("Product Category by Shift")
            ax_cat_shift.tick_params(axis='x', rotation=45)
            st.pyplot(fig_cat_shift)
        with colR2:
            st.markdown("""
            **Why this chart?** To assess whether agent shifts align with product expertise.

            **Insight:** Some categories are mostly handled in certain shifts.

            **Impact:** Optimizing agent allocation by product and shift may improve CSAT.
            """)

        colR3, colR4 = st.columns(2)
        with colR3:
            fig_sentiment_csat, ax_sentiment_csat = plt.subplots(figsize=(4, 3))
            sns.countplot(x='sentiment_label', hue='CSAT Score', data=df, ax=ax_sentiment_csat)
            ax_sentiment_csat.set_title("Sentiment Label vs CSAT Score")
            st.pyplot(fig_sentiment_csat)
        with colR4:
            st.markdown("""
            **Why this chart?** Sentiment was derived from CSAT. Confirms mapping integrity.

            **Insight:** CSAT 5 aligns with 'positive', CSAT 1 with 'negative'.

            **Impact:** Confirms reliable engineered label for classification modeling.
            """)

        # MODELING SECTION CONTINUES BELOW (NO CHANGE)
        X = df.drop(columns=['CSAT Score'])
        Y = df['CSAT Score']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=42)

        st.subheader("⚙️ Model Training & Evaluation")
        models = train_models(X_train, Y_train)

        for name, model in models.items():
            acc, prec, f2, report, Y_pred = evaluate_model(name, model, X_test, Y_test)
            st.markdown(f"### 📊 {name} Performance")
            st.write(f"**Accuracy:** {acc:.4f} | **Precision (macro):** {prec:.4f} | **F2 Score:** {f2:.4f}")
            st.dataframe(pd.DataFrame(report).transpose())
            if name == 'Random Forest':
                colCM1, colCM2 = st.columns(2)
                with colCM1:
                    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                    cm = confusion_matrix(Y_test, Y_pred, labels=[1, 2, 3, 4, 5])
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                                xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
                    ax_cm.set_xlabel('Predicted')
                    ax_cm.set_ylabel('Actual')
                    ax_cm.set_title('Confusion Matrix')
                    st.pyplot(fig_cm)
                with colCM2:
                    st.markdown("""
                    **Why this chart?** Summarizes model performance across classes.

                    **Insight:** High accuracy for CSAT 5 and CSAT 1. Lower accuracy for mid-scores (2–4).

                    **Impact:** Highlights that the model is best at identifying extremes (high/low satisfaction).
                    May need more samples or tuned weights for mid-range satisfaction scores.
                    """)

else:
    st.info("👈 Upload a .csv file to begin analysis")
