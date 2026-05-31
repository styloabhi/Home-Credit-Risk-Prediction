import streamlit as st
import pandas as pd
import numpy as np
from millify import millify

from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import toml

import joblib

st.set_page_config(
    page_title= "Data-Driven Home Credit Default Risk Analysis",
    page_icon= "🏠",
    layout="wide"
)

try:
    USERNAME = st.secrets["username"]
    PASSWORD = st.secrets["password"]

except Exception:

    secrets_path= (
        Path(__file__).parent
        /".streamlit"
        /"secrets.toml"
    )
    secrets = toml.load(secrets_path)
    USERNAME = secrets["username"]
    PASSWORD = secrets["password"]

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("🔐 Login")

    username = st.text_input("User ID")
    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        if (
            username == USERNAME
            and password == PASSWORD
        ):

            st.session_state.logged_in = True
            st.rerun()

        else:

            st.error(
                "Invalid User ID or Password"
            )

    st.stop()
st.sidebar.success(
    f"Logged in as {USERNAME}"
)

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

### Custom CSS for Streamlit App ###

st.markdown("""
<style>

/* Main App */
.stApp{
    background-color:#82B1A3;
}

/* Caption */
div[data-testid="stCaptionContainer"]{
    color:black !important;
    font-weight:600;
}

/* Tabs */
.stTabs [data-baseweb="tab"]{
    color:black !important;
    font-weight:600;
    font-size:16px;
}

/* Active Tab */
.stTabs [aria-selected="true"]{
    color:black !important;
    font-weight:700;
}

/* KPI Cards */
[data-testid="stMetric"]{
    background-color:#61867B;
    padding:15px;
    border-radius:10px;
    text-align:center;
    box-shadow:4px 4px 10px rgba(100,13,13,0.25);
}

/* KPI Labels */
[data-testid="stMetricLabel"]{
    color:white;
    font-weight:600;
}

/* KPI Values */
[data-testid="stMetricValue"]{
    color:white;
    font-weight:bold;
}

/* Headers */
h1,h2,h3{
    color:black;
    font-weight:bold;
}
            
/* Sidebar background */
section[data-testid="stSidebar"]{
    background-color:#61867B;
}

/* Multiselect box */
div[data-baseweb="select"] > div{
    background-color:#82B1A3 !important;
    border:2px solid #3D4F4A !important;
    color:black !important;
}

/* Selected chips */
span[data-baseweb="tag"]{
    background-color:#3D4F4A !important;
    color:white !important;
}

/* Labels */
label{
    color:white !important;
    font-weight:bold;
}

/* Dropdown popup */
div[role="listbox"]{
    background-color:#61867B !important;
    color:white !important;
}

/* Dropdown options */
div[role="option"]{
    background-color:#61867B !important;
    color:white !important;
}

/* Hover effect */
div[role="option"]:hover{
    background-color:#3D4F4A !important;
}

/* Selected option */
div[aria-selected="true"]{
    background-color:#3D4F4A !important;
    color:white !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #82B1A3;
}

::-webkit-scrollbar-thumb {
    background: #3D4F4A;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #640D0D;
}

</style>
""", unsafe_allow_html=True)

### Plotly Template for Consistent Styling ###

def apply_theme(fig):

    fig.update_layout(
        paper_bgcolor="#82B1A3",
        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="black"
        ),

        title=dict(
            x=0.5,
            font=dict(
                size=22,
                color="white"
            )
        ),

        xaxis=dict(
            showgrid=False,
            title=None,
            tickfont=dict(
                color="black",
                size=12
            )
        ),

        yaxis=dict(
            showgrid=False,
            title=None,
            tickfont=dict(
                color="black",
                size=12
            )
        ),

        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    return fig

### Application Title and Description ###

st.title("🏦 Data-Driven Home Credit Default Risk Analysis")
st.markdown("""
<p style="
color:black;
font-size:18px;
font-weight:600;
">
Business Intelligence • Risk Analytics • Portfolio Monitoring • Default Prediction
</p>
""", unsafe_allow_html=True)
st.divider()
BASE_DIR= Path(__file__).parent


#  =====================================================================
### Data Loading with Caching ###
#   ====================================================================

@st.cache_data
def load_data():
    # Load the dataset
    train = pd.read_parquet(
        BASE_DIR / "cleaned_data" / "train_final.parquet"
    )
    test = pd.read_parquet(
        BASE_DIR / "cleaned_data" / "test_final.parquet"
    )
    return train, test


## memory reduction
def reduce_mem_usage(df):

    for col in df.columns:

        col_type = df[col].dtype

        if str(col_type)[:3] == "int":

            c_min = df[col].min()
            c_max = df[col].max()

            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)

            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)

            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)

        elif str(col_type)[:5] == "float":

            df[col] = df[col].astype(np.float32)

    return df

try:
    train, test = load_data()
except Exception as e:
    st.error(f"Data loading failed:{e}")
    st.stop()
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
train["RISK_SEGMENT"] = np.where(
    train['EXT_SOURCE_MEAN']<0.4, "High Risk",
    np.where(
        train["EXT_SOURCE_MEAN"]<0.7,
        "Medium Risk",
        "Low Risk"
    )
)
# ===============================================================================
### load the model
# ===============================================================================

@st.cache_resource
def load_models():

    cat_model = joblib.load(
        BASE_DIR/"models"/"catboost_model.pkl"
    )

    lgb_model = joblib.load(
        BASE_DIR/"models"/"lightgbm_model.pkl"
    )

    xgb_model = joblib.load(
        BASE_DIR/"models"/"xgboost_model.pkl"
    )

    feature_cols = joblib.load(
        BASE_DIR/"models"/"feature_columns.pkl"
    )

    ensemble_info = joblib.load(
        BASE_DIR/"models"/"ensemble_info.pkl"
    )
    encoders=joblib.load(
        BASE_DIR/"models"/"encoders.pkl"
    )
    model_metrics = joblib.load(
    BASE_DIR/"models"/"model_metrics.pkl"
    )

    return (
        cat_model,
        lgb_model,
        xgb_model,
        feature_cols,
        ensemble_info,
        encoders,
        model_metrics
    )



### slicers
 # creating sidebar filters for the executive dashboard
st.sidebar.header("Filters")
# organization type filter
org_filter = st.sidebar.multiselect(
"Organization Type",
sorted(train['ORGANIZATION_TYPE'].dropna().unique())
)

# Income type filter
income_filter = st.sidebar.multiselect(
"Income Type",
sorted(train['NAME_INCOME_TYPE'].dropna().unique())
)

# CITY RATING FILTER
city_filter = st.sidebar.multiselect(
"City Rating",
sorted(train['REGION_RATING_CLIENT_W_CITY'].dropna().unique())
)

# Risk Filter
risk_filter = st.sidebar.multiselect(
"Risk Segment",
sorted(train['RISK_SEGMENT'].dropna().unique())
)
# loan type filter
loan_filter = st.sidebar.multiselect(
    "Loan Type",
    sorted(train['NAME_CONTRACT_TYPE'].dropna().unique())
)

exec_df= train
# apply filters to the executive dashboard dataframe
if org_filter:
    exec_df=exec_df[
        exec_df['ORGANIZATION_TYPE'].isin(org_filter)]
if income_filter:
    exec_df=exec_df[
        exec_df['NAME_INCOME_TYPE'].isin(income_filter)]
    
if city_filter:
    exec_df=exec_df[
        exec_df['REGION_RATING_CLIENT_W_CITY'].isin(city_filter)]
if risk_filter:
    exec_df=exec_df[
        exec_df['RISK_SEGMENT'].isin(risk_filter)]
if loan_filter:
    exec_df=exec_df[
        exec_df['NAME_CONTRACT_TYPE'].isin(loan_filter)]


### Create Tabs for Different Dashboards ###

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Dashboard",
    "Risk Analyst Dashbaord",
    "Portfolio Manager Dashboard",
    "Customer Predictor"
])


### Executive Dashboard ###

with tab1:
    st.header("Executive Dashboard")

    ### Key Performance Indicators (KPIs) ###
    c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
    with c1:
        st.metric("Total Customers", millify(len(exec_df))
                  )
    with c2:
        st.metric("Default Rate %", f"{exec_df['TARGET'].mean()*100:.2f}%")

    with c3: 
        st.metric("Total Credit Exposure",
                  millify(exec_df['AMT_CREDIT'].sum())
                  )
        
    with c4:
        total_prev=exec_df['TOTAL_PREV_APPLICATIONS'].sum()
        approval_rate = (
        exec_df['APPROVED_COUNT'].sum() /
        total_prev * 100
        ) if total_prev > 0 else 0

        st.metric(
            "approval_rate %",
            f"{approval_rate:.2f}%"
        )

    with c5:
        st.metric(
            "Bad Borrowers EXT",
            f"{exec_df.loc[
            exec_df['TARGET']==1,
            "EXT_SOURCE_MEAN"
            ].mean():.2f}"
        )
    with c6:
        st.metric(
            "Good Borrowers EXT",
            f"{exec_df.loc[
            exec_df['TARGET']==0,
            "EXT_SOURCE_MEAN"
            ].mean():.2f}"
        )
    with c7:
        st.metric(
            "High Risk Customers %",
            f"{((exec_df['RISK_SEGMENT']=='High Risk').mean()*100):.2f}")
    with c8:
        st.metric(
            "High Risk Exposure",
            f"{exec_df.loc[
                exec_df["RISK_SEGMENT"]=="High Risk",
                "AMT_CREDIT"].sum()/
                exec_df["AMT_CREDIT"].sum()*100:.2f}%"
        )
        
    st.divider()
    # visuals

    # housing type vs default

    housing_default= (
        exec_df.groupby("NAME_HOUSING_TYPE")['TARGET']
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
    )
    fig_housing = px.line(
        housing_default,
        x = "NAME_HOUSING_TYPE",
        y = "TARGET",
        markers=True,
        title="Housing Type vs Default",
    )

    fig_housing.update_traces(
        line_color="#3D4F4A",
        line_width=4
    )
    fig_housing.update_yaxes(
        title="Default Rate (%)",
        ticksuffix="%"
    )
    fig_housing = apply_theme(fig_housing)
    st.plotly_chart(fig_housing, use_container_width=True)

    ## Late payment history vs default

    late_payment= (
        exec_df.groupby("HAS_DPD_HISTORY",
        observed=False
        )['TARGET']
        .mean().mul(100).reset_index())
    late_payment["HAS_DPD_HISTORY"] = (
        late_payment["HAS_DPD_HISTORY"]
        .astype(int)
        .map({
            0:"No DPD History",
            1:"DPD History"
        })
    )

    fig_late= px.bar(
        late_payment,
        x="HAS_DPD_HISTORY",
        y="TARGET",
        color="HAS_DPD_HISTORY",
        title="Late Payment History vs Default",
        color_discrete_sequence=["#61867B","#3D4F4A"]
    )
    fig_late.update_layout(
        showlegend=False
    )
    fig_late.update_yaxes(
        title="Default Rate (%)",
        ticksuffix="%"
    )
    fig_late = apply_theme(fig_late)
    st.plotly_chart(fig_late, use_container_width=True)

    ## Ext source mean vs default
    bins = np.arange(0, 1.1, 0.1)

    exec_df["EXT_BIN"] = pd.cut(
        exec_df["EXT_SOURCE_MEAN"],
        bins=bins,
        labels=np.round(bins[:-1], 1),  # 0.0 to 0.9
        include_lowest=True
    )
    ext_default = (
        exec_df.groupby("EXT_BIN", observed=False)['TARGET']
        .mean()
        .mul(100)
        .reset_index()
    )
    fig_ext = px.bar(
        ext_default,
        x="EXT_BIN",
        y="TARGET",
        title="External Score vs Default",
        color_discrete_sequence=["#3D4F4A"]
    )
    fig_ext.update_xaxes(type='category')
    fig_ext.update_yaxes(
        title="Default Rate (%)",
        ticksuffix="%"
    )
    fig_ext= apply_theme(fig_ext)
    st.plotly_chart(fig_ext, use_container_width=True)

    ## city rating vs default
    city_default = (
        exec_df.groupby("REGION_RATING_CLIENT_W_CITY", observed=False)['TARGET']
        .mean()
        .mul(100)
        .reset_index()
    )
    fig_city = px.bar(
        city_default,
        x="REGION_RATING_CLIENT_W_CITY",
        y="TARGET",
        title="City Rating vs Default",
        color_discrete_sequence=["#3D4F4A"]
    )
    fig_city.update_yaxes(
        title="Default Rate (%)",
        ticksuffix="%"
    )
    fig_city= apply_theme(fig_city)
    st.plotly_chart(fig_city, use_container_width=True)

    ### previous Overdue vs default
    overdue_default = (
        exec_df.dropna(subset=["HAS_OVERDUE"])
        .groupby("HAS_OVERDUE", observed=False)["TARGET"]
        .mean()
        .mul(100)
        .reset_index()
        )
    overdue_default["HAS_OVERDUE"] = overdue_default["HAS_OVERDUE"].astype(int).map({
        0:"No Previous Overdue",
        1:"Previous Overdue"
    }
    )
    fig_overdue = px.bar(
        overdue_default,
        x="HAS_OVERDUE",
        y="TARGET",
        color="HAS_OVERDUE",
        title="Previous Overdue vs Default",
        color_discrete_sequence=["#61867B","#3D4F4A"]
    )
    fig_overdue.update_yaxes(
        title="Default Rate (%)",
        ticksuffix="%"
    )
    fig_overdue= apply_theme(fig_overdue)
    st.plotly_chart(fig_overdue, use_container_width=True)


    ### Risk Exposure Donut
    risk_exposure = (
        exec_df.groupby("RISK_SEGMENT", observed=False)['AMT_CREDIT']
        .sum()
        .reset_index()
    )
    fig_risk = px.pie(
        risk_exposure,
        names="RISK_SEGMENT",
        values="AMT_CREDIT",
        hole=0.6,
        title="Risk Exposure Distribution",
        color="RISK_SEGMENT",
        color_discrete_map={
            "High Risk": "#640D0D",
            "Medium Risk": "#3D4F4A",
            "Low Risk": "#61867B"
        }
    )
    fig_risk = apply_theme(fig_risk)
    st.plotly_chart(fig_risk, use_container_width=True)


#### Risk Analyst Dashboard ###
with tab2:
    st.header("Risk Analyst Dashboard")
    col1,col2,col3,col4,col5,col6=st.columns(6)
    with col1:
        st.metric(
            "Avg Payment Delay Days",
            f"{exec_df['AVG_PAYMENT_DELAY'].mean():.2f}"
        )
    with col2:
        st.metric(
            "Delinquency Ratio %",
            f"{exec_df['AVG_DPD_RATIO'].mean()*100:.2f}%"
        )
    with col3:
        st.metric(
            "Avg Outstanding Amount",
            millify(exec_df['TOTAL_DEBT'].mean(),2)
        )
    with col4:
        st.metric(
            "AVG Credit Utilization %",
            f"{exec_df['AVG_LIMIT_USAGE_RATIO'].mean()*100:.2f}%"
        )
        
    with col5:
        refusal_rate = (exec_df['REFUSED_COUNT'].sum()/
                        total_prev * 100)if total_prev > 0 else 0
        st.metric(
            "Previous Refusal Rate %",
            f"{refusal_rate:.2f}%"
        )
    with col6:
        st.metric(
            "avg_active_credit",
            f"{exec_df["ACTIVE_CREDIT_COUNT"].mean():.2f}"
        )
    st.divider()


    #### visuals

    # credit utilization vs default
    risk_util = exec_df[
    exec_df["AVG_LIMIT_USAGE_RATIO"].notna()].copy()
    risk_util['UTIL_BUCKET'] = np.where(
        risk_util['AVG_LIMIT_USAGE_RATIO']<=0.2,'0-20%',
        np.where( risk_util['AVG_LIMIT_USAGE_RATIO']<=0.4,'20-40%',
        np.where(risk_util['AVG_LIMIT_USAGE_RATIO']<=0.6,'40-60%',
        np.where(risk_util['AVG_LIMIT_USAGE_RATIO']<=0.8,'60-80%',
        "80%+"))))
    credit_utilization = (risk_util
                          .groupby("UTIL_BUCKET")
                          ['TARGET'].mean().mul(100)
                          ).reset_index()
    fig_util = px.line(
        credit_utilization,
        x= 'UTIL_BUCKET',
        y= "TARGET",
        title="Credit Utilization vs Default",
        color_discrete_sequence=["#3D4F4A"])
    
    fig_util.update_yaxes(
        title="Default Rate (%)",
        ticksuffix="%")
    fig_util= apply_theme(fig_util)
    st.plotly_chart(fig_util, use_container_width=True)



    ## risk segment vs default
    risk_default= (exec_df
                   .groupby('RISK_SEGMENT')['TARGET']
                   .mean().mul(100)).sort_values(ascending=False).reset_index()
    fig_risk_default= px.bar(
        risk_default,
        x='RISK_SEGMENT',
        y='TARGET',
        title="Risk Segment vs Default",
        color_discrete_sequence=["#3D4F4A"])
    fig_risk_default=apply_theme(fig_risk_default)
    st.plotly_chart(fig_risk_default,use_container_width=True)

    # Previous Debt vs Default
    prev_debt= exec_df[exec_df['TOTAL_DEBT'].notna()].copy()
    prev_debt['BRACKET'] = pd.cut(
    prev_debt['TOTAL_DEBT'],
    bins=[-1, 0, 50000, 200000, 500000, np.inf],
    labels=[
        '0',
        '0-50K',
        '50K-200K',
        '200K-500K',
        '500K+'
    ])
    
    debt_default=(prev_debt
                  .groupby('BRACKET',observed=False)['TARGET']
                  .mean().mul(100)).reset_index()
    fig_debt_default=px.bar(
        debt_default,
        x='BRACKET',
        y='TARGET',
        title='Previous Debt vs Default',
        color_discrete_sequence=["#3D4F4A"]
    )
    fig_debt_default=apply_theme(fig_debt_default)
    st.plotly_chart(fig_debt_default,use_container_width=True)

    ## Debt Burden by Income Type

    debt_burden= (exec_df
                  .groupby('NAME_INCOME_TYPE',observed=False)['TOTAL_DEBT']
                  .mean().mul(100)).sort_values().reset_index()
    fig_debt_burden= px.bar(
        debt_burden,
        x='NAME_INCOME_TYPE',
        y='TOTAL_DEBT',
        title='Debt Burden by Income Type',
        color_discrete_sequence=["#3D4F4A"]
    )
    fig_debt_burden=apply_theme(fig_debt_burden)
    st.plotly_chart(fig_debt_burden,use_container_width=True)

    ## Repayment Behavior Distribution
    repayment_distribution = exec_df[exec_df['AVG_PAYMENT_DELAY'].notna()].copy()
    repayment_distribution['BUCKET']=(
        pd.cut(repayment_distribution['AVG_PAYMENT_DELAY'],
            bins=[-1,0,5,15,30,np.inf],
            labels=['0 Days','1-5 Days','6-15 Days','16-30 Days','30+ Days']
        ))
    repayment_distribution=(repayment_distribution
                            .groupby('BUCKET',observed=False)['SK_ID_CURR'].count().reset_index())
    
    fig_repayment=px.bar(
        repayment_distribution,
        x='BUCKET',
        y='SK_ID_CURR',
        title="Repayment Behavior Distribution",
        color_discrete_sequence=["#3D4F4A"]
    )
    fig_repayment=apply_theme(fig_repayment)
    st.plotly_chart(fig_repayment,use_container_width=True)

    ### severe Delinquency History
    
    severe=exec_df[
        exec_df['AVG_SEVERE_DPD_RATIO'].notna()].copy()
    
    severe['HAS_SEVERE_DPD'] = np.where(
        severe['AVG_SEVERE_DPD_RATIO']>0,
        'Severe Deliquency',
        'No Severe Deliquency'
        )
    
    deliquency_history=(severe
                        .groupby('HAS_SEVERE_DPD',observed=False)['SK_ID_CURR']
                        .count().reset_index(name='Total Customers'))
    fig_deliquency=px.pie(
        deliquency_history,
        names='HAS_SEVERE_DPD',
        values='Total Customers',
        title='Severe Deliquency History',
        color='HAS_SEVERE_DPD',
        color_discrete_map={
            "Severe Deliquency": "#640D0D",
            "No Severe Deliquency": "#3D4F4A"
        }
    )
    fig_deliquency=apply_theme(fig_deliquency)
    st.plotly_chart(fig_deliquency,use_container_width=True)

with tab3:
    st.header("Portfolio Manager Dashboard")
    p1,p2,p3,p4,p5,p6= st.columns(6)
    with p1:
        st.metric(
            "Total Portfolio Exposure",
            millify(exec_df['AMT_CREDIT'].sum())
        )
    with p2:
        st.metric(
            "Avg Loan Amount",
            millify(exec_df["AMT_CREDIT"].mean())
        )
    with p3:
        st.metric(
            "Avg Installment Amount",
            millify(exec_df['AMT_ANNUITY'].mean())
        )
    with p4:
        st.metric(
            "High Risk Exposure %",
            f"{
                exec_df.loc[
                    exec_df['RISK_SEGMENT']=='High Risk',
                    "AMT_CREDIT"
                ].sum()/
                exec_df['AMT_CREDIT'].sum()*100:.2f}%"
        )

    with p5:
        st.metric(
            "Avg Loan to Income",
            f"{exec_df['LOAN_TO_INCOME'].mean():.2f}"
        )
    with p6:
        st.metric(
            "Portfolio At Risk %",
            f"{(exec_df['TARGET']==1).mean()*100:.2f}"
            )
    st.divider()

    ### visuals
    # avg loan to income by income type
    LN= exec_df[exec_df['LOAN_TO_INCOME'].notna()].copy()
    loan_income= (
        LN.groupby(
            "NAME_INCOME_TYPE",
            observed=False
        )['LOAN_TO_INCOME']
        .mean()
        .sort_values()
        .reset_index()
    )

    fig_lti = px.bar(
        loan_income,
        x="LOAN_TO_INCOME",
        y="NAME_INCOME_TYPE",
        orientation="h",
        title="Avg Loan To Income by Type",
        color_discrete_sequence=["#3D4F4A"])

    fig_lti = apply_theme(fig_lti)

    st.plotly_chart(
        fig_lti,
        use_container_width=True
    )

    ## Exposure by Loan Type

    loan_exposure = (
    exec_df.groupby(
        "NAME_CONTRACT_TYPE",
        observed=False
    )["AMT_CREDIT"]
    .sum()
    .reset_index()
    )

    fig_loan = px.pie(
        loan_exposure,
        names="NAME_CONTRACT_TYPE",
        values="AMT_CREDIT",
        hole=0.6,
        title="Exposure by Loan Type",
        color="NAME_CONTRACT_TYPE",
        color_discrete_map={
            "Cash loans":"#3D4F4A",
            "Revolving loans":"#61867B"
        }
    )

    fig_loan = apply_theme(fig_loan)

    st.plotly_chart(
        fig_loan,
        use_container_width=True
    )

    ### High Risk Exposure by Income Type
    high_risk_credit = (
    exec_df[exec_df["RISK_SEGMENT"] == "High Risk"]
    .groupby("NAME_INCOME_TYPE")["AMT_CREDIT"]
    .sum()
    )

    total_credit = (
        exec_df.groupby("NAME_INCOME_TYPE")["AMT_CREDIT"]
        .sum()
    )

    risk_tree = (
        (high_risk_credit / total_credit * 100)
        .reset_index(name="HIGH_RISK_EXPOSURE_PCT")
    )

    fig_tree = px.treemap(
    risk_tree,
    path=["NAME_INCOME_TYPE"],
    values="HIGH_RISK_EXPOSURE_PCT",
    title="High Risk Exposure %",
    color="HIGH_RISK_EXPOSURE_PCT",
    color_continuous_scale="Greens"
    )

    fig_tree = apply_theme(fig_tree)

    st.plotly_chart(
        fig_tree,
        use_container_width=True
    )
    
    ### Loan size distribution

    portfolio = exec_df[
        exec_df["AMT_CREDIT"].notna()
        ].copy()

    portfolio["LOAN_BUCKET"] = pd.cut(
        portfolio["AMT_CREDIT"],
        bins=[0,100000,500000,1000000,np.inf],
        labels=[
            "0-100K",
            "100K-500K",
            "500K-1M",
            "1M+"
        ]
        )

    loan_size = (
        portfolio.groupby(
            "LOAN_BUCKET",
            observed=False
        )
        .size()
        .reset_index(name="CUSTOMERS")
    )

    fig_size = px.bar(
        loan_size,
        x="LOAN_BUCKET",
        y="CUSTOMERS",
        title="Loan Size Distribution",
        color_discrete_sequence=["#3D4F4A"]
    )

    fig_size = apply_theme(fig_size)

    st.plotly_chart(
        fig_size,
        use_container_width=True
    )

    ### total exposure by city Rating

    city_exposure = (
        exec_df.groupby(
            "REGION_RATING_CLIENT_W_CITY",
            observed=False
        )["AMT_CREDIT"]
        .sum()
        .reset_index()
    )

    fig_city_exp = px.bar(
        city_exposure,
        x="REGION_RATING_CLIENT_W_CITY",
        y="AMT_CREDIT",
        title="Total Exposure by City",
        color_discrete_sequence=["#3D4F4A"]
    )

    fig_city_exp = apply_theme(fig_city_exp)

    st.plotly_chart(
        fig_city_exp,
        use_container_width=True
    )

    ### Avg Installment Burden

    instalment = (
        exec_df.groupby(
            "NAME_INCOME_TYPE",
            observed=False
        )["AMT_ANNUITY"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig_inst = px.bar(
        instalment,
        x="AMT_ANNUITY",
        y="NAME_INCOME_TYPE",
        orientation="h",
        title="Avg Installment Burden",
        color_discrete_sequence=["#3D4F4A"]
    )

    fig_inst = apply_theme(fig_inst)

    st.plotly_chart(
        fig_inst,
        use_container_width=True
    )

### Customer prediction
with tab4:

    st.header("Customer Default Predictor")



    try:
        (
        cat_model,
        lgb_model,
        xgb_model,
        feature_cols,
        ensemble_info,
        encoders,
        model_metrics
        ) = load_models()
    except Exception as e:
        st.error(f"Model loading failed: {e}")

    try:
        for col, encoder in encoders.items():
            if col in test.columns:
                test[col] = encoder.transform(
                    test[col].astype(str)
                )
    except Exception as e:
        st.error(f"Encoding failed:{e}")
        st.stop()



    customer_id = st.selectbox(
        "Select Customer ID",
        sorted(test["SK_ID_CURR"].unique())
    )

    customer_data = test[
        test["SK_ID_CURR"] == customer_id
    ].copy()

    st.subheader("Customer Profile")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Income",
            millify(
                customer_data["AMT_INCOME_TOTAL"].iloc[0]
            )
        )

    with c2:
        st.metric(
            "Credit Amount",
            millify(
                customer_data["AMT_CREDIT"].iloc[0]
            )
        )

    with c3:
        st.metric(
            "Annuity",
            millify(
                customer_data["AMT_ANNUITY"].iloc[0]
            )
        )

    with c4:
        st.metric(
            "Loan To Income",
            f"{customer_data['LOAN_TO_INCOME'].iloc[0]:.2f}"
        )

    st.divider()

    st.subheader("Model Settings")

    selected_threshold = st.slider(
        "Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(ensemble_info["threshold"]),
        step=0.05
    )

    st.caption(
        f"Current Threshold: {selected_threshold:.2f}"
    )

    if st.button(
        "Predict Default Risk",
        use_container_width=True
    ):

        X = customer_data.drop(
            columns=["SK_ID_CURR"],
            errors="ignore"
        )

        missing_cols = [
            col for col in feature_cols
            if col not in X.columns
        ]

        if missing_cols:
            st.error(
                f"Missing {len(missing_cols)} model features."
            )
            st.stop()

        X = X[feature_cols]
        try:
            st.write("Running CatBoost")
            cat_prob = float(cat_model.predict_proba(X)[:, 1])
            st.write("Running LightGBM")
            lgb_prob = float(lgb_model.predict_proba(X)[:, 1])
            st.write("Running XGBoost")
            xgb_prob = float(xgb_model.predict_proba(X)[:, 1])
            st.write("Models Complete")
        except Exception as e:
            st.error(f"Prediction Error:{e}")
            st.exception(e)
            st.stop()

        probability = (
            cat_prob * ensemble_info["cat_weight"]
            +
            lgb_prob * ensemble_info["lgb_weight"]
            +
            xgb_prob * ensemble_info["xgb_weight"]
        )

        prediction = int(
        probability >= selected_threshold
        )

        if probability >= 0.60:

            risk_segment = "High Risk"
            risk_color = "#640D0D"

        elif probability >= 0.30:

            risk_segment = "Medium Risk"
            risk_color = "#3D4F4A"

        else:

            risk_segment = "Low Risk"
            risk_color = "#61867B"

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "Default Probability",
                f"{probability*100:.2f}%"
            )

        with r2:
            st.metric(
                "Prediction",
                "Likely Default"
                if prediction == 1
                else "Likely Non Default"
            )

        with r3:
            st.metric(
                "Risk Segment",
                risk_segment
            )

        st.divider()

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={
                    "text": "Default Probability (%)"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "bar": {
                        "color": risk_color
                    },
                    "steps": [
                        {
                            "range": [0, 30],
                            "color": "#61867B"
                        },
                        {
                            "range": [30, 60],
                            "color": "#3D4F4A"
                        },
                        {
                            "range": [60, 100],
                            "color": "#640D0D"
                        }
                    ]
                }
            )
        )

        gauge.update_layout(
            paper_bgcolor="#82B1A3",
            font=dict(color="black")
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        if probability >= 0.60:

            recommendation = """
            ❌ High Risk Customer
            
            Recommendation:
            
            • Reject application or perform enhanced review.
            • Request additional income verification.
            • Reduce approved credit limit.
            • Apply stricter monitoring.
            """

        elif probability >= 0.30:

            recommendation = """
            ⚠️ Medium Risk Customer
            
            Recommendation:
            
            • Conditional approval.
            • Moderate credit limit.
            • Monitor repayment behaviour.
            • Require additional documentation if needed.
            """

        else:

            recommendation = """
            ✅ Low Risk Customer
            
            Recommendation:
            
            • Approve application.
            • Eligible for standard credit products.
            • Consider premium offers.
            • Suitable for portfolio growth initiatives.
            """
        st.subheader("Credit Recommendation")

        if probability >= 0.60:
            st.error(recommendation)

        elif probability >= 0.30:
            st.warning(recommendation)

        else:
            st.success(recommendation)


        st.subheader("Key Customer Information")

        c1,c2,c3 = st.columns(3)


        def safe_metric(value,
                decimals=2,
                use_millify=False):

            if pd.isna(value):
                return "N/A"

            if use_millify:
                return millify(value)

            return f"{value:.{decimals}f}"
        
        with c1:
            st.metric(
                "External Score",
                safe_metric(
                    customer_data["EXT_SOURCE_MEAN"].iloc[0]
                )
            )

            st.metric(
                "Active Credits",
                safe_metric(int(
                    customer_data["ACTIVE_CREDIT_COUNT"].iloc[0]
                ))
            )

        with c2:
            st.metric(
                "Loan To Income",
                safe_metric(
                    customer_data["LOAN_TO_INCOME"].iloc[0]
                )
            )

            st.metric(
                "Total Debt",
                safe_metric(customer_data["TOTAL_DEBT"].iloc[0],use_millify=True)
                )

        with c3:
            st.metric(
                "Limit Usage",
                safe_metric(
                    customer_data[
                        "AVG_LIMIT_USAGE_RATIO"
                    ].iloc[0]
                )
            )

            st.metric(
                "Payment Delay",
                safe_metric(
                    customer_data[
                        "AVG_PAYMENT_DELAY"
                    ].iloc[0]
                )
            )
    st.divider()
    st.header("Model Performance")
    m1,m2,m3,m4,m5=st.columns(5)
    with m1:
        st.metric(
            "ROC-AUC",
            round(model_metrics["roc_auc"],4)
            )
    with m2:
        st.metric(
            "Catboost Weight",
            model_metrics["cat_weight"]
        )
    with m3:
        st.metric(
            "LightGBM",
            model_metrics["lgb_weight"]
        )
    with m4:
        st.metric(
            "XGBoost Weight",
            model_metrics['xgb_weight']  
        )
    with m5:
        st.metric(
            "Threshold",
            model_metrics['threshold']
        )
  