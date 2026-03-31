import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_revenue_model(df: pd.DataFrame):
    features = ['Active_Tenants', 'Total_Energy_Cost', 'Total_Opex']
    target = 'Revenue'
    X = df[features].values
    y = df[target].values
    if len(X) < 2:
        return None, None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score, (X_test, y_test)

def train_cost_model(df: pd.DataFrame):
    features = ['Diesel_Liters', 'Electricity_kWh', 'Maintenance_Cost', 'Repair_Cost', 'Staff_Visits']
    target = 'Total_Opex'
    X = df[features].values
    y = df[target].values
    if len(X) < 2:
        return None, None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score, (X_test, y_test)

def train_classification_model(df: pd.DataFrame):
    # Create productivity label
    df = df.copy()
    df['Productivity_Label'] = pd.cut(df['Productivity'],
                                      bins=[-np.inf, 1, 1.5, np.inf],
                                      labels=['Low', 'Medium', 'High'])
    features = ['Active_Tenants', 'Total_Energy_Cost', 'Total_Opex']
    target = 'Productivity_Label'
    df_clean = df.dropna(subset=[target]).copy()
    if len(df_clean) < 2 or len(df_clean[target].unique()) < 2:
        return None, None, None
    X = df_clean[features].values
    y = df_clean[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc, (X_test, y_test)