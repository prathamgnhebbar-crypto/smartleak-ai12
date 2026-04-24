import pandas as pd
import numpy as np
import streamlit as st
import time
import matplotlib.pyplot as plt
import sqlite3
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# DATABASE
# -----------------------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

# Users table
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT,
    password TEXT,
    role TEXT
)
""")

# Reports table (NEW)
c.execute("""
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    address TEXT,
    timestamp TEXT
)
""")

conn.commit()

# -----------------------------
# FUNCTIONS
# -----------------------------
def add_user(username, password, role):
    c.execute("INSERT INTO users VALUES (?, ?, ?)", (username, password, role))
    conn.commit()

def login_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return c.fetchall()

def add_report(address):
    time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO reports (address, timestamp) VALUES (?, ?)", (address, time_now))
    conn.commit()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="SmartLeak AI", layout="wide")

# -----------------------------
# DATA + MODEL
# -----------------------------
np.random.seed(42)
data_size = 2000

pressure = np.random.normal(50, 10, data_size)
flow = np.random.normal(200, 50, data_size)
zones = np.random.choice(['Zone A', 'Zone B', 'Zone C'], data_size)

leak = [1 if (p < 40 and f > 230) else 0 for p, f in zip(pressure, flow)]

df = pd.DataFrame({
    'Pressure': pressure,
    'Flow': flow,
    'Zone': zones,
    'Leak': leak
})

X = df[['Pressure', 'Flow']]
y = df['Leak']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# -----------------------------
# SESSION
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

menu = ["Login", "Signup"]
choice = st.sidebar.selectbox("Menu", menu)

# -----------------------------
# LOGIN
# -----------------------------
if not st.session_state.logged_in:

    if choice == "Signup":
        st.title("📝 Create Account")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        r = st.selectbox("Role", ["public", "bbmp"])

        if st.button("Signup"):
            add_user(u, p, r)
            st.success("Account created")

    else:
        st.title("🔐 SmartLeak AI Login")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            res = login_user(u, p)
            if res:
                st.session_state.logged_in = True
                st.session_state.role = res[0][2]
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

# -----------------------------
# DASHBOARDS
# -----------------------------
else:

    role = st.session_state.role

    # =========================
    # 🌍 PUBLIC DASHBOARD
    # =========================
    if role == "public":

        st.title("🚰 SmartLeak AI (Public Dashboard)")

        col1, col2, col3 = st.columns(3)
        col1.metric("💧 Water Supplied", "1.2M L")
        col2.metric("⚠️ Active Leaks", "2")
        col3.metric("💸 Water Loss", "₹8,500")

        st.markdown("---")

        # Zone Status
        st.subheader("📍 Zone Status")
        zones_status = {"Zone A": "Normal", "Zone B": "Leak", "Zone C": "Normal"}

        for z, s in zones_status.items():
            if s == "Leak":
                st.error(f"{z} - Leak Detected")
            else:
                st.success(f"{z} - Normal")

        # Map
        st.subheader("🗺️ Live Map")
        st.map(pd.DataFrame({
            'lat': [13.1, 13.2, 13.3],
            'lon': [77.5, 77.6, 77.7]
        }))

        # Graph
        st.subheader("📈 Water Usage Trend")
        fig, ax = plt.subplots()
        ax.plot(np.random.randint(150, 300, 10))
        st.pyplot(fig)

        # Alerts
        st.subheader("🚨 Alerts")
        st.warning("Leak detected in Zone B")
        st.info("Maintenance team dispatched")

        # 🔥 REPORT LEAK FEATURE
        st.subheader("📢 Report a Leak")

        address = st.text_input("Enter Leak Location / Address")

        if st.button("Submit Report"):
            if address.strip() != "":
                add_report(address)
                st.success("Leak reported successfully!")
            else:
                st.error("Please enter an address")

    # =========================
    # 🏢 BBMP DASHBOARD
    # =========================
    else:

        st.title("🏢 SmartLeak AI Control Center")

        # Overview
        st.subheader("📊 System Overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Zones", "3")
        col2.metric("Active Leaks", "2")
        col3.metric("Critical Zones", "1")
        col4.metric("System Health", "Stable")

        st.markdown("---")

        # Controls
        st.sidebar.header("Control Panel")
        zone = st.sidebar.selectbox("Zone", ['Zone A', 'Zone B', 'Zone C'])
        pressure_val = st.sidebar.slider("Pressure", 0, 100, 50)
        flow_val = st.sidebar.slider("Flow", 0, 500, 200)

        prob = model.predict_proba([[pressure_val, flow_val]])[0][1]
        pred = model.predict([[pressure_val, flow_val]])

        # Prediction
        st.subheader("🔍 Leak Analysis")

        if pred[0] == 1:
            st.error("Leak Detected")
        else:
            st.success("No Leak")

        st.metric("Probability", f"{prob*100:.2f}%")

        # Graph
        st.subheader("📈 Data Analysis")

        fig, ax = plt.subplots()
        ax.scatter(df['Pressure'], df['Flow'], alpha=0.5)
        st.pyplot(fig)

        # 🔥 VIEW REPORTS (NEW)
        st.subheader("📍 Reported Leak Locations")

        c.execute("SELECT * FROM reports")
        reports = c.fetchall()

        if reports:
            report_df = pd.DataFrame(reports, columns=["ID", "Address", "Time"])
            st.dataframe(report_df)
        else:
            st.info("No reports yet")

        # Live Monitoring
        st.subheader("📡 Live Monitoring")

        if st.button("Start Monitoring"):

            data = pd.DataFrame(columns=["Pressure", "Flow"])
            chart = st.empty()
            status = st.empty()

            for i in range(15):
                p = np.random.normal(50, 10)
                f = np.random.normal(200, 50)

                new = pd.DataFrame([[p, f]], columns=["Pressure", "Flow"])
                data = pd.concat([data, new], ignore_index=True)

                chart.line_chart(data)

                if model.predict([[p, f]])[0] == 1:
                    status.error("⚠️ Leak Detected!")
                else:
                    status.success("System Normal")

                time.sleep(0.5)

    # Logout
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()