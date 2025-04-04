import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Set the page configuration
st.set_page_config(page_title="ED Simulation Dashboard", layout="wide")

# Title
st.title("Emergency Department Simulation Dashboard")

# Simulated data generation (replace this with real simulation input/output)
def simulate_ed_data(n=100):
    np.random.seed(42)
    priorities = np.random.choice(['High', 'Medium', 'Low'], size=n, p=[0.2, 0.5, 0.3])
    wait_times = np.random.exponential(scale=30, size=n)  # in minutes
    treatment_times = np.random.exponential(scale=60, size=n)  # in minutes
    arrival_times = np.cumsum(np.random.exponential(scale=5, size=n))
    is_admitted = np.random.choice([True, False], size=n, p=[0.3, 0.7])

    df = pd.DataFrame({
        'Priority': priorities,
        'Wait Time (min)': wait_times,
        'Treatment Time (min)': treatment_times,
        'Arrival Time': arrival_times,
        'Admitted': is_admitted
    })
    return df

# Simulate data
data = simulate_ed_data(200)

# Sidebar filters
priority_filter = st.sidebar.multiselect("Select Priority Levels", options=['High', 'Medium', 'Low'], default=['High', 'Medium', 'Low'])

filtered_data = data[data['Priority'].isin(priority_filter)]

# Key Performance Indicators (KPIs)
st.subheader("Key Performance Indicators")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

avg_wait = filtered_data['Wait Time (min)'].mean()
avg_treatment = filtered_data['Treatment Time (min)'].mean()
admit_rate = filtered_data['Admitted'].mean() * 100

kpi1.metric(label="Avg Wait Time (min)", value=f"{avg_wait:.1f}")
kpi2.metric(label="Avg Treatment Time (min)", value=f"{avg_treatment:.1f}")
kpi3.metric(label="Admission Rate (%)", value=f"{admit_rate:.1f}")
kpi4.metric(label="Total Patients", value=len(filtered_data))

# Charts
st.subheader("Performance Charts")
chart1, chart2 = st.columns(2)

with chart1:
    st.markdown("### Wait Time by Priority")
    fig1, ax1 = plt.subplots()
    filtered_data.boxplot(column='Wait Time (min)', by='Priority', ax=ax1)
    plt.title('')
    st.pyplot(fig1)

with chart2:
    st.markdown("### Treatment Time Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(filtered_data['Treatment Time (min)'], bins=20, color='skyblue')
    ax2.set_xlabel('Treatment Time (min)')
    ax2.set_ylabel('Frequency')
    st.pyplot(fig2)

# Real-time Simulation View
st.subheader("Simulated Real-time Patient Arrival")
placeholder = st.empty()

for i in range(10):
    random_patient = data.sample(1).iloc[0]
    with placeholder.container():
        st.write(f"**Patient Priority:** {random_patient['Priority']}")
        st.write(f"**Wait Time:** {random_patient['Wait Time (min)']:.1f} minutes")
        st.write(f"**Treatment Time:** {random_patient['Treatment Time (min)']:.1f} minutes")
        st.write(f"**Admitted:** {'Yes' if random_patient['Admitted'] else 'No'}")
    time.sleep(1)
