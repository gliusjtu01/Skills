import streamlit as st
import simpy
import random
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# Simulation parameters
RANDOM_SEED = 42
SIM_TIME = 12 * 60  # 12 hours
ARRIVAL_RATE = 5  # Patients arrive every 5 minutes
NUM_DOCTORS = 3  
NUM_NURSES = 2  
TRIAGE_TIME = (2, 6)  
TREATMENT_TIME = {1: (10, 20), 2: (15, 25), 3: (20, 30)}  

# Data storage
waiting_times = []
treatment_times = []
total_times = []
triage_counts = {1: 0, 2: 0, 3: 0}
live_patient_queue = []

# Streamlit UI
st.title("🏥 Emergency Department Simulation Dashboard")
status_placeholder = st.empty()
chart_placeholder = st.empty()

class EmergencyDepartment:
    def __init__(self, env, num_doctors, num_nurses):
        self.env = env
        self.doctor = simpy.PriorityResource(env, num_doctors)  
        self.nurse = simpy.Resource(env, num_nurses)  

    def triage(self, patient):
        triage_time = random.randint(*TRIAGE_TIME)
        yield self.env.timeout(triage_time)
        priority = random.choices([1, 2, 3], weights=[0.2, 0.5, 0.3])[0]  
        triage_counts[priority] += 1
        return triage_time, priority

    def treat(self, patient, priority):
        treatment_time = random.randint(*TREATMENT_TIME[priority])
        yield self.env.timeout(treatment_time)
        return treatment_time

def patient(env, name, ed):
    arrival_time = env.now

    with ed.nurse.request() as req:
        yield req
        triage_time, priority = yield env.process(ed.triage(name))
    
    with ed.doctor.request(priority=priority) as req:
        live_patient_queue.append((name, priority))
        yield req
        wait_time = env.now - arrival_time
        waiting_times.append(wait_time)
        live_patient_queue.remove((name, priority))

        treatment_time = yield env.process(ed.treat(name, priority))
        treatment_times.append(treatment_time)

    total_time = env.now - arrival_time
    total_times.append(total_time)

def patient_generator(env, ed):
    patient_id = 0
    while True:
        yield env.timeout(random.expovariate(1.0 / ARRIVAL_RATE))
        patient_id += 1
        env.process(patient(env, f"Patient-{patient_id}", ed))

# Streamlit simulation control
def run_simulation():
    random.seed(RANDOM_SEED)
    env = simpy.Environment()
    ed = EmergencyDepartment(env, NUM_DOCTORS, NUM_NURSES)
    env.process(patient_generator(env, ed))

    while env.peek() < SIM_TIME:
        env.step()
        time.sleep(0.1)  # Simulate real-time effect

        # Update patient queue & stats
        status_placeholder.markdown(f"### ⏳ Live Patient Queue ({len(live_patient_queue)})")
        queue_df = pd.DataFrame(live_patient_queue, columns=["Patient", "Priority"])
        status_placeholder.dataframe(queue_df)

        # Update triage distribution chart
        triage_df = pd.DataFrame.from_dict(triage_counts, orient='index', columns=['Count'])
        triage_df.index = ["Critical (1)", "Urgent (2)", "Non-Urgent (3)"]

        fig, ax = plt.subplots()
        triage_df.plot(kind='bar', legend=False, ax=ax, color=['red', 'orange', 'green'])
        ax.set_title("Triage Distribution")
        ax.set_ylabel("Number of Patients")
        ax.set_xlabel("Priority Level")
        chart_placeholder.pyplot(fig)

    st.success("Simulation Complete! ✅")

if st.button("🚀 Start Simulation"):
    run_simulation()
