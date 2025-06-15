import os

dashboard_path = os.path.join(os.path.dirname(__file__), 'app', 'dashboard.py')
os.system(f"streamlit run \"{dashboard_path}\"")
