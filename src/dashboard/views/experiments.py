"""
Machine learning experiments and models registry dashboard view.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.client import DashboardClient
from src.dashboard.components import empty_state, error_box, safe_call


def render(client: DashboardClient, run_id: str | None = None) -> None:
    """Render trained models list and comparison view."""
    st.header("ML Experiments & Model Registry")

    models, err = safe_call(client.list_models)
    if err:
        error_box(f"Failed to fetch model registry: {err}")
        return

    if not models:
        empty_state("No trained model artifacts found in the registry.")
        return

    st.subheader("Available Model Artifacts")
    df_models = pd.DataFrame(models)
    st.dataframe(df_models, use_container_width=True)

    if len(models) >= 2:
        st.divider()
        st.subheader("Side-by-Side Model Comparison")
        m_names = [m["name"] for m in models]
        c1, c2 = st.columns(2)
        with c1:
            sel_1 = st.selectbox("Select Model A", options=m_names, index=0)
        with c2:
            sel_2 = st.selectbox("Select Model B", options=m_names, index=1)

        mod_a = next(m for m in models if m["name"] == sel_1)
        mod_b = next(m for m in models if m["name"] == sel_2)

        c1.json(mod_a)
        c2.json(mod_b)
