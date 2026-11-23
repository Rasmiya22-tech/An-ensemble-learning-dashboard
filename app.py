from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import malaria_models as mm


st.set_page_config(
    page_title="Sri Lanka Malaria: Ensemble Forecasting",
    layout="wide",
)

PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
GROUP_COLORS = {
    "Bagging": PALETTE[0],
    "Boosting": PALETTE[1],
    "Stacking": PALETTE[2],
    "Hybrid": PALETTE[3],
    "Time series": PALETTE[6],
    "Baseline": "#999999",
}


@st.cache_data(show_spinner=False)
def load_dataset():
    """Load the project's CSV automatically from the folder containing app.py."""
    project_dir = Path(__file__).resolve().parent
    candidates = [
        project_dir / "srilanka_malaria_monthly_2000_2024.csv",
        
    ]
    csv_path = next((path for path in candidates if path.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            "Dataset CSV not found. Place 'srilanka_malaria_monthly_2000_2024.csv' "
            "in the same folder as app.py."
        )

    df = mm.load(str(csv_path))
    required_columns = {
        "date", "year", "month", "district", "climate_zone",
        "indigenous_cases", "imported_cases", "rainfall_mm",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    return df

@st.cache_data(show_spinner=False)
def run_experiment_cached(df, train_end, selected_models, rf_params, boosting_params, sarima_order, smearing):
    """Cache a completed experiment until the user requests a fresh training run."""
    return mm.run_experiment(
        df,
        train_end=train_end,
        rf_p=rf_params,
        xgb_p=boosting_params,
        order=sarima_order,
        use=selected_models,
        smearing=smearing,
    )


def config_key(train_end, selected_models, rf_params, boosting_params, sarima_order, smearing):
    return (
        train_end,
        tuple(selected_models),
        tuple(sorted(rf_params.items())),
        tuple(sorted(boosting_params.items())),
        tuple(sarima_order),
        bool(smearing),
    )


def metric_card_value(table, metric_column):
    if metric_column not in table.columns:
        return "Unavailable"
    values = table[metric_column].dropna()
    if values.empty:
        return "Unavailable"
    return str(values.idxmin())


st.title("Malaria Case Trend Prediction in Sri Lanka")
st.caption(
    "Ensemble learning dashboard: Bagging · Boosting · Stacking · Hybrid "
    "(RF + ANN, SARIMA + RF). The project dataset is simulated/reconstructed "
    "to follow Sri Lanka's malaria elimination trajectory (2000–2024)."
)

# Load the project dataset automatically; no CSV upload control.
try:
    df = load_dataset()
except Exception as exc:
    st.error(f"Could not load the project dataset: {exc}")
    st.info("Make sure the malaria CSV file is in the same folder as app.py.")
    st.stop()

nat = mm.national(df)

# Sidebar: experiment controls
st.sidebar.header("Experiment settings")
train_end = st.sidebar.select_slider(
    "Train until (target month)",
    options=[f"{year}-12-01" for year in range(2004, 2011)],
    value="2007-12-01",
    format_func=lambda value: value[:4],
)

DEFAULT_MODELS = [
    "Random Forest",
    "Bagging (trees)",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Stacking",
    "RF + ANN",
    "SARIMA",
    "SARIMA + RF",
]
selected_models = st.sidebar.multiselect(
    "Models",
    options=mm.ALL_MODELS,
    default=DEFAULT_MODELS,
)

with st.sidebar.expander("Random Forest / Bagging", expanded=False):
    rf_params = {
        "n_estimators": st.slider("Number of trees", 50, 600, 300, 50),
        "max_depth": st.slider("Maximum tree depth", 3, 25, 12),
        "min_samples_leaf": st.slider("Minimum samples per leaf", 1, 30, 5),
    }

with st.sidebar.expander("Boosting models", expanded=False):
    boosting_params = {
        "n_estimators": st.slider("Boosting rounds", 50, 800, 400, 50),
        "max_depth": st.slider("Boosting tree depth", 2, 8, 4),
        "learning_rate": st.select_slider(
            "Learning rate",
            options=[0.01, 0.03, 0.05, 0.1, 0.2],
            value=0.05,
        ),
    }

with st.sidebar.expander("SARIMA", expanded=False):
    sarima_order = (
        int(st.number_input("p", min_value=0, max_value=3, value=1)),
        int(st.number_input("d", min_value=0, max_value=2, value=1)),
        int(st.number_input("q", min_value=0, max_value=3, value=1)),
    )

smearing = st.sidebar.checkbox(
    "Duan smearing correction",
    value=True,
    help="Corrects downward bias when converting log predictions back to case counts.",
)
st.sidebar.caption("A fresh training run starts when you click the button in Model comparison.")

current_config = config_key(
    train_end,
    selected_models,
    rf_params,
    boosting_params,
    sarima_order,
    smearing,
)

# Tabs
tab_story, tab_models, tab_districts, tab_about = st.tabs(
    ["The story", "Model comparison", "District forecasts", "About"]
)

# -----------------------------------------------------------------------------
# Tab 1: Story and exploratory analysis
# -----------------------------------------------------------------------------
with tab_story:
    endemic_data = df[df["year"] <= 2012]
    total_endemic_cases = endemic_data["indigenous_cases"].sum()
    dry_zone_cases = endemic_data.loc[
        endemic_data["climate_zone"] == "Dry", "indigenous_cases"
    ].sum()
    dry_zone_share = dry_zone_cases / total_endemic_cases if total_endemic_cases else 0

    col1, col2, col3, col4 = st.columns(4)
    cases_2000 = df.loc[df["year"] == 2000, "indigenous_cases"].sum()
    cases_2012 = df.loc[df["year"] == 2012, "indigenous_cases"].sum()
    cases_after_2012 = df.loc[df["year"] >= 2013, "indigenous_cases"].sum()

    col1.metric("Cases in 2000 (dataset)", f"{int(cases_2000):,}")
    col2.metric("Cases in 2012", f"{int(cases_2012):,}")
    col3.metric("Indigenous cases 2013–2024", f"{int(cases_after_2012):,}")
    col4.metric("Share of cases in the Dry zone", f"{dry_zone_share:.1%}")

    trend_fig = go.Figure()
    trend_fig.add_scatter(
        x=nat["date"],
        y=nat["indigenous_cases"] + 1,
        name="Indigenous cases",
        line=dict(color=PALETTE[0], width=2.5),
    )
    trend_fig.add_scatter(
        x=nat["date"],
        y=nat["imported_cases"] + 1,
        name="Imported cases",
        line=dict(color=PALETTE[1], width=2.5),
    )
    trend_fig.add_vline(
        x=pd.Timestamp("2012-12-01"),
        line_dash="dash",
        line_color="#888888",
    )
    trend_fig.update_yaxes(type="log", title="Monthly cases (log scale, +1)")
    trend_fig.update_layout(
        title="Malaria case trend in Sri Lanka",
        height=430,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12),
        template="plotly_white",
        margin=dict(l=50, r=30, t=80, b=50),
    )
    st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})

    left, right = st.columns(2)
    zone_summary = (
        endemic_data.groupby(["year", "climate_zone"], as_index=False)["indigenous_cases"]
        .sum()
    )
    zone_fig = px.bar(
        zone_summary,
        x="year",
        y="indigenous_cases",
        color="climate_zone",
        barmode="group",
        title="Cases by climate zone (endemic era)",
        color_discrete_sequence=PALETTE,
        template="plotly_white",
    )
    zone_fig.update_layout(height=380, xaxis_title="Year", yaxis_title="Indigenous cases")
    left.plotly_chart(zone_fig, use_container_width=True, config={"displayModeBar": False})

    season_summary = (
        endemic_data.groupby("month", as_index=False)
        .agg(cases=("indigenous_cases", "sum"), mean_rainfall=("rainfall_mm", "mean"))
    )
    season_fig = px.bar(
        season_summary,
        x="month",
        y="cases",
        title="Seasonality: monthly case distribution",
        color_discrete_sequence=[PALETTE[0]],
        template="plotly_white",
    )
    season_fig.update_layout(height=380, xaxis_title="Month", yaxis_title="Indigenous cases")
    right.plotly_chart(season_fig, use_container_width=True, config={"displayModeBar": False})

    districts = sorted(df["district"].dropna().unique())
    preferred_district = "Anuradhapura"
    district_index = districts.index(preferred_district) if preferred_district in districts else 0
    selected_district = st.selectbox(
        "Explore a district",
        districts,
        index=district_index,
        key="story_district",
    )
    district_data = df[df["district"] == selected_district]
    left, right = st.columns(2)

    district_cases_fig = px.line(
        district_data,
        x="date",
        y="indigenous_cases",
        title=f"{selected_district}: indigenous malaria cases",
        color_discrete_sequence=[PALETTE[0]],
        template="plotly_white",
    )
    district_cases_fig.update_layout(height=360)
    left.plotly_chart(district_cases_fig, use_container_width=True, config={"displayModeBar": False})

    rainfall_fig = px.bar(
        district_data,
        x="date",
        y="rainfall_mm",
        title=f"{selected_district}: rainfall",
        color_discrete_sequence=[PALETTE[2]],
        template="plotly_white",
    )
    rainfall_fig.update_layout(height=360, yaxis_title="Rainfall (mm)")
    right.plotly_chart(rainfall_fig, use_container_width=True, config={"displayModeBar": False})

# -----------------------------------------------------------------------------
# Tab 2: Model comparison and training
# -----------------------------------------------------------------------------
with tab_models:
    st.subheader("Train and compare machine learning models")
    st.write(
        "Choose the training period and models in the sidebar, then click the button "
        "to start a new training run."
    )

    train_clicked = st.button(
        "▶ Train & compare models",
        type="primary",
        use_container_width=False,
    )

    if train_clicked:
        if not selected_models:
            st.warning("Select at least one model from the sidebar before training.")
        else:
            run_experiment_cached.clear()
            st.session_state.pop("malaria_result", None)
            st.session_state.pop("malaria_config", None)
            try:
                with st.spinner("Training models and evaluating forecasts. Please wait..."):
                    result = run_experiment_cached(
                        df,
                        train_end,
                        tuple(selected_models),
                        rf_params,
                        boosting_params,
                        sarima_order,
                        smearing,
                    )
                st.session_state["malaria_result"] = result
                st.session_state["malaria_config"] = current_config
                st.success("Training and model evaluation completed.")
            except Exception as exc:
                st.exception(exc)

    result = st.session_state.get("malaria_result")
    saved_config = st.session_state.get("malaria_config")

    if result is not None:
        if saved_config != current_config:
            st.warning(
                "Your sidebar settings have changed since the last training run. "
                "Click **Train & compare models** to train again using the new settings. "
                "The charts below still show the previous run until then."
            )

        for note in result.get("notes", []):
            st.warning(note)

        metrics_table = result["table"].copy()
        district_metrics = metrics_table.dropna(subset=["RMSE (log)"])

        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Training rows", f"{result['n_train']:,}")
        metric2.metric("Testing rows", f"{result['n_test']:,}")

        if not district_metrics.empty:
            best_district_model = district_metrics["RMSE (log)"].idxmin()
            metric1_caption = (
                f"R²: {district_metrics.loc[best_district_model, 'R2 (log)']:.3f}"
            )
            metric3.metric(
                "Best district model",
                best_district_model,
                metric1_caption,
            )
        else:
            metric3.metric("Best district model", "Unavailable")

        metric4.metric(
            "Best national model",
            metric_card_value(metrics_table, "National RMSE (cases)"),
        )

        st.divider()
        st.subheader("Model performance comparison")
        chart_left, chart_right = st.columns(2)

        # District-level RMSE chart
        if not district_metrics.empty:
            district_plot_data = (
                district_metrics.reset_index()
                .rename(columns={"index": "Model"})
                .sort_values("RMSE (log)", ascending=True)
            )
            district_fig = px.bar(
                district_plot_data,
                x="RMSE (log)",
                y="Model",
                color="Technique",
                orientation="h",
                color_discrete_map=GROUP_COLORS,
                text="RMSE (log)",
                title="District-level forecast error",
                template="plotly_white",
                hover_data={"R2 (log)": ":.3f", "MAE (log)": ":.3f"},
            )
            district_fig.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside",
                cliponaxis=False,
            )
            district_fig.update_layout(
                yaxis=dict(autorange="reversed", title="Model"),
                xaxis=dict(title="RMSE on log cases (lower is better)", rangemode="tozero"),
                height=max(480, 48 * len(district_plot_data)),
                margin=dict(l=170, r=65, t=80, b=65),
                legend_title_text="Technique",
                uniformtext_minsize=10,
                uniformtext_mode="hide",
            )
            chart_left.plotly_chart(district_fig, use_container_width=True, config={"displayModeBar": False})
        else:
            chart_left.info("No district-level model metrics are available for this run.")

        # National-level RMSE chart
        if "National RMSE (cases)" in metrics_table.columns:
            national_plot_data = (
                metrics_table.dropna(subset=["National RMSE (cases)"])
                .reset_index()
                .rename(columns={"index": "Model"})
                .sort_values("National RMSE (cases)", ascending=True)
            )
        else:
            national_plot_data = pd.DataFrame()

        if not national_plot_data.empty:
            national_fig = px.bar(
                national_plot_data,
                x="National RMSE (cases)",
                y="Model",
                color="Technique",
                orientation="h",
                color_discrete_map=GROUP_COLORS,
                text="National RMSE (cases)",
                title="National forecast error",
                template="plotly_white",
            )
            national_fig.update_traces(
                texttemplate="%{text:.1f}",
                textposition="outside",
                cliponaxis=False,
            )
            national_fig.update_layout(
                yaxis=dict(autorange="reversed", title="Model"),
                xaxis=dict(title="RMSE in cases per month (lower is better)", rangemode="tozero"),
                height=max(480, 48 * len(national_plot_data)),
                margin=dict(l=170, r=65, t=80, b=65),
                legend_title_text="Technique",
                uniformtext_minsize=10,
                uniformtext_mode="hide",
            )
            chart_right.plotly_chart(national_fig, use_container_width=True, config={"displayModeBar": False})
        else:
            chart_right.info("No national-level metrics are available for this run.")

        # National actual versus forecast lines
        national_predictions = result["national"].copy()
        if not national_predictions.empty and "date" in national_predictions.columns:
            national_predictions = national_predictions.set_index("date")
            forecast_columns = [
                column for column in national_predictions.columns if column != "actual"
            ]
            default_forecasts = [
                name
                for name in ["SARIMA", "Random Forest", "Stacking", "SARIMA + RF"]
                if name in forecast_columns
            ]
            selected_forecasts = st.multiselect(
                "Forecast models to display",
                options=forecast_columns,
                default=default_forecasts,
                key="national_forecast_models",
            )

            forecast_fig = go.Figure()
            forecast_fig.add_scatter(
                x=national_predictions.index,
                y=national_predictions["actual"],
                name="Actual cases",
                mode="lines",
                line=dict(color="#111111", width=3),
            )
            for index, model_name in enumerate(selected_forecasts):
                forecast_fig.add_scatter(
                    x=national_predictions.index,
                    y=national_predictions[model_name],
                    name=model_name,
                    mode="lines",
                    line=dict(color=PALETTE[index % len(PALETTE)], width=2),
                )
            forecast_fig.update_layout(
                title="One-month-ahead national forecasts",
                xaxis_title="Month",
                yaxis_title="Indigenous malaria cases per month",
                height=460,
                hovermode="x unified",
                legend=dict(orientation="h", y=1.12),
                template="plotly_white",
                margin=dict(l=55, r=30, t=90, b=55),
            )
            st.plotly_chart(forecast_fig, use_container_width=True, config={"displayModeBar": False})

        # Feature importance
        importances = result.get("importances", pd.DataFrame())
        if not importances.empty:
            st.subheader("Feature importance")
            importance_model = st.selectbox(
                "Select model",
                options=list(importances.columns),
                key="importance_model",
            )
            importance_data = (
                importances[importance_model]
                .sort_values(ascending=True)
                .tail(12)
                .rename("importance")
                .reset_index()
                .rename(columns={"index": "Feature"})
            )
            importance_fig = px.bar(
                importance_data,
                x="importance",
                y="Feature",
                orientation="h",
                title=f"Top 12 features: {importance_model}",
                color_discrete_sequence=[PALETTE[0]],
                template="plotly_white",
                text="importance",
            )
            importance_fig.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside",
                cliponaxis=False,
            )
            importance_fig.update_layout(
                height=430,
                margin=dict(l=150, r=55, t=75, b=55),
                xaxis_title="Relative importance",
                yaxis_title="Feature",
            )
            st.plotly_chart(importance_fig, use_container_width=True, config={"displayModeBar": False})

        # Metrics table and download
        st.subheader("Full model metrics")
        st.dataframe(
            metrics_table.style.format(
                {
                    column: "{:.3f}"
                    for column in metrics_table.select_dtypes(include="number").columns
                },
                na_rep="-",
            ),
            use_container_width=True,
        )
        st.download_button(
            "⬇ Download results CSV",
            data=metrics_table.to_csv().encode("utf-8"),
            file_name="results.csv",
            mime="text/csv",
        )

        with st.expander("How to interpret the results"):
            st.markdown(
                """
                - **District level:** compare RMSE (log), MAE (log), and R² (log). Lower RMSE and MAE are better; higher R² is better.
                - **Bagging:** Random Forest averages multiple trees and can reduce prediction variance.
                - **Boosting:** builds models sequentially to correct previous errors.
                - **Stacking:** combines predictions from multiple base models using a meta-model.
                - **Hybrid:** combines different model families, such as Random Forest + ANN.
                - **National level:** compare RMSE in cases per month. Lower values indicate smaller forecast errors.
                - Rankings can change when the training cutoff or hyperparameters change.
                """
            )
    else:
        st.info("Choose your experiment settings in the sidebar, then click **Train & compare models**.")

# -----------------------------------------------------------------------------
# Tab 3: District forecasts
# -----------------------------------------------------------------------------
with tab_districts:
    result = st.session_state.get("malaria_result")
    saved_config = st.session_state.get("malaria_config")

    if result is None:
        st.info("Train the models in the **Model comparison** tab first.")
    else:
        if saved_config != current_config:
            st.warning(
                "The sidebar settings differ from the last training run. "
                "These district forecasts show the previous run. Train again to update them."
            )

        district_test = result["district_test"].copy()
        excluded_columns = {"target_date", "district", "target", "actual"}
        available_models = [
            column for column in district_test.columns if column not in excluded_columns
        ]

        if available_models and not district_test.empty:
            default_model_index = (
                available_models.index("Random Forest")
                if "Random Forest" in available_models
                else 0
            )
            selected_model = st.selectbox(
                "Forecast model",
                available_models,
                index=default_model_index,
                key="district_forecast_model",
            )

            district_names = sorted(district_test["district"].dropna().unique())
            preferred = "Mannar"
            district_index = district_names.index(preferred) if preferred in district_names else 0
            selected_district = st.selectbox(
                "District",
                district_names,
                index=district_index,
                key="district_forecast_district",
            )

            selected_data = district_test[
                district_test["district"] == selected_district
            ].sort_values("target_date")

            district_fig = go.Figure()
            district_fig.add_scatter(
                x=selected_data["target_date"],
                y=selected_data["actual"],
                name="Actual",
                mode="lines",
                line=dict(color="#111111", width=3),
            )
            if "Naive (next = this month)" in selected_data.columns:
                district_fig.add_scatter(
                    x=selected_data["target_date"],
                    y=selected_data["Naive (next = this month)"],
                    name="Naive baseline",
                    mode="lines",
                    line=dict(color="#999999", width=1.5, dash="dot"),
                )
            district_fig.add_scatter(
                x=selected_data["target_date"],
                y=selected_data[selected_model],
                name=selected_model,
                mode="lines",
                line=dict(color=PALETTE[0], width=2.5),
            )
            district_fig.update_layout(
                title=f"{selected_district}: next-month forecast versus actual",
                xaxis_title="Target month",
                yaxis_title="Indigenous cases",
                height=430,
                hovermode="x unified",
                template="plotly_white",
            )
            st.plotly_chart(district_fig, use_container_width=True, config={"displayModeBar": False})

            district_error = district_test.groupby("district").apply(
                lambda group: pd.Series(
                    {
                        "Total cases (test)": group["actual"].sum(),
                        "Naive MAE": (
                            group["actual"] - group["Naive (next = this month)"]
                        ).abs().mean(),
                        f"{selected_model} MAE": (
                            group["actual"] - group[selected_model]
                        ).abs().mean(),
                    }
                )
            )
            naive_mae = district_error["Naive MAE"].where(
                district_error["Naive MAE"] != 0
            )
            district_error["Improvement vs naive %"] = (
                1 - district_error[f"{selected_model} MAE"] / naive_mae
            ) * 100
            district_error = district_error.sort_values(
                "Total cases (test)", ascending=False
            )

            improvement_fig = px.bar(
                district_error.reset_index(),
                x="district",
                y="Improvement vs naive %",
                title=f"{selected_model}: MAE improvement compared with naive baseline",
                color_discrete_sequence=[PALETTE[0]],
                template="plotly_white",
            )
            improvement_fig.update_layout(
                xaxis_title="District",
                yaxis_title="MAE improvement (%)",
                height=430,
            )
            st.plotly_chart(improvement_fig, use_container_width=True, config={"displayModeBar": False})
            st.dataframe(district_error.round(2), use_container_width=True)
        else:
            st.warning("No district forecast predictions are available in this training result.")

# -----------------------------------------------------------------------------
# Tab 4: About
# -----------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
        ### Project

        **Research question:** Can ensemble models forecast next month's malaria cases per district as an early-warning tool for Sri Lanka's Anti-Malaria Campaign?

        **Target:** `log1p(indigenous cases next month)`. The models learn the month-on-month change in log cases and add it back to the current level.

        **Train/test split:** time-ordered, without shuffling. The default trains through December 2007 and tests on January 2008–December 2012.

        **After 2012:** the dataset contains no indigenous cases, so the surveillance concern shifts towards imported cases and possible reintroduction.

        | Technique | Models |
        |---|---|
        | Bagging | Random Forest, BaggingRegressor |
        | Boosting | XGBoost, LightGBM, CatBoost, Gradient Boosting |
        | Stacking | Random Forest + XGBoost + LightGBM with a Ridge meta-model |
        | Hybrid | Random Forest + ANN (MLP), SARIMA + Random Forest on residuals |

        **Data limitation:** the dataset is reconstructed. It follows Sri Lanka's broad malaria elimination trajectory but does not exactly match official Anti-Malaria Campaign figures. Results should be re-evaluated using official district-level data.

        """
    )
