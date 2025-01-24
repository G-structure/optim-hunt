from optim_hunter.experiments.regressors_comparison import compare_llm_and_regressors
from optim_hunter.datasets import get_original2
from optim_hunter.llama_model import load_llama_model
from optim_hunter.utils import prepare_prompt, extract_model_prediction
from optim_hunter.plot_html import create_multi_line_plot, create_bar_plot, with_identifier
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Callable
import time

def analyze_llm_component_understanding(
    model: Any,
    n_samples: int = 25,
    random_seed: int = 42
) -> str:
    """Analyze LLM's understanding of different mathematical components."""
    # Define mathematical components to test
    components: List[Tuple[str, Callable[[pd.DataFrame], np.ndarray]]] = [
        ("Fourth Power (x₀⁴)", lambda x: x['Feature 0']**4),
        ("Square Root (2/√x₁)", lambda x: 2/np.sqrt(x['Feature 1'])),
        ("Multiplication (x₁×x₂)", lambda x: x['Feature 1']*x['Feature 2']),
        ("Combined ((x₁×x₂ - 2/√x₁/√x₃)²)",
         lambda x: (x['Feature 1']*x['Feature 2'] -
                   2/np.sqrt(x['Feature 1'])/np.sqrt(x['Feature 3']))**2),
        ("Full Function",
         lambda x: (x['Feature 0']**4 +
                   (x['Feature 1']*x['Feature 2'] -
                    2/np.sqrt(x['Feature 1'])/np.sqrt(x['Feature 3']))**2)**0.75)
    ]

    # Storage for results
    results: Dict[str, Dict[str, List[float]]] = {
        "mse": {comp[0]: [] for comp in components},
        "predictions": {comp[0]: [] for comp in components},
        "actuals": {comp[0]: [] for comp in components}
    }

    # Test each component multiple times
    for seed in range(random_seed, random_seed + n_samples):
        # Get fresh dataset
        x_train, y_train, x_test, y_test = get_original2(random_state=seed)

        for comp_name, comp_fn in components:
            # Calculate component values
            y_train_comp = pd.Series(comp_fn(x_train), index=y_train.index, name='Output')
            y_test_comp = pd.Series(comp_fn(x_test), index=y_test.index, name='Output')

            # Prepare prompt and get prediction
            prompt = prepare_prompt(x_train, y_train_comp, x_test)
            prompt_tensor = model.to_tokens(prompt, prepend_bos=True)

            pred = extract_model_prediction(model, prompt_tensor, seed)

            if pred is not None:
                # Calculate MSE
                mse = float((pred - y_test_comp.iloc[0])**2)

                # Store results
                results["mse"][comp_name].append(mse)
                results["predictions"][comp_name].append(pred)
                results["actuals"][comp_name].append(float(y_test_comp.iloc[0]))

    # Create visualization
    @with_identifier("component-understanding")
    def create_component_plot() -> str:
        # Calculate average MSE for each component
        avg_mse = {comp: np.mean(mses) for comp, mses in results["mse"].items()}

        # Sort components by average MSE
        sorted_components = sorted(avg_mse.items(), key=lambda x: x[1])

        return create_bar_plot(
            x_values=[comp[0] for comp in sorted_components],
            y_values=[comp[1] for comp in sorted_components],
            title="LLM Understanding of Mathematical Components",
            x_label="Component",
            y_label="Mean Squared Error",
            include_plotlyjs=True,
            include_theme_js=True
        )

    # Create correlation plot
    @with_identifier("prediction-correlation")
    def create_correlation_plot() -> str:
        # Prepare data for multi-line plot
        y_values_list = []
        labels = []

        for comp_name in components[-2:]:  # Just show last two components
            predictions = results["predictions"][comp_name[0]]
            actuals = results["actuals"][comp_name[0]]

            # Add both predictions and actuals
            y_values_list.extend([predictions, actuals])
            labels.extend([f"{comp_name[0]} (Predicted)",
                         f"{comp_name[0]} (Actual)"])

        return create_multi_line_plot(
            y_values_list=y_values_list,
            labels=labels,
            title="Predictions vs Actuals for Complex Components",
            x_label="Sample Index",
            y_label="Value",
            include_plotlyjs=False,
            include_theme_js=True
        )

    # Calculate correlation coefficients
    correlations = {
        comp_name: np.corrcoef(
            results["predictions"][comp_name],
            results["actuals"][comp_name]
        )[0,1]
        for comp_name in results["mse"].keys()
    }

    # Create correlation bar plot
    @with_identifier("component-correlation")
    def create_correlation_bar_plot() -> str:
        return create_bar_plot(
            x_values=list(correlations.keys()),
            y_values=list(correlations.values()),
            title="Prediction-Actual Correlation by Component",
            x_label="Component",
            y_label="Correlation Coefficient",
            include_plotlyjs=False,
            include_theme_js=True
        )

    # Combine all plots
    component_plot = create_component_plot()
    correlation_plot = create_correlation_plot()
    correlation_bar = create_correlation_bar_plot()

    # Add summary statistics
    summary = f"""
    <div style='margin: 20px; padding: 20px; background: rgba(0,0,0,0.1); border-radius: 5px;'>
        <h3>Analysis Summary:</h3>
        <ul>
            <li>Best understood component: {min(results['mse'].items(), key=lambda x: np.mean(x[1]))[0]}</li>
            <li>Most challenging component: {max(results['mse'].items(), key=lambda x: np.mean(x[1]))[0]}</li>
            <li>Highest correlation: {max(correlations.items(), key=lambda x: x[1])[0]} ({max(correlations.values()):.3f})</li>
            <li>Number of samples tested: {n_samples}</li>
        </ul>
    </div>
    """

    return f"{component_plot}\n{correlation_plot}\n{correlation_bar}\n{summary}"

# Example usage:
if __name__ == "__main__":
    model = load_llama_model()
    html = analyze_llm_component_understanding(model, n_samples=25)
    print(html)
