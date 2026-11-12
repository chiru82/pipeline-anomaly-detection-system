# Pipeline anomaly detection dashboard

This Streamlit app loads the XGBoost model trained in Colab from `pipeline_xgboost_outputs.zip`. It does not retrain the model. The app uses the metadata from that package to apply the same fluid one-hot encoding and probability temperature used during training.

The app expects `train_dataset_50000.csv`, `combined_dataset_65000.csv`, and `generate_pipeline_splits.py` in the parent project folder. It uses these for input-range warnings and ten new demo cases. The demo inputs are checked against the combined dataset before they are shown; their reference labels are simulated, not real-world validation.

## Run on Windows

Open PowerShell in the project folder and run:

```powershell
python -m pip install -r website_trial\requirements.txt
python -m streamlit run website_trial\app.py
```

If your computer uses the Windows Python launcher, replace `python` with `py` in both commands.

The first launch loads the saved model and creates the demo cases. After it starts, Streamlit prints a local address to open in your browser.

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository for this project.
2. Upload the files while preserving the folder structure above. Upload `pipeline_xgboost_outputs.zip` as a file; the app extracts the model bundle when it starts.
3. In Streamlit Community Cloud, choose **Create app**, select your repository and branch, and set the main file to `website_trial/app.py`.
4. Deploy. If startup fails, open the app's logs and check the missing-file or dependency message.

## Inputs and outputs

The form accepts the six project inputs: flow rate (L/min), inlet and outlet pressure (bar), pipe diameter (mm), pipe length (m), and fluid type. It shows the predicted class, class scores, pressure drop, derived velocity, the held-out test metrics, a confusion matrix, and model-wide feature importance.

The displayed top class score is not test accuracy or guaranteed field confidence. The pressure line between the two endpoints is a straight-line illustration, not a measured pressure profile. Overall test metrics are loaded from the Colab package; they measure performance on simulated data.

