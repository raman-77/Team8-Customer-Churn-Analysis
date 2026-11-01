BAO NGOc NGUYEN — ANN CUSTOMER CHURN PREDICTION

1. Put Bao_ANN_Churn_Prediction.ipynb and Updated_Dataset_with_Clusters.csv in the same directory.
2. In Jupyter, run the notebook cells from top to bottom.
3. Dependencies: pandas, numpy, scikit-learn, matplotlib, keras and either tensorflow or torch.
   Example installation (run once): %pip install pandas numpy scikit-learn matplotlib keras tensorflow
   If TensorFlow is not supported by your Python version, install keras + torch instead.
4. Check the notebook's outputs and exported ANN_outputs/ charts and tables.
5. The notebook includes a completed preview of most evaluation cells from our local run.
   It should be rerun from top to bottom in your own environment to export held-out test predictions.

NOTES:
- This file does not contain a genuine CustomerID. DatasetRowNumber is only a row index.
- The input column SeniorCitizen was standardised; notebook restores its 0/1 values from its two known values.
- For an unbiased comparison, Model B creates fresh clustering labels from training rows rather than relying on supplied full-data labels.
- Small differences in results are expected with different Keras backends.
