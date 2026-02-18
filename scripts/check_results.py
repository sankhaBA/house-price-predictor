import joblib

metadata = joblib.load('models/model_metadata.pkl')

print('='*70)
print('MODEL PERFORMANCE RESULTS')
print('='*70)
print(f"\nModel Type: {metadata['model_type']}")
print(f"Model Name: {metadata['model_name']}")

print(f"\nTraining Data: {metadata['train_samples']} houses")
print(f"Test Data: {metadata['test_samples']} houses")

print(f"\nBest Hyperparameters Found:")
for param, value in metadata['hyperparameters'].items():
    print(f"  {param}: {value}")

print(f"\n{'='*70}")
print('TEST SET PERFORMANCE (What matters!)')
print('='*70)
print(f"R² Score: {metadata['metrics']['test_r2']:.4f}")
print(f"RMSE: LKR {metadata['metrics']['test_rmse_lkr']:,.0f}")
print(f"MAE: LKR {metadata['metrics']['test_mae_lkr']:,.0f}")
print(f"MAPE: {metadata['metrics']['test_mape']:.2f}%")
