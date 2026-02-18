# 🏠 Sri Lankan House Price Prediction - Streamlit App

This Streamlit application provides an interactive interface for predicting house prices in Sri Lanka using a trained Random Forest model with SHAP explainability.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Install Streamlit** (if not already installed):
```bash
pip install streamlit==1.41.1
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### Running the App

From the project root directory, run:

```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

## 📋 Features

### Input Parameters
- **City**: Select from 50+ Sri Lankan cities (automatically mapped to price tiers)
- **Bedrooms**: Number of bedrooms (1-15)
- **Bathrooms**: Number of bathrooms (1-15)
- **House Size**: Enter in square feet (accepts text with commas)
- **Land Size**: Enter in perches (optional)
- **Brand New**: Checkbox for new properties
- **Modern/Luxury**: Checkbox for modern/luxury properties

### Output
- **Predicted Price**: Displayed in LKR with comma formatting
- **Property Summary**: Overview of all entered details
- **SHAP Waterfall Plot**: Visual explanation of feature contributions to the prediction

## 🎯 How It Works

1. **Load Artifacts**: Pre-trained model and preprocessing artifacts are loaded once
2. **User Input**: Enter property details through intuitive form controls
3. **Preprocessing**: 
   - City mapped to tier (1-6) using saved mappings
   - Sizes cleaned and imputed if missing
   - Features scaled using saved StandardScaler
4. **Prediction**: Random Forest predicts log price, converted to actual price
5. **Explainability**: SHAP generates feature importance visualization

## 🏗️ Architecture

```
app.py
├── load_artifacts()          # Cache model and preprocessing artifacts
├── clean_size()              # Extract numeric values from size strings
├── preprocess_input()        # Apply full preprocessing pipeline
├── generate_shap_waterfall() # Create SHAP explanation plot
└── main()                    # Streamlit UI and workflow
```

## 📊 Model Information

- **Model Type**: Random Forest Regressor
- **Training Samples**: 724 Sri Lankan properties
- **Test R²**: ~0.75-0.85
- **Features**: 7 (Bedrooms, Bathrooms, House_Size, Land_Size, Is_Brand_New, Is_Modern, City_Tier)
- **Target**: Log-transformed price (Price_Log)

## 🎨 UI Features

- **Wide Layout**: Optimal space utilization
- **Two-Column Design**: Input form on left, results on right
- **Real-time Validation**: Input constraints prevent invalid entries
- **Responsive**: Adapts to different screen sizes
- **Professional Styling**: Clean, modern interface

## 🔧 Configuration

The app uses default Streamlit configuration. To customize:

1. Create `.streamlit/config.toml` in the project root
2. Add custom theme settings:

```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

## 📁 Required Files

Ensure these files exist before running:
- `models/preprocessing_artifacts.pkl` - Scaler and mappings
- `models/best_model.pkl` - Trained Random Forest model

## 🐛 Troubleshooting

### "File not found" errors
- Ensure you're running from the project root directory
- Verify `models/` folder contains required .pkl files

### "Module not found" errors
- Install streamlit: `pip install streamlit`
- Check all requirements: `pip install -r requirements.txt`

### Port already in use
- Stop other Streamlit instances
- Or specify a different port: `streamlit run app.py --server.port 8502`

## 📝 Notes

- The app caches model and artifacts for performance (`@st.cache_resource`)
- SHAP plots are generated on-demand for each prediction
- Input validation prevents common errors (e.g., negative sizes, invalid ranges)

## 🎓 Project Context

This application is part of the Machine Learning project on Sri Lankan house price prediction. For detailed documentation, see:
- `docs/DATA_PREPROCESSING_DOCUMENTATION.md`
- `docs/MODEL_TRAINING_DOCUMENTATION.md`
- `docs/MODEL_EXPLAINABILITY_DOCUMENTATION.md`

---

**Built with Streamlit, SHAP, and scikit-learn** 🚀
