import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

st.set_page_config(
    page_title="Sri Lankan House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_artifacts():
    """Load preprocessing artifacts and trained model"""
    artifacts = joblib.load('models/preprocessing_artifacts.pkl')
    model = joblib.load('models/best_model.pkl')
    return artifacts, model

def clean_size(value):
    """Extract numeric value from size strings"""
    if pd.isna(value) or value == "":
        return np.nan
    value_str = str(value)
    value_str = re.sub(r'[^\d.]', '', value_str)
    try:
        return float(value_str) if value_str else np.nan
    except ValueError:
        return np.nan

def preprocess_input(city, bedrooms, bathrooms, house_size, land_size, is_brand_new, is_modern, artifacts):
    """Preprocess user input using saved artifacts"""
    city_tier_map = artifacts['city_tier_map']
    scaler = artifacts['scaler']
    house_median = artifacts['imputation_values']['House_Size_median']
    land_median = artifacts['imputation_values']['Land_Size_median']
    
    city_tier = city_tier_map.get(city, 4)
    
    house_size_clean = clean_size(house_size) if house_size else np.nan
    land_size_clean = clean_size(land_size) if land_size else np.nan
    
    if pd.isna(house_size_clean):
        house_size_clean = house_median
    if pd.isna(land_size_clean):
        land_size_clean = land_median
    
    features_to_scale = pd.DataFrame({
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'House_Size': [house_size_clean],
        'Land_Size': [land_size_clean]
    })
    
    scaled_features = scaler.transform(features_to_scale)
    
    final_input = pd.DataFrame({
        'Bedrooms': [scaled_features[0][0]],
        'Bathrooms': [scaled_features[0][1]],
        'House_Size': [scaled_features[0][2]],
        'Land_Size': [scaled_features[0][3]],
        'Is_Brand_New': [1 if is_brand_new else 0],
        'Is_Modern': [1 if is_modern else 0],
        'City_Tier': [city_tier]
    })
    
    return final_input, city_tier

def get_tier_description(tier):
    """Get description for city tier"""
    descriptions = {
        1: "Luxury Prime (Colombo 3, 7, etc.)",
        2: "Premium Urban",
        3: "Mid-tier Suburban",
        4: "Standard Residential",
        5: "Developing Areas",
        6: "Budget Outer Suburbs"
    }
    return descriptions.get(tier, "Standard Residential")

def main():
    artifacts, model = load_artifacts()
    city_tier_map = artifacts['city_tier_map']
    
    st.title("🏠 Sri Lankan House Price Predictor")
    st.markdown("### Machine Learning-Powered Property Valuation")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📍 Location & Property Details")
        
        city_options = sorted(city_tier_map.keys())
        city = st.selectbox(
            "City",
            options=city_options,
            help="Select the city where the property is located"
        )
        
        col_bed, col_bath = st.columns(2)
        with col_bed:
            bedrooms = st.number_input(
                "Bedrooms",
                min_value=1,
                max_value=15,
                value=3,
                step=1,
                help="Number of bedrooms"
            )
        
        with col_bath:
            bathrooms = st.number_input(
                "Bathrooms",
                min_value=1,
                max_value=15,
                value=2,
                step=1,
                help="Number of bathrooms"
            )
        
        st.subheader("📏 Size Specifications")
        
        house_size = st.text_input(
            "House Size (sqft)",
            value="2000",
            help="Enter house size in square feet. You can include commas (e.g., 2,000)"
        )
        
        land_size = st.text_input(
            "Land Size (perches)",
            value="10",
            help="Enter land size in perches. Leave empty if not applicable"
        )
        
        st.subheader("✨ Property Features")
        
        col_feat1, col_feat2 = st.columns(2)
        with col_feat1:
            is_brand_new = st.checkbox(
                "Brand New",
                help="Is this a newly built property?"
            )
        
        with col_feat2:
            is_modern = st.checkbox(
                "Modern/Luxury",
                help="Does this property have modern/luxury features?"
            )
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if st.button("Predict Price", type="primary", use_container_width=True):
            with st.spinner("Analyzing property..."):
                try:
                    input_data, city_tier = preprocess_input(
                        city, bedrooms, bathrooms, house_size, land_size,
                        is_brand_new, is_modern, artifacts
                    )
                    
                    log_price = model.predict(input_data)[0]
                    predicted_price = np.exp(log_price)
                    
                    st.success("✅ Prediction Complete!")
                    
                    st.metric(
                        label="Predicted Price",
                        value=f"LKR {predicted_price:,.0f}",
                        help="Estimated market value in Sri Lankan Rupees"
                    )
                    
                    price_million = predicted_price / 1_000_000
                    st.info(f"💰 **{price_million:.2f} Million LKR**")
                    
                    with st.expander("📊 Property Summary", expanded=True):
                        summary_data = {
                            "Location": f"{city} (Tier {city_tier} - {get_tier_description(city_tier)})",
                            "Bedrooms": bedrooms,
                            "Bathrooms": bathrooms,
                            "House Size": f"{clean_size(house_size):,.0f} sqft" if house_size else "Not specified",
                            "Land Size": f"{clean_size(land_size):,.0f} perches" if land_size else "Not specified",
                            "Brand New": "Yes" if is_brand_new else "No",
                            "Modern/Luxury": "Yes" if is_modern else "No"
                        }
                        
                        for key, value in summary_data.items():
                            st.text(f"{key:20s}: {value}")
                
                except Exception as e:
                    st.error(f"❌ Prediction Error: {str(e)}")
                    st.error("Please check your inputs and try again.")
        
        else:
            st.info("👆 Fill in the property details and click 'Predict Price' to get started")
            
            st.markdown("### 📖 How It Works")
            st.markdown("""
            1. **Enter Property Details**: Provide information about the property
            2. **Smart Processing**: Data is cleaned and scaled using trained preprocessing
            3. **ML Prediction**: Random Forest model predicts the price instantly
            
            **Model Performance:**
            - Test R²: ~0.75-0.85
            - Mean Absolute Percentage Error: <25%
            - Trained on 724 Sri Lankan properties
            """)
    
    st.markdown("---")
    st.caption("🎓 Built for Machine Learning Project | Model: Random Forest Regressor")

if __name__ == "__main__":
    main()
