from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from prophet import Prophet
import os
import traceback

app = Flask(__name__)
CORS(app)

# Absolute path to ensure no mistakes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'Walmart.csv')

def load_data():
    try:
        if not os.path.exists(DATA_PATH):
            return None, f"File not found at {DATA_PATH}"
        
        df = pd.read_csv(DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        print(f"Successfully loaded {len(df)} rows.")
        return df, None
    except Exception as e:
        return None, str(e)

df, load_error = load_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    return jsonify({
        "data_loaded": df is not None if 'df' in globals() else False,
        "load_error": load_error,
        "rows": len(df) if df is not None else 0
    })

@app.route('/api/stats')
def get_stats():
    store_id = request.args.get('store', type=int)
    if df is None: return jsonify({"error": load_error}), 500
    try:
        data = df if store_id is None else df[df['Store'] == store_id]
        if data.empty: return jsonify({"error": "No data for this store"}), 404
        
        return jsonify({
            "total_revenue": f"${data['Weekly_Sales'].sum():,.0f}",
            "avg_sales": f"${data['Weekly_Sales'].mean():,.0f}",
            "cpi": f"{data['CPI'].iloc[-1]:.2f}",
            "unemployment": f"{data['Unemployment'].mean():.2f}%"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chart')
def get_chart():
    store_id = request.args.get('store', type=int)
    if df is None: return jsonify({"error": "No data"}), 500
    try:
        data = df if store_id is None else df[df['Store'] == store_id]
        res = data.groupby('Date')['Weekly_Sales'].sum().reset_index()
        res = res.sort_values('Date')
        return jsonify({
            "labels": res['Date'].dt.strftime('%Y-%m-%d').tolist(),
            "data": res['Weekly_Sales'].tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/correlation')
def get_correlation():
    if df is None: return jsonify({"error": "No data"}), 500
    try:
        # Correlation between sales and economic factors
        cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        corr_matrix = df[cols].corr()['Weekly_Sales'].to_dict()
        return jsonify(corr_matrix)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stores')
def get_stores():
    if df is None: return jsonify({"error": "No data"}), 500
    return jsonify(sorted(df['Store'].unique().tolist()))

@app.route('/api/forecast')
def get_forecast():
    store_id = request.args.get('store', type=int)
    if df is None: return jsonify({"error": "No data"}), 500
    try:
        data = df if store_id is None else df[df['Store'] == store_id]
        prophet_df = data.groupby('Date')['Weekly_Sales'].sum().reset_index()
        prophet_df.columns = ['ds', 'y']
        
        # Calculate a realistic growth baseline
        # The dataset shows very stable sales (~47M average for all stores).
        # Compounding 3.5% over 14 years might be too aggressive for mature retail stores.
        # We will use 2.5% (inflation + slight expansion) which is more realistic for same-store/stable groups.
        last_date = prophet_df['ds'].max()
        # Use the median of the last 6 months to avoid being skewed by outliers
        recent_data = prophet_df.sort_values('ds').tail(26)
        baseline_val = recent_data['y'].median()
        
        synthetic_points = []
        # Create yearly anchor points to maintain the trend without overshooting
        for year in range(2013, 2027):
            # 2.5% CAGR is more conservative and realistic for this specific data scale
            growth_mult = (1.025) ** (year - 2012)
            synthetic_date = pd.Timestamp(year=year, month=6, day=15)
            synthetic_points.append({'ds': synthetic_date, 'y': baseline_val * growth_mult})
        
        guide_df = pd.DataFrame(synthetic_points)
        full_df = pd.concat([prophet_df, guide_df]).sort_values('ds')

        # Multiplicative seasonality is better for retail (spikes grow with volume)
        model = Prophet(
            yearly_seasonality=True, 
            weekly_seasonality=False, 
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.01 # Make it less flexible to follow the linear anchor trend
        )
        model.fit(full_df)
        
        target_date = pd.Timestamp('2026-12-31')
        days_diff = (target_date - last_date).days
        weeks_to_forecast = (days_diff // 7) + 1
        
        future = model.make_future_dataframe(periods=weeks_to_forecast, freq='W')
        forecast = model.predict(future)
        
        # Ensure we don't have negative predictions
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        
        return jsonify({
            "labels": forecast['ds'].dt.strftime('%Y-%m-%d').tolist(),
            "forecast": forecast['yhat'].tolist(),
            "actual": prophet_df['y'].tolist(),
            "historical_labels": prophet_df['ds'].dt.strftime('%Y-%m-%d').tolist()
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use run.py for production/docker, but keep this for quick local dev
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
