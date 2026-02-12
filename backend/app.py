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
        
        # 1. EXTRACT SEASONAL PATTERN (Weekly averages by month)
        # We use this to make the "gap" years look like real Walmart data
        prophet_df['month'] = prophet_df['ds'].dt.month
        seasonal_pattern = prophet_df.groupby('month')['y'].mean().to_dict()
        
        last_real_date = prophet_df['ds'].max()
        baseline_val = prophet_df.sort_values('ds').tail(26)['y'].median()
        
        # 2. GENERATE SEASONAL SYNTHETIC DATA (2013 - 2025)
        # Instead of 1 point/year, we add 12 points/year to keep the "rhythm"
        synthetic_points = []
        for year in range(2013, 2026):
            growth_mult = (1.025) ** (year - 2012)
            for month in range(1, 13):
                synthetic_date = pd.Timestamp(year=year, month=month, day=15)
                # Apply historical month pattern + growth
                val = seasonal_pattern.get(month, baseline_val) * growth_mult
                synthetic_points.append({'ds': synthetic_date, 'y': val})
        
        guide_df = pd.DataFrame(synthetic_points)
        # Remove columns used for pattern extraction before concat
        full_df = pd.concat([prophet_df[['ds', 'y']], guide_df]).sort_values('ds')

        model = Prophet(
            yearly_seasonality=True, 
            weekly_seasonality=True, 
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.02
        )
        model.fit(full_df)
        
        # 3. FIX PERIOD CALCULATION (Only forecast until end of 2026)
        new_last_date = full_df['ds'].max()
        target_date = pd.Timestamp('2026-12-31')
        days_to_forecast = (target_date - new_last_date).days
        weeks_to_forecast = max(1, (days_to_forecast // 7) + 1)
        
        future = model.make_future_dataframe(periods=weeks_to_forecast, freq='W')
        forecast = model.predict(future)
        
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
