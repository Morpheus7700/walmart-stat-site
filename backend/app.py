from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_caching import Cache
import pandas as pd
import numpy as np
from prophet import Prophet
import os
import traceback
import logging
from werkzeug.utils import secure_filename
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Cache configuration
cache = Cache(app, config={'CACHE_TYPE': 'simple', 'CACHE_DEFAULT_TIMEOUT': 3600})

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'Walmart.csv')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# State management
class DataState:
    def __init__(self):
        self.df = None
        self.error = None
        self.is_user_data = False
        self.load_base_data()

    def load_base_data(self):
        try:
            if not os.path.exists(DATA_PATH):
                self.error = "Base data file not found"
                return
            self.df = pd.read_csv(DATA_PATH)
            self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d-%m-%Y')
            self.is_user_data = False
            self.error = None
            logger.info("Base data loaded.")
        except Exception as e:
            self.error = str(e)
            logger.error(f"Error loading base data: {self.error}")

    def set_user_data(self, df):
        self.df = df
        self.is_user_data = True
        self.error = None
        cache.clear() # Clear all cached stats/forecasts for new data
        logger.info("User data active.")

state = DataState()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            new_df = pd.read_csv(filepath)
            # Basic validation
            required_cols = ['Date', 'Weekly_Sales']
            if not all(col in new_df.columns for col in required_cols):
                return jsonify({"error": f"Missing required columns: {required_cols}"}), 400
            
            # Try to parse date
            try:
                new_df['Date'] = pd.to_datetime(new_df['Date'])
            except:
                # Try specific format if default fails
                new_df['Date'] = pd.to_datetime(new_df['Date'], dayfirst=True)
            
            state.set_user_data(new_df)
            return jsonify({
                "message": "Data uploaded successfully",
                "rows": len(new_df),
                "is_user_data": True
            })
        except Exception as e:
            return jsonify({"error": f"Invalid CSV format: {str(e)}"}), 400
    
    return jsonify({"error": "Only CSV files are allowed"}), 400

@app.route('/api/reset', methods=['POST'])
def reset_data():
    state.load_base_data()
    cache.clear()
    return jsonify({"message": "Reset to base data", "is_user_data": False})

@app.route('/api/status')
def get_status():
    return jsonify({
        "data_loaded": state.df is not None,
        "load_error": state.error,
        "rows": len(state.df) if state.df is not None else 0,
        "is_user_data": state.is_user_data,
        "cache_enabled": True
    })

@app.route('/api/anomalies')
@cache.memoize(timeout=3600)
def get_anomalies():
    if state.df is None: return jsonify({"error": "No data"}), 500
    try:
        data = state.df.groupby('Date')['Weekly_Sales'].sum().reset_index()
        # Simple Z-score anomaly detection
        z_scores = np.abs(stats.zscore(data['Weekly_Sales']))
        anomalies = data[z_scores > 2.5] # Points further than 2.5 std devs
        
        return jsonify({
            "dates": anomalies['Date'].dt.strftime('%Y-%m-%d').tolist(),
            "values": anomalies['Weekly_Sales'].tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/insights')
@cache.memoize(timeout=3600)
def get_insights():
    if state.df is None: return jsonify({"error": "No data"}), 500
    try:
        data = state.df.groupby('Date')['Weekly_Sales'].sum().sort_values(ascending=True).reset_index()
        # Re-sort by date for trend analysis
        data = data.sort_values('Date')
        recent_data = data.tail(8)
        
        if len(recent_data) < 4:
            return jsonify(["Insufficient data for trend analysis."])

        # Trend calculation
        first_half = recent_data.head(len(recent_data)//2)['Weekly_Sales'].mean()
        second_half = recent_data.tail(len(recent_data)//2)['Weekly_Sales'].mean()
        trend_pct = ((second_half - first_half) / first_half) * 100
        
        insights = []
        if trend_pct > 5:
            insights.append(f"Strong upward trend detected: Sales increased by {trend_pct:.1f}% over the last few weeks.")
        elif trend_pct < -5:
            insights.append(f"Downward trend warning: Sales decreased by {abs(trend_pct):.1f}% recently.")
        else:
            insights.append("Stable performance: Sales are maintaining a steady plateau.")

        # Peak detection
        max_row = data.loc[data['Weekly_Sales'].idxmax()]
        insights.append(f"Historical peak of ${max_row['Weekly_Sales']:,.0f} occurred on {max_row['Date'].strftime('%Y-%m-%d')}.")
        
        return jsonify(insights)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats')
@cache.memoize(timeout=600)
def get_stats():
    store_id = request.args.get('store', type=int)
    if state.df is None: return jsonify({"error": state.error}), 500
    try:
        data = state.df
        if store_id is not None and 'Store' in data.columns:
            data = data[data['Store'] == store_id]
        
        if data.empty: return jsonify({"error": "No data available"}), 404
        
        res = {
            "total_revenue": f"${data['Weekly_Sales'].sum():,.0f}",
            "avg_sales": f"${data['Weekly_Sales'].mean():,.0f}",
        }
        
        if 'CPI' in data.columns:
            res["cpi"] = f"{data['CPI'].iloc[-1]:.2f}"
        else:
            res["cpi"] = "N/A"
            
        if 'Unemployment' in data.columns:
            res["unemployment"] = f"{data['Unemployment'].mean():.2f}%"
        else:
            res["unemployment"] = "N/A"
            
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chart')
@cache.memoize(timeout=600)
def get_chart():
    store_id = request.args.get('store', type=int)
    if state.df is None: return jsonify({"error": "No data"}), 500
    try:
        data = state.df
        if store_id is not None and 'Store' in data.columns:
            data = data[data['Store'] == store_id]
            
        res = data.groupby('Date')['Weekly_Sales'].sum().reset_index()
        res = res.sort_values('Date')
        return jsonify({
            "labels": res['Date'].dt.strftime('%Y-%m-%d').tolist(),
            "data": res['Weekly_Sales'].tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/correlation')
@cache.cached(timeout=3600)
def get_correlation():
    if state.df is None: return jsonify({"error": "No data"}), 500
    try:
        cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        available_cols = [c for c in cols if c in state.df.columns]
        if len(available_cols) < 2:
            return jsonify({"error": "Insufficient columns for correlation"})
            
        corr_matrix = state.df[available_cols].corr()['Weekly_Sales'].to_dict()
        return jsonify(corr_matrix)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stores')
@cache.cached(timeout=86400)
def get_stores():
    if state.df is None: return jsonify({"error": "No data"}), 500
    if 'Store' not in state.df.columns:
        return jsonify([])
    return jsonify(sorted(state.df['Store'].unique().tolist()))

@app.route('/api/forecast')
@cache.memoize(timeout=3600)
def get_forecast():
    store_id = request.args.get('store', type=int)
    if state.df is None: return jsonify({"error": "No data"}), 500
    try:
        data = state.df
        if store_id is not None and 'Store' in data.columns:
            data = data[data['Store'] == store_id]
            
        prophet_df = data.groupby('Date')['Weekly_Sales'].sum().reset_index()
        prophet_df.columns = ['ds', 'y']
        
        # Determine if we need synthetic data (only for the specific Walmart base data scenario)
        if not state.is_user_data and len(prophet_df) < 200: # Heuristic for original base data
            prophet_df['month'] = prophet_df['ds'].dt.month
            seasonal_pattern = prophet_df.groupby('month')['y'].mean().to_dict()
            last_real_date = prophet_df['ds'].max()
            baseline_val = prophet_df.sort_values('ds').tail(26)['y'].median()
            
            synthetic_points = []
            for year in range(2013, 2026):
                growth_mult = (1.025) ** (year - 2012)
                for month in range(1, 13):
                    synthetic_date = pd.Timestamp(year=year, month=month, day=15)
                    val = seasonal_pattern.get(month, baseline_val) * growth_mult
                    synthetic_points.append({'ds': synthetic_date, 'y': val})
            
            guide_df = pd.DataFrame(synthetic_points)
            full_df = pd.concat([prophet_df[['ds', 'y']], guide_df]).sort_values('ds')
        else:
            full_df = prophet_df[['ds', 'y']].sort_values('ds')

        model = Prophet(
            yearly_seasonality=True, 
            weekly_seasonality=True, 
            seasonality_mode='multiplicative' if not state.is_user_data else 'additive',
            changepoint_prior_scale=0.02,
            interval_width=0.80 # 80% confidence interval
        )
        model.fit(full_df)
        
        new_last_date = full_df['ds'].max()
        # Forecast 12 weeks ahead if user data, otherwise until 2026
        if state.is_user_data:
            weeks_to_forecast = 12
        else:
            target_date = pd.Timestamp('2026-12-31')
            days_to_forecast = (target_date - new_last_date).days
            weeks_to_forecast = max(1, (days_to_forecast // 7) + 1)
        
        future = model.make_future_dataframe(periods=weeks_to_forecast, freq='W')
        forecast = model.predict(future)
        
        return jsonify({
            "labels": forecast['ds'].dt.strftime('%Y-%m-%d').tolist(),
            "forecast": forecast['yhat'].tolist(),
            "upper": forecast['yhat_upper'].tolist(),
            "lower": forecast['yhat_lower'].tolist(),
            "actual": prophet_df['y'].tolist(),
            "historical_labels": prophet_df['ds'].dt.strftime('%Y-%m-%d').tolist()
        })
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
