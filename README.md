# Walmart Analytics Pro (Enterprise Edition)

A high-performance analytics dashboard for Walmart sales data featuring AI-driven forecasting and real-time intelligence.

## 🚀 Features

- **Unified Architecture**: Streamlined Flask backend serving a responsive React SPA.
- **Mobile First**: Fully responsive design with drawer navigation and touch-optimized charts.
- **Advanced Analytics**: Integrated **Facebook Prophet** for 12-week sales forecasting at global and store levels.
- **Market Correlation**: Real-time statistical analysis of economic indicators (CPI, Fuel Price, Unemployment).
- **Modern UI/UX**: Built with **React 18**, **Tailwind CSS**, and **Lucide Icons** with support for **Dark Mode**.

## 🛠 Tech Stack

- **Backend**: Python 3.9+, Flask, Pandas, Facebook Prophet.
- **Frontend**: React (via CDN), Tailwind CSS, Chart.js, Lucide Icons.
- **DevOps**: Docker, Docker Compose.

## 🏃 Quick Start

### Using Docker (Recommended)
```bash
docker-compose up --build
```
Access the dashboard at: `http://localhost:5000`

### Manual Setup

1. `cd backend`
2. `pip install -r requirements.txt`
3. `python run.py`

Access the dashboard at: `http://localhost:5000`

## 📊 Analytics Modules
- **Performance Insights**: KPIs and historical sales trends with store-level filtering.
- **AI Prediction Engine**: Machine learning forecasts powered by Prophet.
- **Market Engine**: Pearson correlation coefficients for economic impact analysis.
