# walmart-stat-site

> A Walmart sales analytics dashboard with Prophet-based forecasting, served by a Flask API and a React frontend.

A containerised analytics app that forecasts Walmart sales and analyses their relationship to economic indicators, presented in an interactive dashboard.

## Features
- 12-week sales forecasting with Facebook Prophet (global and store level)
- Statistical analysis against economic indicators (CPI, fuel price, unemployment)
- Interactive charts with dark-mode support
- Dockerised for one-command startup

## Tech Stack
- **Backend:** Python, Flask, Pandas, Prophet, scikit-learn, SciPy
- **Frontend:** React (via CDN), Tailwind CSS, Chart.js
- **Deploy:** Docker + Docker Compose

## Getting Started
```bash
docker-compose up --build
```
Then open `http://localhost:5000`.
