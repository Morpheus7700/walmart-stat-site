from app import app
import os

if __name__ == '__main__':
    # Default to 5000 for Docker/Enterprise standard, but allow override
    port = int(os.environ.get("PORT", 5000))
    # host '0.0.0.0' is essential for Docker and mobile access on local network
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
