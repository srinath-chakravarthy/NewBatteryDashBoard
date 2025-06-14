import panel as pn
from battery_dashboard.main import create_app

if __name__ == "__main__":
    pn.config.autoreload = True
    app = create_app()
    app.show(port=8060, threaded=True)