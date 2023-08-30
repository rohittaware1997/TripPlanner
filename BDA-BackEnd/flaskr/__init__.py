import os

from flask import Flask
from flask_redis import FlaskRedis
from flask_cors import CORS

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    CORS(app)
    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    from . import categories
    app.register_blueprint(categories.bp)

    from . import hotels
    app.register_blueprint(hotels.bp)

    return app
