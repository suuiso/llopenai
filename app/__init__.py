import os
from flask import Flask, render_template
from dotenv import load_dotenv
from werkzeug.exceptions import RequestEntityTooLarge

from .config import get_config


def create_app() -> Flask:
    load_dotenv()
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )

    # Load configuration from environment
    app.config.from_mapping(get_config())

    # Ensure secret key exists (required for flashing messages)
    if not app.config.get("SECRET_KEY"):
        app.config["SECRET_KEY"] = os.urandom(24)

    # Register blueprints
    from .routes import bp as main_bp

    app.register_blueprint(main_bp)

    # Error handlers
    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(e):
        return (
            render_template(
                "index.html",
                error=f"El archivo excede el tamaño máximo permitido (MAX_CONTENT_LENGTH).",
            ),
            413,
        )

    @app.errorhandler(500)
    def handle_internal_error(e):
        return render_template("index.html", error="Ocurrió un error inesperado."), 500

    return app

