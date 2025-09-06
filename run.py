from app import create_app


app = create_app()


if __name__ == "__main__":
    try:
        from waitress import serve

        serve(app, host="0.0.0.0", port=8000)
    except Exception:
        # Fallback to Flask's built-in server for local/dev usage
        app.run(host="0.0.0.0", port=8000, debug=True)
