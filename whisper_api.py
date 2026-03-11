"""
Backward compatibility wrapper for whisper_api.py
Imports the refactored application from app/main.py
"""

from app.main import app

# Re-export for backward compatibility
__all__ = ["app"]

if __name__ == "__main__":
    # Just run the app's main function - no code duplication!
    import app.main
    app.main.__main__()
