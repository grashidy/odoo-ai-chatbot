import os
bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"
timeout = 120
workers = 1
