#!/bin/bash
# Setup script for Streamlit Cloud deployment
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[theme]\n\
primaryColor = '#7c4dff'\n\
backgroundColor = '#0e1117'\n\
secondaryBackgroundColor = '#1a1a2e'\n\
textColor = '#fafafa'\n\
font = 'sans serif'\n\
\n\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = \$PORT\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
" > ~/.streamlit/config.toml
