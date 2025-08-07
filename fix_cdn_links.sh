#!/bin/bash

# Fix Plotly CDN links in all HTML files
echo "Updating Plotly CDN links to latest version..."

# List of files to update
files=(
    "gplates_advanced.html"
    "ml_dashboard.html"
    "professional_tectonic_simulation.html"
    "volcano_magma_simulation.html"
)

# Replace old Plotly CDN with new one
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "Updating $file..."
        sed -i 's|https://cdn.plot.ly/plotly-latest.min.js|https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.18.2/plotly.min.js|g' "$file"
    fi
done

echo "CDN links updated!"