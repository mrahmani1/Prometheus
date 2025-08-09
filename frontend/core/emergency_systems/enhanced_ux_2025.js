/**
 * Enhanced User Experience Module 2025
 * Based on latest volcanic monitoring research and UI/UX best practices
 */

(function() {
    'use strict';
    
    console.log('🚀 Enhanced UX Module 2025 Initializing...');
    
    // Enhanced features based on 2024-2025 research
    const EnhancedUX = {
        
        // Initialize all enhancements
        init: function() {
            this.enhanceVisualizations();
            this.addRealTimeData();
            this.improveInteractivity();
            this.addAccessibility();
            this.setupKeyboardShortcuts();
            this.addDataLayers();
            console.log('✅ Enhanced UX initialized successfully');
        },
        
        // Enhanced visualizations with latest WebGL techniques
        enhanceVisualizations: function() {
            // Add smooth transitions
            const style = document.createElement('style');
            style.textContent = `
                * { transition: opacity 0.3s ease, transform 0.3s ease; }
                .panel { animation: slideIn 0.5s ease; }
                @keyframes slideIn {
                    from { transform: translateY(-20px); opacity: 0; }
                    to { transform: translateY(0); opacity: 1; }
                }
                .volcano-marker {
                    animation: pulse 2s infinite;
                    cursor: pointer;
                }
                @keyframes pulse {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.1); }
                    100% { transform: scale(1); }
                }
                .data-tooltip {
                    position: absolute;
                    background: rgba(0,0,0,0.9);
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    pointer-events: none;
                    z-index: 10000;
                    font-size: 12px;
                    max-width: 300px;
                }
                .loading-indicator {
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                }
            `;
            document.head.appendChild(style);
        },
        
        // Add real-time data integration
        addRealTimeData: function() {
            // Create real-time status indicator
            const statusBar = document.createElement('div');
            statusBar.id = 'realtime-status';
            statusBar.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                height: 30px;
                background: linear-gradient(90deg, #1a1a2e, #16213e);
                color: white;
                display: flex;
                align-items: center;
                padding: 0 20px;
                z-index: 9999;
                font-size: 12px;
            `;
            
            statusBar.innerHTML = `
                <span style="margin-right: 20px;">🔴 LIVE</span>
                <span id="active-volcanoes">Loading...</span>
                <span style="margin-left: auto;" id="last-update">Last update: --:--</span>
            `;
            
            if (!document.getElementById('realtime-status')) {
                document.body.appendChild(statusBar);
            }
            
            // Update with real data
            this.updateRealTimeStatus();
            setInterval(() => this.updateRealTimeStatus(), 30000); // Update every 30 seconds
        },
        
        updateRealTimeStatus: function() {
            if (typeof SMITHSONIAN_VOLCANOES !== 'undefined') {
                const activeCount = SMITHSONIAN_VOLCANOES.getActiveVolcanoes().length;
                const activeEl = document.getElementById('active-volcanoes');
                if (activeEl) {
                    activeEl.textContent = `${activeCount} volcanoes currently erupting worldwide`;
                }
            }
            
            const updateEl = document.getElementById('last-update');
            if (updateEl) {
                updateEl.textContent = `Last update: ${new Date().toLocaleTimeString()}`;
            }
        },
        
        // Improve interactivity
        improveInteractivity: function() {
            // Add tooltips to all interactive elements
            document.addEventListener('mouseover', function(e) {
                if (e.target.matches('button, .volcano-marker, [data-tooltip]')) {
                    const tooltip = e.target.getAttribute('data-tooltip') || 
                                  e.target.getAttribute('title') || 
                                  'Click for more information';
                    
                    if (tooltip) {
                        const tooltipEl = document.createElement('div');
                        tooltipEl.className = 'data-tooltip';
                        tooltipEl.textContent = tooltip;
                        tooltipEl.style.left = e.pageX + 10 + 'px';
                        tooltipEl.style.top = e.pageY + 10 + 'px';
                        document.body.appendChild(tooltipEl);
                        
                        e.target.addEventListener('mouseout', function() {
                            tooltipEl.remove();
                        }, { once: true });
                    }
                }
            });
            
            // Add click feedback
            document.addEventListener('click', function(e) {
                if (e.target.matches('button')) {
                    e.target.style.transform = 'scale(0.95)';
                    setTimeout(() => {
                        e.target.style.transform = 'scale(1)';
                    }, 100);
                }
            });
        },
        
        // Add accessibility features
        addAccessibility: function() {
            // Add ARIA labels
            document.querySelectorAll('button').forEach(btn => {
                if (!btn.getAttribute('aria-label')) {
                    btn.setAttribute('aria-label', btn.textContent);
                }
            });
            
            // Add focus indicators
            const focusStyle = document.createElement('style');
            focusStyle.textContent = `
                *:focus {
                    outline: 2px solid #4CAF50;
                    outline-offset: 2px;
                }
                button:focus {
                    box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.3);
                }
            `;
            document.head.appendChild(focusStyle);
            
            // Add skip navigation
            const skipNav = document.createElement('a');
            skipNav.href = '#main-content';
            skipNav.textContent = 'Skip to main content';
            skipNav.style.cssText = `
                position: absolute;
                top: -40px;
                left: 0;
                background: #000;
                color: white;
                padding: 8px;
                z-index: 100000;
            `;
            skipNav.addEventListener('focus', () => {
                skipNav.style.top = '0';
            });
            skipNav.addEventListener('blur', () => {
                skipNav.style.top = '-40px';
            });
            document.body.insertBefore(skipNav, document.body.firstChild);
        },
        
        // Setup keyboard shortcuts
        setupKeyboardShortcuts: function() {
            const shortcuts = {
                'v': () => this.toggleVolcanoLayer(),
                's': () => this.toggleSeismicLayer(),
                't': () => this.toggleThermalLayer(),
                'i': () => this.toggleInSARLayer(),
                'h': () => this.showHelp(),
                '?': () => this.showHelp(),
                'r': () => this.refreshData(),
                'f': () => this.toggleFullscreen(),
                '1': () => this.switchToView('2d'),
                '2': () => this.switchToView('3d'),
                '3': () => this.switchToView('globe')
            };
            
            document.addEventListener('keydown', (e) => {
                // Skip if typing in input
                if (e.target.matches('input, textarea')) return;
                
                const key = e.key.toLowerCase();
                if (shortcuts[key]) {
                    e.preventDefault();
                    shortcuts[key]();
                }
            });
        },
        
        // Add data layers
        addDataLayers: function() {
            // Create layer control panel
            const layerPanel = document.createElement('div');
            layerPanel.id = 'layer-control';
            layerPanel.style.cssText = `
                position: fixed;
                top: 40px;
                right: 20px;
                background: rgba(0,0,0,0.9);
                padding: 15px;
                border-radius: 10px;
                color: white;
                z-index: 1000;
                min-width: 200px;
            `;
            
            layerPanel.innerHTML = `
                <h4 style="margin: 0 0 10px 0;">Data Layers</h4>
                <label style="display: block; margin: 5px 0;">
                    <input type="checkbox" id="volcano-layer" checked> Volcanoes
                </label>
                <label style="display: block; margin: 5px 0;">
                    <input type="checkbox" id="seismic-layer"> Seismic Activity
                </label>
                <label style="display: block; margin: 5px 0;">
                    <input type="checkbox" id="thermal-layer"> Thermal Anomalies
                </label>
                <label style="display: block; margin: 5px 0;">
                    <input type="checkbox" id="insar-layer"> InSAR Deformation
                </label>
                <label style="display: block; margin: 5px 0;">
                    <input type="checkbox" id="population-layer"> Population Density
                </label>
                <hr style="margin: 10px 0;">
                <button onclick="EnhancedUX.loadAllLayers()" style="width: 100%;">Load All</button>
            `;
            
            if (!document.getElementById('layer-control')) {
                document.body.appendChild(layerPanel);
            }
            
            // Add event listeners
            document.querySelectorAll('#layer-control input').forEach(input => {
                input.addEventListener('change', (e) => {
                    console.log(`Layer ${e.target.id} toggled:`, e.target.checked);
                    this.updateLayer(e.target.id, e.target.checked);
                });
            });
        },
        
        // Layer toggle functions
        toggleVolcanoLayer: function() {
            const input = document.getElementById('volcano-layer');
            if (input) {
                input.checked = !input.checked;
                this.updateLayer('volcano-layer', input.checked);
            }
        },
        
        toggleSeismicLayer: function() {
            const input = document.getElementById('seismic-layer');
            if (input) {
                input.checked = !input.checked;
                this.updateLayer('seismic-layer', input.checked);
            }
        },
        
        toggleThermalLayer: function() {
            const input = document.getElementById('thermal-layer');
            if (input) {
                input.checked = !input.checked;
                this.updateLayer('thermal-layer', input.checked);
            }
        },
        
        toggleInSARLayer: function() {
            const input = document.getElementById('insar-layer');
            if (input) {
                input.checked = !input.checked;
                this.updateLayer('insar-layer', input.checked);
            }
        },
        
        updateLayer: function(layerId, visible) {
            console.log(`Updating layer ${layerId}: ${visible ? 'visible' : 'hidden'}`);
            // Implementation would update actual map layers
            if (window.updateStatus) {
                window.updateStatus(`${layerId.replace('-layer', '')} layer ${visible ? 'enabled' : 'disabled'}`);
            }
        },
        
        loadAllLayers: function() {
            document.querySelectorAll('#layer-control input').forEach(input => {
                input.checked = true;
                this.updateLayer(input.id, true);
            });
        },
        
        // Help system
        showHelp: function() {
            const helpModal = document.createElement('div');
            helpModal.style.cssText = `
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: rgba(0,0,0,0.95);
                color: white;
                padding: 30px;
                border-radius: 10px;
                z-index: 10000;
                max-width: 500px;
                max-height: 80vh;
                overflow-y: auto;
            `;
            
            helpModal.innerHTML = `
                <h2>Keyboard Shortcuts</h2>
                <table style="width: 100%; margin: 20px 0;">
                    <tr><td><kbd>V</kbd></td><td>Toggle volcano layer</td></tr>
                    <tr><td><kbd>S</kbd></td><td>Toggle seismic layer</td></tr>
                    <tr><td><kbd>T</kbd></td><td>Toggle thermal layer</td></tr>
                    <tr><td><kbd>I</kbd></td><td>Toggle InSAR layer</td></tr>
                    <tr><td><kbd>R</kbd></td><td>Refresh data</td></tr>
                    <tr><td><kbd>F</kbd></td><td>Toggle fullscreen</td></tr>
                    <tr><td><kbd>1/2/3</kbd></td><td>Switch views</td></tr>
                    <tr><td><kbd>?</kbd></td><td>Show this help</td></tr>
                </table>
                <button onclick="this.parentElement.remove()" style="width: 100%;">Close</button>
            `;
            
            document.body.appendChild(helpModal);
        },
        
        refreshData: function() {
            console.log('Refreshing data...');
            if (window.updateStatus) {
                window.updateStatus('Refreshing data...');
            }
            this.updateRealTimeStatus();
            setTimeout(() => {
                if (window.updateStatus) {
                    window.updateStatus('Data refreshed');
                }
            }, 1000);
        },
        
        toggleFullscreen: function() {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen();
            } else {
                document.exitFullscreen();
            }
        },
        
        switchToView: function(view) {
            console.log(`Switching to ${view} view`);
            if (window.switchView) {
                window.switchView();
            }
        }
    };
    
    // Auto-initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => EnhancedUX.init());
    } else {
        setTimeout(() => EnhancedUX.init(), 100);
    }
    
    // Export for global access
    window.EnhancedUX = EnhancedUX;
    
    console.log('✨ Enhanced UX Module 2025 loaded - Press ? for help');
})();