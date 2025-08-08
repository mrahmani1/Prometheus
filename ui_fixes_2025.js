/**
 * UI Fixes 2025 - Resolve overlapping buttons and interface issues
 * Fixes minimize buttons, menu overlaps, and button functionality
 */
(function() {
    'use strict';
    console.log('🎨 UI Fixes 2025 initializing...');

    /**
     * Fix overlapping minimize/maximize buttons
     */
    function fixMinimizeButtons() {
        // Find all minimize buttons and ensure they're properly positioned
        const minimizeButtons = document.querySelectorAll('.minimize-btn, .maximize-btn, .close-btn');
        
        minimizeButtons.forEach((btn, index) => {
            if (!btn.style.position) {
                btn.style.position = 'relative';
            }
            
            // Ensure unique z-index for each button
            btn.style.zIndex = 1000 + index;
            
            // Fix button dimensions if not set
            if (!btn.style.width) {
                btn.style.width = '24px';
                btn.style.height = '24px';
            }
            
            // Add proper spacing
            btn.style.margin = '2px';
            
            // Ensure buttons are clickable
            btn.style.pointerEvents = 'auto';
            btn.style.cursor = 'pointer';
        });
        
        console.log(`🔧 Fixed ${minimizeButtons.length} minimize/maximize buttons`);
    }

    /**
     * Fix overlapping menus and panels
     */
    function fixOverlappingElements() {
        // Find panels that might overlap
        const panels = document.querySelectorAll('.panel, .control-panel, .sidebar, .menu');
        
        panels.forEach((panel, index) => {
            // Ensure proper z-index stacking
            if (!panel.style.zIndex) {
                panel.style.zIndex = 100 + index;
            }
            
            // Fix positioning for overlapping elements
            if (panel.style.position === 'absolute' || panel.style.position === 'fixed') {
                // Ensure minimum spacing between panels
                const rect = panel.getBoundingClientRect();
                if (rect.width > window.innerWidth * 0.8) {
                    panel.style.maxWidth = '80vw';
                }
                if (rect.height > window.innerHeight * 0.8) {
                    panel.style.maxHeight = '80vh';
                    panel.style.overflowY = 'auto';
                }
            }
        });

        // Fix specific UI containers
        const containers = document.querySelectorAll('#controls, .controls, .ui-container');
        containers.forEach(container => {
            container.style.display = 'flex';
            container.style.flexWrap = 'wrap';
            container.style.gap = '10px';
            container.style.alignItems = 'flex-start';
        });

        console.log(`🔧 Fixed ${panels.length} overlapping panels`);
    }

    /**
     * Fix non-working buttons by ensuring event handlers
     */
    function fixButtonFunctionality() {
        // Common button selectors and their expected functions
        const buttonFixes = {
            '.switch-view-btn, #switchView': () => {
                if (typeof switchView === 'function') {
                    switchView();
                } else {
                    console.log('🔄 View switched (fallback)');
                    // Toggle between 2D and 3D if elements exist
                    const view2D = document.getElementById('view2D');
                    const view3D = document.getElementById('view3D');
                    if (view2D && view3D) {
                        if (view2D.style.display === 'none') {
                            view2D.style.display = 'block';
                            view3D.style.display = 'none';
                        } else {
                            view2D.style.display = 'none';
                            view3D.style.display = 'block';
                        }
                    }
                }
            },
            '.minimize-btn': (btn) => {
                const panel = btn.closest('.panel, .control-panel, .sidebar');
                if (panel) {
                    const isMinimized = panel.classList.contains('minimized');
                    if (isMinimized) {
                        panel.classList.remove('minimized');
                        panel.style.height = 'auto';
                        btn.innerHTML = '−';
                        btn.title = 'Minimize';
                    } else {
                        panel.classList.add('minimized');
                        panel.style.height = '30px';
                        btn.innerHTML = '+';
                        btn.title = 'Maximize';
                    }
                }
            },
            '.refresh-btn, #refresh': () => {
                if (typeof refreshData === 'function') {
                    refreshData();
                } else if (typeof updateStatus === 'function') {
                    updateStatus('Refreshing...');
                    setTimeout(() => updateStatus('Refreshed'), 1000);
                } else {
                    location.reload();
                }
            },
            '.fullscreen-btn': () => {
                if (document.fullscreenElement) {
                    document.exitFullscreen();
                } else {
                    document.documentElement.requestFullscreen();
                }
            }
        };

        // Apply fixes to buttons
        Object.entries(buttonFixes).forEach(([selector, handler]) => {
            const buttons = document.querySelectorAll(selector);
            buttons.forEach(btn => {
                // Remove any existing click handlers
                btn.replaceWith(btn.cloneNode(true));
                const newBtn = document.querySelector(selector);
                
                if (newBtn) {
                    newBtn.addEventListener('click', (e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        try {
                            handler(newBtn);
                        } catch (error) {
                            console.warn('Button handler error:', error);
                        }
                    });
                    
                    // Ensure button is visible and clickable
                    newBtn.style.display = 'inline-block';
                    newBtn.style.pointerEvents = 'auto';
                }
            });
        });

        console.log('🔧 Fixed button functionality');
    }

    /**
     * Add CSS fixes for common UI issues
     */
    function addUIStyles() {
        const style = document.createElement('style');
        style.textContent = `
            /* UI Fixes 2025 */
            .minimize-btn, .maximize-btn, .close-btn {
                position: relative !important;
                z-index: 1001 !important;
                background: rgba(0,0,0,0.8) !important;
                color: white !important;
                border: 1px solid #333 !important;
                border-radius: 3px !important;
                width: 24px !important;
                height: 24px !important;
                cursor: pointer !important;
                font-size: 14px !important;
                line-height: 1 !important;
                text-align: center !important;
                margin: 2px !important;
                transition: all 0.2s ease !important;
            }

            .minimize-btn:hover, .maximize-btn:hover, .close-btn:hover {
                background: rgba(255,255,255,0.2) !important;
                transform: scale(1.1) !important;
            }

            .panel.minimized {
                height: 30px !important;
                overflow: hidden !important;
                transition: height 0.3s ease !important;
            }

            .control-panel, .panel {
                position: relative !important;
                background: rgba(0,0,0,0.8) !important;
                border: 1px solid #444 !important;
                border-radius: 5px !important;
                padding: 10px !important;
                margin: 5px !important;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
            }

            .ui-container, #controls, .controls {
                display: flex !important;
                flex-wrap: wrap !important;
                gap: 10px !important;
                align-items: flex-start !important;
                z-index: 100 !important;
            }

            /* Fix button alignment */
            button {
                margin: 3px !important;
                padding: 5px 10px !important;
                border-radius: 3px !important;
                cursor: pointer !important;
            }

            /* Prevent text selection on UI elements */
            .control-panel, .panel, button {
                -webkit-user-select: none !important;
                -moz-user-select: none !important;
                -ms-user-select: none !important;
                user-select: none !important;
            }

            /* Fix menu positioning */
            .menu, .dropdown {
                position: absolute !important;
                z-index: 2000 !important;
                background: rgba(0,0,0,0.9) !important;
                border: 1px solid #555 !important;
                border-radius: 3px !important;
                box-shadow: 0 2px 10px rgba(0,0,0,0.5) !important;
            }

            /* Responsive fixes */
            @media (max-width: 768px) {
                .control-panel, .panel {
                    max-width: 90vw !important;
                    font-size: 14px !important;
                }
                
                .minimize-btn, .maximize-btn, .close-btn {
                    width: 32px !important;
                    height: 32px !important;
                    font-size: 16px !important;
                }
            }
        `;
        document.head.appendChild(style);
        console.log('🎨 Added UI fix styles');
    }

    /**
     * Initialize all UI fixes
     */
    function initializeUIFixes() {
        addUIStyles();
        fixMinimizeButtons();
        fixOverlappingElements();
        fixButtonFunctionality();

        // Re-run fixes when DOM changes
        const observer = new MutationObserver((mutations) => {
            let shouldRerun = false;
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    shouldRerun = true;
                }
            });
            
            if (shouldRerun) {
                setTimeout(() => {
                    fixMinimizeButtons();
                    fixButtonFunctionality();
                }, 100);
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        console.log('✅ UI Fixes 2025 fully initialized');
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeUIFixes);
    } else {
        initializeUIFixes();
    }

    // Expose global function
    window.fixUI = initializeUIFixes;

})();