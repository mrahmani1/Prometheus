/**
 * Critical UI Fixes for Prometheus Platform
 * Addresses immediate breaking issues observed in screenshots
 * Priority: URGENT - Deploy immediately
 */

(function() {
    console.log('🚨 CRITICAL UI FIXES - Applying emergency repairs...');
    
    // Fix 1: Panel Overlap and Z-Index Issues
    function fixPanelOverlaps() {
        const panels = document.querySelectorAll('.panel');
        const baseZIndex = 1000;
        
        panels.forEach((panel, index) => {
            // Set proper z-index hierarchy
            panel.style.zIndex = baseZIndex + (panels.length - index);
            panel.style.position = 'absolute';
            
            // Ensure panels don't overlap
            if (panel.classList.contains('minimized')) {
                panel.style.zIndex = baseZIndex - 100;
            }
            
            // Fix panel headers
            const header = panel.querySelector('.panel-header');
            if (header) {
                header.style.cursor = 'move';
                header.style.userSelect = 'none';
                
                // Make panels draggable
                makeDraggable(panel, header);
            }
            
            // Fix minimize buttons
            const minimizeBtn = panel.querySelector('.minimize-btn');
            if (minimizeBtn) {
                minimizeBtn.style.zIndex = '10000';
                minimizeBtn.style.position = 'relative';
                minimizeBtn.style.pointerEvents = 'auto';
                minimizeBtn.style.cursor = 'pointer';
                
                // Remove duplicate event listeners
                const newBtn = minimizeBtn.cloneNode(true);
                minimizeBtn.parentNode.replaceChild(newBtn, minimizeBtn);
                
                newBtn.addEventListener('click', function(e) {
                    e.stopPropagation();
                    panel.classList.toggle('minimized');
                    if (panel.classList.contains('minimized')) {
                        panel.style.zIndex = baseZIndex - 100;
                    } else {
                        panel.style.zIndex = baseZIndex + (panels.length - index);
                    }
                });
            }
        });
    }
    
    // Fix 2: Volcano Selector Integration
    function fixVolcanoSelector() {
        // Remove duplicate volcano selectors
        const selectors = document.querySelectorAll('.volcano-selector-panel');
        for (let i = 1; i < selectors.length; i++) {
            selectors[i].remove();
        }
        
        // Ensure proper positioning
        const selector = document.querySelector('.volcano-selector-panel');
        if (selector) {
            selector.style.position = 'fixed';
            selector.style.zIndex = '9999';
            selector.style.top = '60px';
            selector.style.right = '20px';
            selector.style.maxHeight = '80vh';
            selector.style.overflowY = 'auto';
        }
    }
    
    // Fix 3: Chart Rendering Issues
    function fixChartRendering() {
        // Destroy and recreate broken charts
        const canvases = document.querySelectorAll('canvas');
        
        canvases.forEach(canvas => {
            if (canvas.id && window.Chart) {
                try {
                    // Get existing chart instance
                    const existingChart = Chart.getChart(canvas.id);
                    if (existingChart) {
                        existingChart.destroy();
                    }
                    
                    // Force canvas size
                    const container = canvas.parentElement;
                    if (container) {
                        canvas.width = container.clientWidth;
                        canvas.height = container.clientHeight || 300;
                    }
                } catch (e) {
                    console.warn('Chart fix for', canvas.id, ':', e.message);
                }
            }
        });
    }
    
    // Fix 4: Map Container Issues
    function fixMapContainers() {
        const maps = document.querySelectorAll('#map, .leaflet-container');
        
        maps.forEach(mapElement => {
            if (mapElement && window.L) {
                // Force map to recalculate size
                const mapId = mapElement.id || mapElement.getAttribute('data-map-id');
                if (mapId && window[mapId + '_map']) {
                    setTimeout(() => {
                        window[mapId + '_map'].invalidateSize();
                    }, 100);
                }
                
                // Fix z-index issues with map controls
                const controls = mapElement.querySelectorAll('.leaflet-control');
                controls.forEach(control => {
                    control.style.zIndex = '1000';
                });
            }
        });
    }
    
    // Fix 5: Dropdown and Select Issues
    function fixDropdowns() {
        const dropdowns = document.querySelectorAll('select');
        
        dropdowns.forEach(dropdown => {
            // Ensure dropdowns are interactive
            dropdown.style.pointerEvents = 'auto';
            dropdown.style.zIndex = '9000';
            dropdown.style.position = 'relative';
            
            // Fix width issues
            if (dropdown.style.width === '0px' || !dropdown.style.width) {
                dropdown.style.width = '100%';
            }
            
            // Ensure visibility
            dropdown.style.opacity = '1';
            dropdown.style.visibility = 'visible';
        });
    }
    
    // Fix 6: Data Layer Checkboxes
    function fixDataLayers() {
        const checkboxes = document.querySelectorAll('input[type="checkbox"]');
        
        checkboxes.forEach(checkbox => {
            // Ensure checkboxes are clickable
            checkbox.style.pointerEvents = 'auto';
            checkbox.style.cursor = 'pointer';
            
            // Fix label associations
            const label = checkbox.parentElement;
            if (label && label.tagName === 'LABEL') {
                label.style.cursor = 'pointer';
                label.style.userSelect = 'none';
            }
        });
    }
    
    // Fix 7: Responsive Grid Layout
    function fixGridLayout() {
        const gridContainers = document.querySelectorAll('.dashboard-grid, .volcano-grid, .panel-grid');
        
        gridContainers.forEach(grid => {
            // Apply modern grid CSS
            grid.style.display = 'grid';
            grid.style.gap = '20px';
            grid.style.padding = '20px';
            
            // Responsive columns
            if (window.innerWidth > 1200) {
                grid.style.gridTemplateColumns = 'repeat(auto-fit, minmax(300px, 1fr))';
            } else if (window.innerWidth > 768) {
                grid.style.gridTemplateColumns = 'repeat(2, 1fr)';
            } else {
                grid.style.gridTemplateColumns = '1fr';
            }
        });
    }
    
    // Fix 8: Button Interactions
    function fixButtons() {
        const buttons = document.querySelectorAll('button, .btn');
        
        buttons.forEach(button => {
            // Ensure buttons are clickable
            button.style.pointerEvents = 'auto';
            button.style.cursor = 'pointer';
            
            // Fix disabled state
            if (!button.disabled) {
                button.style.opacity = '1';
            }
            
            // Add hover effect if missing
            if (!button.matches(':hover')) {
                button.addEventListener('mouseenter', function() {
                    this.style.transform = 'translateY(-2px)';
                    this.style.boxShadow = '0 5px 15px rgba(0,0,0,0.3)';
                });
                
                button.addEventListener('mouseleave', function() {
                    this.style.transform = 'translateY(0)';
                    this.style.boxShadow = '';
                });
            }
        });
    }
    
    // Fix 9: Overflow and Scrolling
    function fixOverflow() {
        // Fix panels with content overflow
        const panelContents = document.querySelectorAll('.panel-content, .panel-body');
        
        panelContents.forEach(content => {
            content.style.overflowY = 'auto';
            content.style.maxHeight = '60vh';
            
            // Custom scrollbar
            content.style.scrollbarWidth = 'thin';
            content.style.scrollbarColor = 'rgba(255,255,255,0.3) rgba(0,0,0,0.3)';
        });
        
        // Fix body overflow
        document.body.style.overflow = 'hidden';
        document.body.style.height = '100vh';
    }
    
    // Fix 10: Navigation Menu
    function fixNavigation() {
        const nav = document.querySelector('nav, .navigation, .header');
        
        if (nav) {
            nav.style.position = 'fixed';
            nav.style.top = '0';
            nav.style.left = '0';
            nav.style.right = '0';
            nav.style.zIndex = '10000';
            nav.style.background = 'rgba(0, 0, 0, 0.9)';
            nav.style.backdropFilter = 'blur(10px)';
            
            // Add padding to body to account for fixed nav
            document.body.style.paddingTop = nav.offsetHeight + 'px';
        }
    }
    
    // Helper: Make elements draggable
    function makeDraggable(element, handle) {
        let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
        
        handle.onmousedown = dragMouseDown;
        
        function dragMouseDown(e) {
            e = e || window.event;
            e.preventDefault();
            pos3 = e.clientX;
            pos4 = e.clientY;
            document.onmouseup = closeDragElement;
            document.onmousemove = elementDrag;
            
            // Bring to front
            const panels = document.querySelectorAll('.panel');
            const maxZ = Math.max(...Array.from(panels).map(p => parseInt(p.style.zIndex) || 0));
            element.style.zIndex = maxZ + 1;
        }
        
        function elementDrag(e) {
            e = e || window.event;
            e.preventDefault();
            pos1 = pos3 - e.clientX;
            pos2 = pos4 - e.clientY;
            pos3 = e.clientX;
            pos4 = e.clientY;
            
            const newTop = element.offsetTop - pos2;
            const newLeft = element.offsetLeft - pos1;
            
            // Boundary checking
            const maxX = window.innerWidth - element.offsetWidth;
            const maxY = window.innerHeight - element.offsetHeight;
            
            element.style.top = Math.min(Math.max(0, newTop), maxY) + "px";
            element.style.left = Math.min(Math.max(0, newLeft), maxX) + "px";
        }
        
        function closeDragElement() {
            document.onmouseup = null;
            document.onmousemove = null;
        }
    }
    
    // Apply all fixes
    function applyAllFixes() {
        console.log('Applying critical UI fixes...');
        
        try {
            fixPanelOverlaps();
            fixVolcanoSelector();
            fixChartRendering();
            fixMapContainers();
            fixDropdowns();
            fixDataLayers();
            fixGridLayout();
            fixButtons();
            fixOverflow();
            fixNavigation();
            
            console.log('✅ Critical UI fixes applied successfully');
        } catch (error) {
            console.error('❌ Error applying UI fixes:', error);
        }
    }
    
    // Apply fixes on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', applyAllFixes);
    } else {
        applyAllFixes();
    }
    
    // Reapply fixes on window resize
    let resizeTimeout;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(function() {
            fixGridLayout();
            fixMapContainers();
            fixChartRendering();
        }, 250);
    });
    
    // Monitor for dynamic content changes
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.addedNodes.length > 0) {
                // Reapply fixes for new content
                setTimeout(function() {
                    fixButtons();
                    fixDropdowns();
                    fixDataLayers();
                }, 100);
            }
        });
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
    
    // Export for manual triggering
    window.applyCriticalUIFixes = applyAllFixes;
    
    console.log('🔧 Critical UI fix system initialized. Call applyCriticalUIFixes() to manually reapply.');
})();