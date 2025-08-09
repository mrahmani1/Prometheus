// Universal WebGL Fix and Minimize Functionality
// Add this script to all HTML files to fix WebGL warning and add minimize functionality

// WebGL Extension Fix
function fixWebGLExtensions(renderer) {
    if (renderer && renderer.capabilities) {
        const gl = renderer.getContext();
        if (gl) {
            // Explicitly enable color buffer float extensions
            const extensions = [
                'EXT_color_buffer_float',
                'WEBGL_color_buffer_float',
                'OES_texture_float',
                'OES_texture_float_linear',
                'EXT_float_blend',
                'WEBGL_depth_texture',
                'EXT_frag_depth',
                'EXT_shader_texture_lod',
                'EXT_sRGB'
            ];
            
            extensions.forEach(ext => {
                try {
                    const extension = gl.getExtension(ext);
                    if (extension) {
                        console.log(`✓ Enabled WebGL extension: ${ext}`);
                    }
                } catch (e) {
                    console.warn(`Could not enable extension: ${ext}`);
                }
            });
        }
    }
}

// Universal Panel Minimize/Maximize Functionality
class PanelManager {
    constructor() {
        this.panels = new Map();
        this.minimizedStates = this.loadStates();
        this.init();
    }
    
    init() {
        // Find all panels with class names containing 'panel', 'control', or 'controls'
        const panelSelectors = [
            '[class*="panel"]',
            '[class*="control"]',
            '[id*="control"]',
            '[id*="panel"]'
        ];
        
        panelSelectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(panel => {
                if (!this.panels.has(panel.id || panel.className)) {
                    this.addMinimizeButton(panel);
                }
            });
        });
        
        // Add keyboard shortcuts
        this.setupKeyboardShortcuts();
    }
    
    addMinimizeButton(panel) {
        // Skip if already has minimize button
        if (panel.querySelector('.minimize-btn')) return;
        
        // Create minimize button
        const btn = document.createElement('button');
        btn.className = 'minimize-btn';
        btn.innerHTML = '−';
        btn.style.cssText = `
            position: absolute;
            top: 5px;
            right: 5px;
            width: 25px;
            height: 25px;
            border: 1px solid rgba(255,255,255,0.3);
            background: rgba(0,0,0,0.5);
            color: white;
            border-radius: 3px;
            cursor: pointer;
            font-size: 18px;
            line-height: 1;
            z-index: 10000;
            transition: all 0.3s;
        `;
        
        btn.onmouseover = () => {
            btn.style.background = 'rgba(76, 175, 80, 0.5)';
            btn.style.transform = 'scale(1.1)';
        };
        
        btn.onmouseout = () => {
            btn.style.background = 'rgba(0,0,0,0.5)';
            btn.style.transform = 'scale(1)';
        };
        
        // Make panel relative positioned if not already
        const position = window.getComputedStyle(panel).position;
        if (position === 'static') {
            panel.style.position = 'relative';
        }
        
        // Store original content
        const panelId = panel.id || `panel-${this.panels.size}`;
        if (!panel.id) panel.id = panelId;
        
        const originalHeight = panel.style.height || 'auto';
        const originalOverflow = panel.style.overflow || 'visible';
        
        this.panels.set(panelId, {
            element: panel,
            originalHeight,
            originalOverflow,
            minimized: this.minimizedStates[panelId] || false
        });
        
        // Add click handler
        btn.onclick = () => this.togglePanel(panelId);
        
        // Insert button
        panel.insertBefore(btn, panel.firstChild);
        
        // Apply saved state
        if (this.minimizedStates[panelId]) {
            this.minimizePanel(panelId, false);
        }
    }
    
    togglePanel(panelId) {
        const panelData = this.panels.get(panelId);
        if (!panelData) return;
        
        if (panelData.minimized) {
            this.maximizePanel(panelId);
        } else {
            this.minimizePanel(panelId);
        }
    }
    
    minimizePanel(panelId, animate = true) {
        const panelData = this.panels.get(panelId);
        if (!panelData) return;
        
        const panel = panelData.element;
        const btn = panel.querySelector('.minimize-btn');
        
        // Store original dimensions
        panelData.originalHeight = panel.offsetHeight + 'px';
        
        // Get title or first heading
        const title = panel.querySelector('h1, h2, h3, h4, .title') || 
                     panel.querySelector('[class*="title"]');
        const titleHeight = title ? title.offsetHeight : 30;
        
        // Minimize with animation
        if (animate) {
            panel.style.transition = 'all 0.3s ease-in-out';
        }
        
        panel.style.height = (titleHeight + 10) + 'px';
        panel.style.overflow = 'hidden';
        
        // Hide all children except title and button
        Array.from(panel.children).forEach(child => {
            if (child !== btn && child !== title) {
                child.style.display = 'none';
            }
        });
        
        btn.innerHTML = '+';
        panelData.minimized = true;
        
        // Save state
        this.saveStates();
    }
    
    maximizePanel(panelId, animate = true) {
        const panelData = this.panels.get(panelId);
        if (!panelData) return;
        
        const panel = panelData.element;
        const btn = panel.querySelector('.minimize-btn');
        
        // Restore with animation
        if (animate) {
            panel.style.transition = 'all 0.3s ease-in-out';
        }
        
        panel.style.height = panelData.originalHeight;
        panel.style.overflow = panelData.originalOverflow;
        
        // Show all children
        Array.from(panel.children).forEach(child => {
            child.style.display = '';
        });
        
        btn.innerHTML = '−';
        panelData.minimized = false;
        
        // Save state
        this.saveStates();
    }
    
    minimizeAll() {
        this.panels.forEach((data, id) => {
            if (!data.minimized) {
                this.minimizePanel(id);
            }
        });
    }
    
    maximizeAll() {
        this.panels.forEach((data, id) => {
            if (data.minimized) {
                this.maximizePanel(id);
            }
        });
    }
    
    toggleAll() {
        const anyMaximized = Array.from(this.panels.values()).some(d => !d.minimized);
        if (anyMaximized) {
            this.minimizeAll();
        } else {
            this.maximizeAll();
        }
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl+M or Cmd+M to toggle all panels
            if ((e.ctrlKey || e.metaKey) && e.key === 'm') {
                e.preventDefault();
                this.toggleAll();
            }
            
            // Ctrl+Shift+M to maximize all
            if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'M') {
                e.preventDefault();
                this.maximizeAll();
            }
        });
    }
    
    saveStates() {
        const states = {};
        this.panels.forEach((data, id) => {
            states[id] = data.minimized;
        });
        localStorage.setItem('panelStates', JSON.stringify(states));
    }
    
    loadStates() {
        try {
            return JSON.parse(localStorage.getItem('panelStates') || '{}');
        } catch {
            return {};
        }
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.panelManager = new PanelManager();
    });
} else {
    window.panelManager = new PanelManager();
}

// Export functions for use in other scripts
window.fixWebGLExtensions = fixWebGLExtensions;
window.PanelManager = PanelManager;