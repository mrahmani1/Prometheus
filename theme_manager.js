/**
 * Theme Manager - Task 2.1
 * Advanced theme management with system preference detection and custom themes
 * @author Prometheus Platform
 * @version 2.0.0
 */

(function(global) {
    'use strict';
    
    console.log('🎨 Theme Manager - Initializing advanced theming system...');
    
    class ThemeManager {
        constructor() {
            this.currentTheme = 'dark';
            this.customThemes = new Map();
            this.observers = new Set();
            this.mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
            this.systemThemeEnabled = false;
            
            // Default themes
            this.themes = {
                dark: {
                    name: 'Dark',
                    icon: '🌙',
                    colors: {
                        primary: '#0a0a0a',
                        secondary: '#1a1a1a',
                        tertiary: '#2a2a2a',
                        accent: '#FF6B35',
                        accentHover: '#ff8555',
                        success: '#4CAF50',
                        warning: '#FF9800',
                        error: '#f44336',
                        info: '#2196F3',
                        textPrimary: '#ffffff',
                        textSecondary: '#b0b0b0',
                        border: 'rgba(255, 255, 255, 0.1)',
                        shadow: 'rgba(0, 0, 0, 0.5)',
                        overlay: 'rgba(0, 0, 0, 0.7)'
                    }
                },
                light: {
                    name: 'Light',
                    icon: '☀️',
                    colors: {
                        primary: '#ffffff',
                        secondary: '#f5f5f5',
                        tertiary: '#e0e0e0',
                        accent: '#FF6B35',
                        accentHover: '#ff8555',
                        success: '#4CAF50',
                        warning: '#FF9800',
                        error: '#f44336',
                        info: '#2196F3',
                        textPrimary: '#1a1a1a',
                        textSecondary: '#666666',
                        border: 'rgba(0, 0, 0, 0.1)',
                        shadow: 'rgba(0, 0, 0, 0.1)',
                        overlay: 'rgba(0, 0, 0, 0.5)'
                    }
                },
                volcano: {
                    name: 'Volcano',
                    icon: '🌋',
                    colors: {
                        primary: '#1a0a0a',
                        secondary: '#2a1515',
                        tertiary: '#3a2020',
                        accent: '#FF6B35',
                        accentHover: '#ff8555',
                        success: '#FF6B35',
                        warning: '#FFA726',
                        error: '#EF5350',
                        info: '#FF7043',
                        textPrimary: '#FFE0B2',
                        textSecondary: '#FFAB91',
                        border: 'rgba(255, 107, 53, 0.2)',
                        shadow: 'rgba(255, 107, 53, 0.3)',
                        overlay: 'rgba(26, 10, 10, 0.8)'
                    }
                },
                ocean: {
                    name: 'Ocean',
                    icon: '🌊',
                    colors: {
                        primary: '#001a33',
                        secondary: '#002244',
                        tertiary: '#003366',
                        accent: '#00ACC1',
                        accentHover: '#00BCD4',
                        success: '#00E676',
                        warning: '#FFD54F',
                        error: '#FF5252',
                        info: '#40C4FF',
                        textPrimary: '#E0F7FA',
                        textSecondary: '#B2EBF2',
                        border: 'rgba(0, 172, 193, 0.2)',
                        shadow: 'rgba(0, 172, 193, 0.3)',
                        overlay: 'rgba(0, 26, 51, 0.8)'
                    }
                },
                matrix: {
                    name: 'Matrix',
                    icon: '💊',
                    colors: {
                        primary: '#000000',
                        secondary: '#0a0a0a',
                        tertiary: '#1a1a1a',
                        accent: '#00ff00',
                        accentHover: '#00ff88',
                        success: '#00ff00',
                        warning: '#ffff00',
                        error: '#ff0000',
                        info: '#00ffff',
                        textPrimary: '#00ff00',
                        textSecondary: '#008800',
                        border: 'rgba(0, 255, 0, 0.2)',
                        shadow: 'rgba(0, 255, 0, 0.3)',
                        overlay: 'rgba(0, 0, 0, 0.9)'
                    }
                },
                midnight: {
                    name: 'Midnight',
                    icon: '🌃',
                    colors: {
                        primary: '#0f0f23',
                        secondary: '#1a1a2e',
                        tertiary: '#252542',
                        accent: '#f39c12',
                        accentHover: '#f1c40f',
                        success: '#27ae60',
                        warning: '#e67e22',
                        error: '#e74c3c',
                        info: '#3498db',
                        textPrimary: '#ecf0f1',
                        textSecondary: '#95a5a6',
                        border: 'rgba(236, 240, 241, 0.1)',
                        shadow: 'rgba(0, 0, 0, 0.6)',
                        overlay: 'rgba(15, 15, 35, 0.9)'
                    }
                },
                highContrast: {
                    name: 'High Contrast',
                    icon: '⚡',
                    colors: {
                        primary: '#000000',
                        secondary: '#ffffff',
                        tertiary: '#ffff00',
                        accent: '#ff0000',
                        accentHover: '#ff6666',
                        success: '#00ff00',
                        warning: '#ffff00',
                        error: '#ff0000',
                        info: '#00ffff',
                        textPrimary: '#ffffff',
                        textSecondary: '#ffff00',
                        border: 'rgba(255, 255, 255, 1)',
                        shadow: 'rgba(255, 255, 255, 0.5)',
                        overlay: 'rgba(0, 0, 0, 1)'
                    }
                }
            };
            
            this.init();
        }
        
        init() {
            console.log('Initializing Theme Manager...');
            
            // Load saved theme preference
            this.loadThemePreference();
            
            // Setup system theme detection
            this.setupSystemThemeDetection();
            
            // Apply initial theme
            this.applyTheme(this.currentTheme);
            
            // Setup theme persistence
            this.setupThemePersistence();
            
            // Create theme switcher UI
            this.createThemeSwitcherUI();
            
            console.log('✅ Theme Manager initialized');
        }
        
        loadThemePreference() {
            const savedTheme = localStorage.getItem('prometheus-theme');
            const systemThemeEnabled = localStorage.getItem('prometheus-system-theme') === 'true';
            
            this.systemThemeEnabled = systemThemeEnabled;
            
            if (systemThemeEnabled) {
                this.currentTheme = this.getSystemTheme();
            } else if (savedTheme && this.themes[savedTheme]) {
                this.currentTheme = savedTheme;
            } else {
                this.currentTheme = this.getSystemTheme();
            }
        }
        
        setupSystemThemeDetection() {
            // Listen for system theme changes
            this.mediaQuery.addEventListener('change', (e) => {
                if (this.systemThemeEnabled) {
                    const newTheme = e.matches ? 'dark' : 'light';
                    this.setTheme(newTheme, false);
                }
            });
        }
        
        getSystemTheme() {
            return this.mediaQuery.matches ? 'dark' : 'light';
        }
        
        applyTheme(themeName, animate = false) {
            const theme = this.themes[themeName] || this.themes.dark;
            
            // Create or update style element
            let styleEl = document.getElementById('theme-manager-styles');
            if (!styleEl) {
                styleEl = document.createElement('style');
                styleEl.id = 'theme-manager-styles';
                document.head.appendChild(styleEl);
            }
            
            // Generate CSS variables
            const cssVars = Object.entries(theme.colors).map(([key, value]) => {
                const varName = key.replace(/([A-Z])/g, '-$1').toLowerCase();
                return `--${varName}: ${value};`;
            }).join('\n    ');
            
            // Apply styles
            styleEl.textContent = `
                :root {
                    ${cssVars}
                    --theme-transition: ${animate ? 'all 0.3s ease' : 'none'};
                }
                
                body {
                    background-color: var(--primary);
                    color: var(--text-primary);
                    transition: var(--theme-transition);
                }
                
                /* Theme-specific adjustments */
                .theme-${themeName} {
                    /* Custom styles for this theme */
                }
                
                /* Smooth transitions */
                * {
                    transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
                }
                
                /* Special effects for specific themes */
                ${this.getSpecialEffects(themeName)}
            `;
            
            // Update body class
            document.body.className = document.body.className
                .replace(/theme-\w+/g, '')
                .trim() + ` theme-${themeName}`;
            
            // Update data attribute
            document.documentElement.setAttribute('data-theme', themeName);
            
            // Notify observers
            this.notifyObservers(themeName, theme);
        }
        
        getSpecialEffects(themeName) {
            const effects = {
                matrix: `
                    body.theme-matrix::before {
                        content: '';
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background: linear-gradient(
                            transparent 0%,
                            rgba(0, 255, 0, 0.03) 50%,
                            transparent 100%
                        );
                        animation: matrix-rain 20s linear infinite;
                        pointer-events: none;
                        z-index: 1;
                    }
                    
                    @keyframes matrix-rain {
                        0% { transform: translateY(-100%); }
                        100% { transform: translateY(100%); }
                    }
                `,
                volcano: `
                    body.theme-volcano::after {
                        content: '';
                        position: fixed;
                        bottom: -50px;
                        left: 50%;
                        transform: translateX(-50%);
                        width: 200px;
                        height: 200px;
                        background: radial-gradient(
                            circle,
                            rgba(255, 107, 53, 0.4) 0%,
                            transparent 70%
                        );
                        animation: lava-glow 4s ease-in-out infinite;
                        pointer-events: none;
                    }
                    
                    @keyframes lava-glow {
                        0%, 100% { opacity: 0.3; transform: translateX(-50%) scale(1); }
                        50% { opacity: 0.6; transform: translateX(-50%) scale(1.2); }
                    }
                `,
                ocean: `
                    body.theme-ocean {
                        background: linear-gradient(
                            180deg,
                            var(--primary) 0%,
                            var(--secondary) 50%,
                            var(--tertiary) 100%
                        );
                    }
                    
                    body.theme-ocean::before {
                        content: '';
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><path d="M0,50 Q25,30 50,50 T100,50 L100,100 L0,100 Z" fill="rgba(0,172,193,0.1)"/></svg>');
                        background-size: 200% 100%;
                        animation: waves 10s linear infinite;
                        pointer-events: none;
                    }
                    
                    @keyframes waves {
                        0% { background-position-x: 0; }
                        100% { background-position-x: 200%; }
                    }
                `
            };
            
            return effects[themeName] || '';
        }
        
        setTheme(themeName, savePreference = true) {
            if (!this.themes[themeName]) {
                console.warn(`Theme "${themeName}" not found`);
                return;
            }
            
            this.currentTheme = themeName;
            this.applyTheme(themeName, true);
            
            if (savePreference) {
                this.systemThemeEnabled = false;
                this.saveThemePreference();
            }
            
            // Update UI
            this.updateThemeSwitcherUI();
            
            // Show notification
            this.showThemeNotification(themeName);
        }
        
        toggleTheme() {
            const themeNames = Object.keys(this.themes);
            const currentIndex = themeNames.indexOf(this.currentTheme);
            const nextIndex = (currentIndex + 1) % themeNames.length;
            this.setTheme(themeNames[nextIndex]);
        }
        
        enableSystemTheme() {
            this.systemThemeEnabled = true;
            this.setTheme(this.getSystemTheme(), false);
            localStorage.setItem('prometheus-system-theme', 'true');
        }
        
        disableSystemTheme() {
            this.systemThemeEnabled = false;
            localStorage.setItem('prometheus-system-theme', 'false');
        }
        
        saveThemePreference() {
            localStorage.setItem('prometheus-theme', this.currentTheme);
            localStorage.setItem('prometheus-system-theme', this.systemThemeEnabled.toString());
        }
        
        setupThemePersistence() {
            // Save theme on page unload
            window.addEventListener('beforeunload', () => {
                this.saveThemePreference();
            });
        }
        
        createThemeSwitcherUI() {
            // Create floating theme switcher
            const switcher = document.createElement('div');
            switcher.id = 'theme-switcher';
            switcher.style.cssText = `
                position: fixed;
                bottom: 20px;
                left: 20px;
                z-index: 9999;
                background: var(--secondary);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 10px;
                display: flex;
                flex-direction: column;
                gap: 8px;
                min-width: 150px;
                box-shadow: 0 4px 12px var(--shadow);
                transition: transform 0.3s;
                transform: translateX(-90%);
            `;
            
            switcher.innerHTML = `
                <div style="display: flex; align-items: center; justify-content: space-between; cursor: pointer;" id="theme-switcher-toggle">
                    <span style="font-size: 12px; font-weight: bold;">Theme</span>
                    <span style="font-size: 20px;" id="current-theme-icon">${this.themes[this.currentTheme].icon}</span>
                </div>
                <div id="theme-options" style="display: none;">
                    <div style="margin-bottom: 8px;">
                        <label style="display: flex; align-items: center; gap: 5px; cursor: pointer;">
                            <input type="checkbox" id="system-theme-checkbox" ${this.systemThemeEnabled ? 'checked' : ''}>
                            <span style="font-size: 11px;">Use system theme</span>
                        </label>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px;">
                        ${Object.entries(this.themes).map(([key, theme]) => `
                            <button class="theme-option" data-theme="${key}" style="
                                background: transparent;
                                border: 1px solid var(--border);
                                padding: 8px;
                                cursor: pointer;
                                border-radius: 6px;
                                font-size: 20px;
                                transition: all 0.2s;
                            " title="${theme.name}">
                                ${theme.icon}
                            </button>
                        `).join('')}
                    </div>
                </div>
            `;
            
            document.body.appendChild(switcher);
            
            // Setup interactions
            const toggle = document.getElementById('theme-switcher-toggle');
            const options = document.getElementById('theme-options');
            
            // Hover to expand
            switcher.addEventListener('mouseenter', () => {
                switcher.style.transform = 'translateX(0)';
                options.style.display = 'block';
            });
            
            switcher.addEventListener('mouseleave', () => {
                switcher.style.transform = 'translateX(-90%)';
                options.style.display = 'none';
            });
            
            // Theme selection
            switcher.querySelectorAll('.theme-option').forEach(btn => {
                btn.addEventListener('click', () => {
                    this.setTheme(btn.dataset.theme);
                });
            });
            
            // System theme checkbox
            document.getElementById('system-theme-checkbox')?.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.enableSystemTheme();
                } else {
                    this.disableSystemTheme();
                }
            });
            
            this.updateThemeSwitcherUI();
        }
        
        updateThemeSwitcherUI() {
            const iconEl = document.getElementById('current-theme-icon');
            if (iconEl && this.themes[this.currentTheme]) {
                iconEl.textContent = this.themes[this.currentTheme].icon;
            }
            
            // Highlight current theme
            document.querySelectorAll('.theme-option').forEach(btn => {
                const isActive = btn.dataset.theme === this.currentTheme;
                btn.style.background = isActive ? 'var(--accent)' : 'transparent';
                btn.style.color = isActive ? 'white' : 'var(--text-primary)';
            });
        }
        
        showThemeNotification(themeName) {
            const theme = this.themes[themeName];
            if (!theme) return;
            
            // Create notification
            const notification = document.createElement('div');
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background: var(--secondary);
                border: 1px solid var(--accent);
                border-radius: 8px;
                padding: 12px 20px;
                display: flex;
                align-items: center;
                gap: 10px;
                z-index: 10000;
                animation: slideIn 0.3s;
                box-shadow: 0 4px 12px var(--shadow);
            `;
            
            notification.innerHTML = `
                <span style="font-size: 24px;">${theme.icon}</span>
                <span>Theme changed to <strong>${theme.name}</strong></span>
            `;
            
            document.body.appendChild(notification);
            
            // Remove after delay
            setTimeout(() => {
                notification.style.animation = 'slideOut 0.3s';
                setTimeout(() => notification.remove(), 300);
            }, 2000);
        }
        
        // Custom theme management
        addCustomTheme(name, themeConfig) {
            if (this.themes[name]) {
                console.warn(`Theme "${name}" already exists`);
                return false;
            }
            
            this.themes[name] = themeConfig;
            this.customThemes.set(name, themeConfig);
            this.updateThemeSwitcherUI();
            return true;
        }
        
        removeCustomTheme(name) {
            if (!this.customThemes.has(name)) {
                console.warn(`Custom theme "${name}" not found`);
                return false;
            }
            
            delete this.themes[name];
            this.customThemes.delete(name);
            
            if (this.currentTheme === name) {
                this.setTheme('dark');
            }
            
            this.updateThemeSwitcherUI();
            return true;
        }
        
        exportTheme(themeName) {
            const theme = this.themes[themeName];
            if (!theme) return null;
            
            return JSON.stringify(theme, null, 2);
        }
        
        importTheme(name, themeJson) {
            try {
                const theme = JSON.parse(themeJson);
                return this.addCustomTheme(name, theme);
            } catch (error) {
                console.error('Failed to import theme:', error);
                return false;
            }
        }
        
        // Observer pattern for theme changes
        observe(callback) {
            this.observers.add(callback);
        }
        
        unobserve(callback) {
            this.observers.delete(callback);
        }
        
        notifyObservers(themeName, theme) {
            this.observers.forEach(callback => {
                try {
                    callback(themeName, theme);
                } catch (error) {
                    console.error('Error in theme observer:', error);
                }
            });
        }
        
        // Get current theme info
        getCurrentTheme() {
            return {
                name: this.currentTheme,
                ...this.themes[this.currentTheme]
            };
        }
        
        getAvailableThemes() {
            return Object.keys(this.themes);
        }
        
        getThemeColor(colorName) {
            const theme = this.themes[this.currentTheme];
            return theme?.colors[colorName] || null;
        }
    }
    
    // CSS animation styles
    const animationStyles = document.createElement('style');
    animationStyles.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    `;
    document.head.appendChild(animationStyles);
    
    // Initialize and expose globally
    const themeManager = new ThemeManager();
    
    // Public API
    global.ThemeManager = {
        instance: themeManager,
        setTheme: (name) => themeManager.setTheme(name),
        toggleTheme: () => themeManager.toggleTheme(),
        getCurrentTheme: () => themeManager.getCurrentTheme(),
        getAvailableThemes: () => themeManager.getAvailableThemes(),
        getThemeColor: (color) => themeManager.getThemeColor(color),
        addCustomTheme: (name, config) => themeManager.addCustomTheme(name, config),
        removeCustomTheme: (name) => themeManager.removeCustomTheme(name),
        exportTheme: (name) => themeManager.exportTheme(name),
        importTheme: (name, json) => themeManager.importTheme(name, json),
        enableSystemTheme: () => themeManager.enableSystemTheme(),
        disableSystemTheme: () => themeManager.disableSystemTheme(),
        observe: (callback) => themeManager.observe(callback),
        unobserve: (callback) => themeManager.unobserve(callback)
    };
    
    console.log('✨ Theme Manager ready. Access via window.ThemeManager');
    
})(window);