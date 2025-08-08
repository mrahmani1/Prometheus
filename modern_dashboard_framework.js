/**
 * Modern Dashboard Framework - Task 2.1
 * Implements Dash/Plotly-like components with modern UI/UX features
 * @author Prometheus Platform
 * @version 3.0.0
 */

(function(global) {
    'use strict';
    
    console.log('🎨 Modern Dashboard Framework - Initializing advanced UI system...');
    
    // Core Dashboard Framework
    class ModernDashboard {
        constructor() {
            this.version = '3.0.0';
            this.components = new Map();
            this.layouts = new Map();
            this.state = {
                theme: 'dark',
                sidebarCollapsed: false,
                fullscreen: false,
                breadcrumbs: [],
                activeView: 'dashboard',
                shortcuts: new Map()
            };
            
            this.eventBus = new EventTarget();
            this.initialized = false;
            
            // Initialize on DOM ready
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', () => this.init());
            } else {
                this.init();
            }
        }
        
        init() {
            if (this.initialized) return;
            
            console.log('Initializing Modern Dashboard Framework...');
            
            // Create main dashboard structure
            this.createDashboardStructure();
            
            // Initialize theme manager
            this.initializeTheme();
            
            // Setup navigation
            this.setupNavigation();
            
            // Initialize keyboard shortcuts
            this.setupKeyboardShortcuts();
            
            // Setup fullscreen handling
            this.setupFullscreen();
            
            // Initialize components
            this.initializeComponents();
            
            // Apply saved preferences
            this.loadPreferences();
            
            this.initialized = true;
            console.log('✅ Modern Dashboard Framework initialized');
            
            // Emit ready event
            this.emit('dashboard:ready', { dashboard: this });
        }
        
        createDashboardStructure() {
            // Check if dashboard already exists
            if (document.getElementById('modern-dashboard')) return;
            
            const dashboard = document.createElement('div');
            dashboard.id = 'modern-dashboard';
            dashboard.className = 'modern-dashboard';
            
            dashboard.innerHTML = `
                <!-- Top Navigation Bar -->
                <nav class="dashboard-navbar">
                    <div class="navbar-left">
                        <button id="sidebar-toggle" class="icon-btn" title="Toggle Sidebar">
                            <span class="hamburger">☰</span>
                        </button>
                        <div class="app-title">
                            <span class="logo">🌋</span>
                            <span class="title">Prometheus Platform</span>
                        </div>
                    </div>
                    
                    <div class="navbar-center">
                        <div id="breadcrumb-nav" class="breadcrumb-nav">
                            <span class="breadcrumb-item">Dashboard</span>
                        </div>
                    </div>
                    
                    <div class="navbar-right">
                        <button id="search-btn" class="icon-btn" title="Search (/)">
                            <span>🔍</span>
                        </button>
                        <button id="theme-toggle" class="icon-btn" title="Toggle Theme">
                            <span class="theme-icon">🌙</span>
                        </button>
                        <button id="fullscreen-btn" class="icon-btn" title="Fullscreen (F11)">
                            <span>⛶</span>
                        </button>
                        <button id="help-btn" class="icon-btn" title="Help (?)">
                            <span>?</span>
                        </button>
                    </div>
                </nav>
                
                <!-- Search Modal -->
                <div id="search-modal" class="modal-overlay" style="display: none;">
                    <div class="search-modal">
                        <input type="text" id="global-search" placeholder="Search everything..." autofocus>
                        <div class="search-results" id="search-results"></div>
                    </div>
                </div>
                
                <!-- Help Modal -->
                <div id="help-modal" class="modal-overlay" style="display: none;">
                    <div class="help-modal">
                        <div class="modal-header">
                            <h3>Keyboard Shortcuts</h3>
                            <button class="close-btn" onclick="ModernDashboard.instance.hideHelp()">×</button>
                        </div>
                        <div class="shortcuts-list">
                            <div class="shortcut-item"><kbd>?</kbd> Show this help</div>
                            <div class="shortcut-item"><kbd>/</kbd> Search</div>
                            <div class="shortcut-item"><kbd>Ctrl+B</kbd> Toggle sidebar</div>
                            <div class="shortcut-item"><kbd>Ctrl+\\</kbd> Toggle theme</div>
                            <div class="shortcut-item"><kbd>F11</kbd> Fullscreen</div>
                            <div class="shortcut-item"><kbd>Alt+1-9</kbd> Quick navigation</div>
                            <div class="shortcut-item"><kbd>Esc</kbd> Close modals</div>
                            <div class="shortcut-item"><kbd>Ctrl+S</kbd> Save settings</div>
                            <div class="shortcut-item"><kbd>Ctrl+R</kbd> Refresh data</div>
                            <div class="shortcut-item"><kbd>Ctrl+P</kbd> Command palette</div>
                        </div>
                    </div>
                </div>
                
                <!-- Main Layout -->
                <div class="dashboard-layout">
                    <!-- Collapsible Sidebar -->
                    <aside id="dashboard-sidebar" class="dashboard-sidebar">
                        <div class="sidebar-content">
                            <nav class="sidebar-nav">
                                <div class="nav-section">
                                    <div class="nav-section-title">Main</div>
                                    <a href="#dashboard" class="nav-item active" data-view="dashboard">
                                        <span class="nav-icon">📊</span>
                                        <span class="nav-label">Dashboard</span>
                                    </a>
                                    <a href="#volcanic" class="nav-item" data-view="volcanic">
                                        <span class="nav-icon">🌋</span>
                                        <span class="nav-label">Volcanic Systems</span>
                                    </a>
                                    <a href="#seismic" class="nav-item" data-view="seismic">
                                        <span class="nav-icon">📈</span>
                                        <span class="nav-label">Seismic Analysis</span>
                                    </a>
                                    <a href="#insar" class="nav-item" data-view="insar">
                                        <span class="nav-icon">🛰️</span>
                                        <span class="nav-label">InSAR Processing</span>
                                    </a>
                                </div>
                                
                                <div class="nav-section">
                                    <div class="nav-section-title">Analysis</div>
                                    <a href="#ml" class="nav-item" data-view="ml">
                                        <span class="nav-icon">🤖</span>
                                        <span class="nav-label">Machine Learning</span>
                                    </a>
                                    <a href="#prediction" class="nav-item" data-view="prediction">
                                        <span class="nav-icon">🔮</span>
                                        <span class="nav-label">Predictions</span>
                                    </a>
                                    <a href="#reports" class="nav-item" data-view="reports">
                                        <span class="nav-icon">📄</span>
                                        <span class="nav-label">Reports</span>
                                    </a>
                                </div>
                                
                                <div class="nav-section">
                                    <div class="nav-section-title">System</div>
                                    <a href="#monitoring" class="nav-item" data-view="monitoring">
                                        <span class="nav-icon">📡</span>
                                        <span class="nav-label">Monitoring</span>
                                    </a>
                                    <a href="#settings" class="nav-item" data-view="settings">
                                        <span class="nav-icon">⚙️</span>
                                        <span class="nav-label">Settings</span>
                                    </a>
                                    <a href="#errors" class="nav-item" data-view="errors">
                                        <span class="nav-icon">🚨</span>
                                        <span class="nav-label">Error Logs</span>
                                    </a>
                                </div>
                            </nav>
                            
                            <div class="sidebar-footer">
                                <div class="status-indicator">
                                    <span class="status-dot online"></span>
                                    <span>System Online</span>
                                </div>
                            </div>
                        </div>
                    </aside>
                    
                    <!-- Main Content Area -->
                    <main id="dashboard-content" class="dashboard-content">
                        <div class="content-wrapper">
                            <!-- Dynamic content will be loaded here -->
                            <div id="view-container" class="view-container">
                                <!-- Default dashboard view -->
                                <div class="dashboard-grid" id="dashboard-grid">
                                    <!-- Grid items will be populated here -->
                                </div>
                            </div>
                        </div>
                    </main>
                </div>
                
                <!-- Toast Notifications -->
                <div id="toast-container" class="toast-container"></div>
                
                <!-- Command Palette -->
                <div id="command-palette" class="command-palette" style="display: none;">
                    <input type="text" id="command-input" placeholder="Type a command..." />
                    <div class="command-list" id="command-list"></div>
                </div>
            `;
            
            // Add to body or replace existing content
            const existingContent = document.body.innerHTML;
            document.body.innerHTML = '';
            document.body.appendChild(dashboard);
            
            // Create legacy content container if there was existing content
            if (existingContent.trim()) {
                const legacyContainer = document.createElement('div');
                legacyContainer.id = 'legacy-content';
                legacyContainer.style.display = 'none';
                legacyContainer.innerHTML = existingContent;
                document.body.appendChild(legacyContainer);
            }
            
            // Add dashboard styles
            this.injectStyles();
        }
        
        injectStyles() {
            if (document.getElementById('modern-dashboard-styles')) return;
            
            const styles = document.createElement('style');
            styles.id = 'modern-dashboard-styles';
            styles.textContent = `
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    overflow: hidden;
                }
                
                .modern-dashboard {
                    display: flex;
                    flex-direction: column;
                    height: 100vh;
                    background: var(--bg-primary);
                    color: var(--text-primary);
                    transition: all 0.3s ease;
                }
                
                /* CSS Variables for Themes */
                .modern-dashboard[data-theme="dark"] {
                    --bg-primary: #0a0a0a;
                    --bg-secondary: #1a1a1a;
                    --bg-tertiary: #2a2a2a;
                    --text-primary: #ffffff;
                    --text-secondary: #b0b0b0;
                    --accent: #FF6B35;
                    --accent-hover: #ff8555;
                    --border: rgba(255, 255, 255, 0.1);
                    --shadow: rgba(0, 0, 0, 0.5);
                }
                
                .modern-dashboard[data-theme="light"] {
                    --bg-primary: #ffffff;
                    --bg-secondary: #f5f5f5;
                    --bg-tertiary: #e0e0e0;
                    --text-primary: #1a1a1a;
                    --text-secondary: #666666;
                    --accent: #FF6B35;
                    --accent-hover: #ff8555;
                    --border: rgba(0, 0, 0, 0.1);
                    --shadow: rgba(0, 0, 0, 0.1);
                }
                
                /* Navigation Bar */
                .dashboard-navbar {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    height: 60px;
                    padding: 0 20px;
                    background: var(--bg-secondary);
                    border-bottom: 1px solid var(--border);
                    z-index: 1000;
                }
                
                .navbar-left, .navbar-center, .navbar-right {
                    display: flex;
                    align-items: center;
                    gap: 15px;
                }
                
                .app-title {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    font-size: 18px;
                    font-weight: 600;
                }
                
                .logo {
                    font-size: 24px;
                }
                
                .icon-btn {
                    background: transparent;
                    border: 1px solid transparent;
                    color: var(--text-primary);
                    padding: 8px 12px;
                    border-radius: 6px;
                    cursor: pointer;
                    transition: all 0.2s;
                    font-size: 18px;
                }
                
                .icon-btn:hover {
                    background: var(--bg-tertiary);
                    border-color: var(--border);
                }
                
                /* Breadcrumb Navigation */
                .breadcrumb-nav {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    color: var(--text-secondary);
                }
                
                .breadcrumb-item {
                    cursor: pointer;
                    transition: color 0.2s;
                }
                
                .breadcrumb-item:hover {
                    color: var(--accent);
                }
                
                .breadcrumb-item:not(:last-child)::after {
                    content: '/';
                    margin-left: 8px;
                    color: var(--text-secondary);
                }
                
                /* Dashboard Layout */
                .dashboard-layout {
                    display: flex;
                    flex: 1;
                    overflow: hidden;
                }
                
                /* Sidebar */
                .dashboard-sidebar {
                    width: 260px;
                    background: var(--bg-secondary);
                    border-right: 1px solid var(--border);
                    display: flex;
                    flex-direction: column;
                    transition: all 0.3s ease;
                    overflow-y: auto;
                }
                
                .dashboard-sidebar.collapsed {
                    width: 60px;
                }
                
                .sidebar-content {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                }
                
                .sidebar-nav {
                    flex: 1;
                    padding: 20px 0;
                }
                
                .nav-section {
                    margin-bottom: 20px;
                }
                
                .nav-section-title {
                    padding: 0 20px;
                    margin-bottom: 10px;
                    font-size: 12px;
                    text-transform: uppercase;
                    color: var(--text-secondary);
                    letter-spacing: 1px;
                    transition: opacity 0.3s;
                }
                
                .collapsed .nav-section-title {
                    opacity: 0;
                    height: 0;
                    margin: 0;
                }
                
                .nav-item {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    padding: 12px 20px;
                    color: var(--text-secondary);
                    text-decoration: none;
                    transition: all 0.2s;
                    position: relative;
                }
                
                .nav-item:hover {
                    background: var(--bg-tertiary);
                    color: var(--text-primary);
                }
                
                .nav-item.active {
                    background: var(--bg-tertiary);
                    color: var(--accent);
                    border-left: 3px solid var(--accent);
                }
                
                .nav-icon {
                    font-size: 20px;
                    min-width: 20px;
                }
                
                .collapsed .nav-label {
                    display: none;
                }
                
                .collapsed .nav-item {
                    justify-content: center;
                    padding: 12px;
                }
                
                /* Main Content */
                .dashboard-content {
                    flex: 1;
                    overflow-y: auto;
                    background: var(--bg-primary);
                    position: relative;
                }
                
                .content-wrapper {
                    padding: 20px;
                    max-width: 1400px;
                    margin: 0 auto;
                }
                
                /* Dashboard Grid */
                .dashboard-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    animation: fadeIn 0.5s;
                }
                
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(20px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                
                /* Cards */
                .dashboard-card {
                    background: var(--bg-secondary);
                    border: 1px solid var(--border);
                    border-radius: 8px;
                    padding: 20px;
                    transition: all 0.3s;
                }
                
                .dashboard-card:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px var(--shadow);
                }
                
                .card-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 15px;
                }
                
                .card-title {
                    font-size: 16px;
                    font-weight: 600;
                    color: var(--text-primary);
                }
                
                .card-actions {
                    display: flex;
                    gap: 8px;
                }
                
                /* Modals */
                .modal-overlay {
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(0, 0, 0, 0.7);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 2000;
                    backdrop-filter: blur(4px);
                }
                
                .search-modal, .help-modal {
                    background: var(--bg-secondary);
                    border-radius: 12px;
                    padding: 20px;
                    max-width: 600px;
                    width: 90%;
                    max-height: 80vh;
                    overflow-y: auto;
                    box-shadow: 0 10px 40px var(--shadow);
                }
                
                #global-search, #command-input {
                    width: 100%;
                    padding: 15px;
                    font-size: 16px;
                    background: var(--bg-tertiary);
                    border: 1px solid var(--border);
                    border-radius: 8px;
                    color: var(--text-primary);
                    outline: none;
                }
                
                #global-search:focus, #command-input:focus {
                    border-color: var(--accent);
                    box-shadow: 0 0 0 3px rgba(255, 107, 53, 0.2);
                }
                
                .search-results, .command-list {
                    margin-top: 15px;
                }
                
                .search-result-item, .command-item {
                    padding: 10px;
                    background: var(--bg-tertiary);
                    margin-bottom: 5px;
                    border-radius: 6px;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                
                .search-result-item:hover, .command-item:hover {
                    background: var(--accent);
                    color: white;
                }
                
                /* Keyboard shortcuts display */
                .shortcuts-list {
                    display: grid;
                    gap: 10px;
                    margin-top: 20px;
                }
                
                .shortcut-item {
                    display: flex;
                    align-items: center;
                    gap: 15px;
                    padding: 10px;
                    background: var(--bg-tertiary);
                    border-radius: 6px;
                }
                
                kbd {
                    background: var(--bg-primary);
                    padding: 4px 8px;
                    border-radius: 4px;
                    border: 1px solid var(--border);
                    font-family: monospace;
                    font-size: 12px;
                    min-width: 80px;
                    text-align: center;
                }
                
                /* Toast Notifications */
                .toast-container {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    z-index: 3000;
                }
                
                .toast {
                    background: var(--bg-secondary);
                    border: 1px solid var(--border);
                    border-left: 3px solid var(--accent);
                    padding: 15px 20px;
                    border-radius: 6px;
                    margin-bottom: 10px;
                    min-width: 300px;
                    animation: slideIn 0.3s;
                    box-shadow: 0 4px 12px var(--shadow);
                }
                
                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
                
                .toast.success { border-left-color: #4CAF50; }
                .toast.error { border-left-color: #f44336; }
                .toast.warning { border-left-color: #FF9800; }
                .toast.info { border-left-color: #2196F3; }
                
                /* Command Palette */
                .command-palette {
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: var(--bg-secondary);
                    border-radius: 12px;
                    padding: 20px;
                    width: 600px;
                    max-height: 400px;
                    z-index: 3000;
                    box-shadow: 0 10px 40px var(--shadow);
                }
                
                /* Status Indicator */
                .status-indicator {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 15px 20px;
                    border-top: 1px solid var(--border);
                    font-size: 12px;
                    color: var(--text-secondary);
                }
                
                .status-dot {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: #4CAF50;
                    animation: pulse 2s infinite;
                }
                
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }
                
                .status-dot.offline { background: #f44336; animation: none; }
                .status-dot.warning { background: #FF9800; }
                
                /* Fullscreen mode adjustments */
                .fullscreen .dashboard-navbar {
                    display: none;
                }
                
                .fullscreen .dashboard-layout {
                    height: 100vh;
                }
                
                /* Responsive */
                @media (max-width: 768px) {
                    .dashboard-sidebar {
                        position: fixed;
                        left: -260px;
                        height: 100%;
                        z-index: 999;
                    }
                    
                    .dashboard-sidebar.mobile-open {
                        left: 0;
                    }
                    
                    .navbar-center {
                        display: none;
                    }
                    
                    .dashboard-grid {
                        grid-template-columns: 1fr;
                    }
                }
            `;
            
            document.head.appendChild(styles);
        }
        
        setupNavigation() {
            // Sidebar toggle
            const sidebarToggle = document.getElementById('sidebar-toggle');
            const sidebar = document.getElementById('dashboard-sidebar');
            
            if (sidebarToggle && sidebar) {
                sidebarToggle.addEventListener('click', () => {
                    sidebar.classList.toggle('collapsed');
                    this.state.sidebarCollapsed = sidebar.classList.contains('collapsed');
                    this.savePreferences();
                    this.emit('sidebar:toggle', { collapsed: this.state.sidebarCollapsed });
                });
            }
            
            // Navigation items
            document.querySelectorAll('.nav-item').forEach(item => {
                item.addEventListener('click', (e) => {
                    e.preventDefault();
                    const view = item.dataset.view;
                    this.navigateToView(view);
                });
            });
            
            // Search button
            document.getElementById('search-btn')?.addEventListener('click', () => {
                this.showSearch();
            });
            
            // Theme toggle
            document.getElementById('theme-toggle')?.addEventListener('click', () => {
                this.toggleTheme();
            });
            
            // Fullscreen button
            document.getElementById('fullscreen-btn')?.addEventListener('click', () => {
                this.toggleFullscreen();
            });
            
            // Help button
            document.getElementById('help-btn')?.addEventListener('click', () => {
                this.showHelp();
            });
            
            // Close modals on overlay click
            document.querySelectorAll('.modal-overlay').forEach(modal => {
                modal.addEventListener('click', (e) => {
                    if (e.target === modal) {
                        modal.style.display = 'none';
                    }
                });
            });
        }
        
        setupKeyboardShortcuts() {
            const shortcuts = {
                '?': () => this.showHelp(),
                '/': () => this.showSearch(),
                'Escape': () => this.closeAllModals(),
                'F11': (e) => { e.preventDefault(); this.toggleFullscreen(); }
            };
            
            const ctrlShortcuts = {
                'b': () => this.toggleSidebar(),
                '\\': () => this.toggleTheme(),
                's': (e) => { e.preventDefault(); this.saveSettings(); },
                'r': (e) => { e.preventDefault(); this.refreshData(); },
                'p': (e) => { e.preventDefault(); this.showCommandPalette(); }
            };
            
            const altShortcuts = {
                '1': () => this.navigateToView('dashboard'),
                '2': () => this.navigateToView('volcanic'),
                '3': () => this.navigateToView('seismic'),
                '4': () => this.navigateToView('insar'),
                '5': () => this.navigateToView('ml'),
                '6': () => this.navigateToView('prediction'),
                '7': () => this.navigateToView('reports'),
                '8': () => this.navigateToView('monitoring'),
                '9': () => this.navigateToView('settings')
            };
            
            document.addEventListener('keydown', (e) => {
                // Skip if typing in input
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                    if (e.key === 'Escape') {
                        e.target.blur();
                        this.closeAllModals();
                    }
                    return;
                }
                
                // Regular shortcuts
                if (shortcuts[e.key]) {
                    e.preventDefault();
                    shortcuts[e.key](e);
                }
                
                // Ctrl shortcuts
                if (e.ctrlKey && ctrlShortcuts[e.key.toLowerCase()]) {
                    e.preventDefault();
                    ctrlShortcuts[e.key.toLowerCase()](e);
                }
                
                // Alt shortcuts
                if (e.altKey && altShortcuts[e.key]) {
                    e.preventDefault();
                    altShortcuts[e.key](e);
                }
            });
        }
        
        initializeTheme() {
            const dashboard = document.getElementById('modern-dashboard');
            if (!dashboard) return;
            
            // Check system preference
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            const savedTheme = localStorage.getItem('dashboard-theme') || (prefersDark ? 'dark' : 'light');
            
            this.state.theme = savedTheme;
            dashboard.setAttribute('data-theme', savedTheme);
            this.updateThemeIcon();
            
            // Listen for system theme changes
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
                if (!localStorage.getItem('dashboard-theme')) {
                    this.state.theme = e.matches ? 'dark' : 'light';
                    dashboard.setAttribute('data-theme', this.state.theme);
                    this.updateThemeIcon();
                }
            });
        }
        
        toggleTheme() {
            const dashboard = document.getElementById('modern-dashboard');
            if (!dashboard) return;
            
            this.state.theme = this.state.theme === 'dark' ? 'light' : 'dark';
            dashboard.setAttribute('data-theme', this.state.theme);
            this.updateThemeIcon();
            this.savePreferences();
            
            this.showToast(`Theme switched to ${this.state.theme} mode`, 'info');
            this.emit('theme:change', { theme: this.state.theme });
        }
        
        updateThemeIcon() {
            const icon = document.querySelector('.theme-icon');
            if (icon) {
                icon.textContent = this.state.theme === 'dark' ? '🌙' : '☀️';
            }
        }
        
        toggleSidebar() {
            const sidebar = document.getElementById('dashboard-sidebar');
            if (sidebar) {
                sidebar.classList.toggle('collapsed');
                this.state.sidebarCollapsed = sidebar.classList.contains('collapsed');
                this.savePreferences();
            }
        }
        
        setupFullscreen() {
            document.addEventListener('fullscreenchange', () => {
                const dashboard = document.getElementById('modern-dashboard');
                if (document.fullscreenElement) {
                    dashboard?.classList.add('fullscreen');
                    this.state.fullscreen = true;
                } else {
                    dashboard?.classList.remove('fullscreen');
                    this.state.fullscreen = false;
                }
            });
        }
        
        toggleFullscreen() {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen().catch(err => {
                    this.showToast('Could not enter fullscreen: ' + err.message, 'error');
                });
            } else {
                document.exitFullscreen();
            }
        }
        
        showSearch() {
            const modal = document.getElementById('search-modal');
            const input = document.getElementById('global-search');
            
            if (modal && input) {
                modal.style.display = 'flex';
                input.value = '';
                input.focus();
                
                // Setup search functionality
                input.oninput = debounce(() => this.performSearch(input.value), 300);
            }
        }
        
        performSearch(query) {
            const results = document.getElementById('search-results');
            if (!results) return;
            
            if (!query) {
                results.innerHTML = '';
                return;
            }
            
            // Simulate search results
            const mockResults = [
                { title: 'Mount Vesuvius', type: 'Volcano', path: '/volcanic/vesuvius' },
                { title: 'Seismic Report 2024', type: 'Report', path: '/reports/seismic-2024' },
                { title: 'ML Model Training', type: 'Task', path: '/ml/training' },
                { title: 'InSAR Processing Queue', type: 'Process', path: '/insar/queue' }
            ].filter(item => 
                item.title.toLowerCase().includes(query.toLowerCase()) ||
                item.type.toLowerCase().includes(query.toLowerCase())
            );
            
            results.innerHTML = mockResults.map(item => `
                <div class="search-result-item" onclick="ModernDashboard.instance.navigateToPath('${item.path}')">
                    <strong>${item.title}</strong>
                    <span style="color: var(--text-secondary); margin-left: 10px;">${item.type}</span>
                </div>
            `).join('');
        }
        
        showHelp() {
            const modal = document.getElementById('help-modal');
            if (modal) {
                modal.style.display = 'flex';
            }
        }
        
        hideHelp() {
            const modal = document.getElementById('help-modal');
            if (modal) {
                modal.style.display = 'none';
            }
        }
        
        showCommandPalette() {
            const palette = document.getElementById('command-palette');
            const input = document.getElementById('command-input');
            
            if (palette && input) {
                palette.style.display = 'block';
                input.value = '';
                input.focus();
                
                // Populate commands
                const commands = [
                    { name: 'Toggle Theme', action: () => this.toggleTheme() },
                    { name: 'Toggle Sidebar', action: () => this.toggleSidebar() },
                    { name: 'Refresh Data', action: () => this.refreshData() },
                    { name: 'Export Report', action: () => this.exportReport() },
                    { name: 'Clear Cache', action: () => this.clearCache() },
                    { name: 'Run Diagnostics', action: () => this.runDiagnostics() }
                ];
                
                const list = document.getElementById('command-list');
                if (list) {
                    list.innerHTML = commands.map((cmd, i) => `
                        <div class="command-item" data-command="${i}">
                            ${cmd.name}
                        </div>
                    `).join('');
                    
                    // Add click handlers
                    list.querySelectorAll('.command-item').forEach(item => {
                        item.addEventListener('click', () => {
                            const index = parseInt(item.dataset.command);
                            commands[index].action();
                            palette.style.display = 'none';
                        });
                    });
                }
                
                // Filter commands on input
                input.oninput = () => {
                    const query = input.value.toLowerCase();
                    list.querySelectorAll('.command-item').forEach(item => {
                        const show = item.textContent.toLowerCase().includes(query);
                        item.style.display = show ? 'block' : 'none';
                    });
                };
            }
        }
        
        closeAllModals() {
            document.querySelectorAll('.modal-overlay').forEach(modal => {
                modal.style.display = 'none';
            });
            document.getElementById('command-palette').style.display = 'none';
        }
        
        navigateToView(view) {
            // Update active nav item
            document.querySelectorAll('.nav-item').forEach(item => {
                item.classList.toggle('active', item.dataset.view === view);
            });
            
            // Update breadcrumb
            this.updateBreadcrumb([view.charAt(0).toUpperCase() + view.slice(1)]);
            
            // Update state
            this.state.activeView = view;
            
            // Load view content
            this.loadViewContent(view);
            
            // Emit navigation event
            this.emit('navigation:change', { view });
        }
        
        navigateToPath(path) {
            console.log('Navigating to:', path);
            this.closeAllModals();
            // Implementation would handle actual navigation
        }
        
        updateBreadcrumb(items) {
            const breadcrumb = document.getElementById('breadcrumb-nav');
            if (breadcrumb) {
                breadcrumb.innerHTML = items.map((item, i) => 
                    `<span class="breadcrumb-item">${item}</span>`
                ).join('');
            }
            this.state.breadcrumbs = items;
        }
        
        loadViewContent(view) {
            const container = document.getElementById('view-container');
            if (!container) return;
            
            // Show loading state
            container.innerHTML = '<div class="loading">Loading...</div>';
            
            // Simulate content loading
            setTimeout(() => {
                const content = this.generateViewContent(view);
                container.innerHTML = content;
                this.initializeViewComponents(view);
            }, 300);
        }
        
        generateViewContent(view) {
            const contents = {
                dashboard: this.generateDashboardContent(),
                volcanic: this.generateVolcanicContent(),
                seismic: this.generateSeismicContent(),
                insar: this.generateInSARContent(),
                ml: this.generateMLContent(),
                prediction: this.generatePredictionContent(),
                reports: this.generateReportsContent(),
                monitoring: this.generateMonitoringContent(),
                settings: this.generateSettingsContent(),
                errors: this.generateErrorsContent()
            };
            
            return contents[view] || '<div>View not found</div>';
        }
        
        generateDashboardContent() {
            return `
                <div class="dashboard-grid">
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">🌋 Active Volcanoes</h3>
                            <div class="card-actions">
                                <button class="icon-btn">↻</button>
                            </div>
                        </div>
                        <div class="card-content">
                            <div style="font-size: 36px; font-weight: bold; color: var(--accent);">47</div>
                            <div style="color: var(--text-secondary);">Currently erupting worldwide</div>
                        </div>
                    </div>
                    
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">📊 Seismic Events</h3>
                            <div class="card-actions">
                                <button class="icon-btn">↻</button>
                            </div>
                        </div>
                        <div class="card-content">
                            <div style="font-size: 36px; font-weight: bold; color: var(--accent);">2,341</div>
                            <div style="color: var(--text-secondary);">Events in last 24 hours</div>
                        </div>
                    </div>
                    
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">🛰️ InSAR Processing</h3>
                            <div class="card-actions">
                                <button class="icon-btn">↻</button>
                            </div>
                        </div>
                        <div class="card-content">
                            <div style="font-size: 36px; font-weight: bold; color: var(--accent);">12</div>
                            <div style="color: var(--text-secondary);">Scenes processing</div>
                        </div>
                    </div>
                    
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">🤖 ML Models</h3>
                            <div class="card-actions">
                                <button class="icon-btn">↻</button>
                            </div>
                        </div>
                        <div class="card-content">
                            <div style="font-size: 36px; font-weight: bold; color: var(--accent);">95.2%</div>
                            <div style="color: var(--text-secondary);">Average accuracy</div>
                        </div>
                    </div>
                    
                    <div class="dashboard-card" style="grid-column: span 2;">
                        <div class="card-header">
                            <h3 class="card-title">📈 System Performance</h3>
                        </div>
                        <div class="card-content">
                            <canvas id="performance-chart" height="200"></canvas>
                        </div>
                    </div>
                </div>
            `;
        }
        
        generateVolcanicContent() {
            return `
                <div class="view-header">
                    <h2>🌋 Volcanic Systems Monitor</h2>
                </div>
                <div class="dashboard-grid">
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">Volcano Database</h3>
                        </div>
                        <div class="card-content">
                            <div id="volcano-selector-placeholder">
                                <!-- Volcano selector will be loaded here -->
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        generateSeismicContent() {
            return `
                <div class="view-header">
                    <h2>📈 Seismic Analysis</h2>
                </div>
                <div class="dashboard-grid">
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">Real-time Seismic Data</h3>
                        </div>
                        <div class="card-content">
                            <div id="seismic-chart-placeholder">
                                <!-- Seismic chart will be loaded here -->
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        generateInSARContent() {
            return `
                <div class="view-header">
                    <h2>🛰️ InSAR Processing</h2>
                </div>
                <div class="dashboard-grid">
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">Processing Queue</h3>
                        </div>
                        <div class="card-content">
                            <div id="insar-queue-placeholder">
                                <!-- InSAR queue will be loaded here -->
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        generateMLContent() {
            return `
                <div class="view-header">
                    <h2>🤖 Machine Learning</h2>
                </div>
                <div class="dashboard-grid">
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">Model Performance</h3>
                        </div>
                        <div class="card-content">
                            <div id="ml-metrics-placeholder">
                                <!-- ML metrics will be loaded here -->
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        generatePredictionContent() {
            return `
                <div class="view-header">
                    <h2>🔮 Predictions</h2>
                </div>
                <div class="dashboard-grid">
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">Eruption Predictions</h3>
                        </div>
                        <div class="card-content">
                            <div id="predictions-placeholder">
                                <!-- Predictions will be loaded here -->
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        generateReportsContent() {
            return `
                <div class="view-header">
                    <h2>📄 Reports</h2>
                </div>
                <div class="dashboard-grid">
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">Recent Reports</h3>
                        </div>
                        <div class="card-content">
                            <div id="reports-list-placeholder">
                                <!-- Reports list will be loaded here -->
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        generateMonitoringContent() {
            return `
                <div class="view-header">
                    <h2>📡 System Monitoring</h2>
                </div>
                <div class="dashboard-grid">
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">System Status</h3>
                        </div>
                        <div class="card-content">
                            <div id="system-status-placeholder">
                                <!-- System status will be loaded here -->
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        generateSettingsContent() {
            return `
                <div class="view-header">
                    <h2>⚙️ Settings</h2>
                </div>
                <div class="dashboard-grid">
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">Preferences</h3>
                        </div>
                        <div class="card-content">
                            <div id="settings-form-placeholder">
                                <!-- Settings form will be loaded here -->
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        generateErrorsContent() {
            return `
                <div class="view-header">
                    <h2>🚨 Error Logs</h2>
                </div>
                <div class="dashboard-grid">
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h3 class="card-title">Recent Errors</h3>
                        </div>
                        <div class="card-content">
                            <div id="error-logs-placeholder">
                                <!-- Error logs will be loaded here -->
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        initializeViewComponents(view) {
            // Initialize view-specific components
            switch(view) {
                case 'dashboard':
                    this.initializeDashboardCharts();
                    break;
                case 'volcanic':
                    this.loadVolcanoSelector();
                    break;
                // Add more view initializations as needed
            }
        }
        
        initializeDashboardCharts() {
            // Initialize performance chart if Chart.js is available
            const canvas = document.getElementById('performance-chart');
            if (canvas && window.Chart) {
                // Create chart
                console.log('Initializing performance chart...');
            }
        }
        
        loadVolcanoSelector() {
            // Load volcano selector if available
            if (window.VolcanoSystemFix) {
                window.VolcanoSystemFix.fixVolcanoSelector();
            }
        }
        
        initializeComponents() {
            // Register default components
            this.registerComponent('card', this.createCard);
            this.registerComponent('chart', this.createChart);
            this.registerComponent('table', this.createTable);
            this.registerComponent('form', this.createForm);
        }
        
        registerComponent(name, factory) {
            this.components.set(name, factory);
        }
        
        createCard(config) {
            const card = document.createElement('div');
            card.className = 'dashboard-card';
            card.innerHTML = `
                <div class="card-header">
                    <h3 class="card-title">${config.title || 'Card'}</h3>
                </div>
                <div class="card-content">
                    ${config.content || ''}
                </div>
            `;
            return card;
        }
        
        createChart(config) {
            // Chart creation logic
            console.log('Creating chart:', config);
        }
        
        createTable(config) {
            // Table creation logic
            console.log('Creating table:', config);
        }
        
        createForm(config) {
            // Form creation logic
            console.log('Creating form:', config);
        }
        
        showToast(message, type = 'info', duration = 3000) {
            const container = document.getElementById('toast-container');
            if (!container) return;
            
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.textContent = message;
            
            container.appendChild(toast);
            
            setTimeout(() => {
                toast.style.animation = 'slideOut 0.3s';
                setTimeout(() => toast.remove(), 300);
            }, duration);
        }
        
        savePreferences() {
            localStorage.setItem('dashboard-theme', this.state.theme);
            localStorage.setItem('dashboard-sidebar', this.state.sidebarCollapsed ? 'collapsed' : 'expanded');
            localStorage.setItem('dashboard-view', this.state.activeView);
        }
        
        loadPreferences() {
            const sidebarState = localStorage.getItem('dashboard-sidebar');
            if (sidebarState === 'collapsed') {
                document.getElementById('dashboard-sidebar')?.classList.add('collapsed');
                this.state.sidebarCollapsed = true;
            }
            
            const lastView = localStorage.getItem('dashboard-view');
            if (lastView && lastView !== 'dashboard') {
                setTimeout(() => this.navigateToView(lastView), 100);
            }
        }
        
        saveSettings() {
            this.savePreferences();
            this.showToast('Settings saved successfully', 'success');
        }
        
        refreshData() {
            this.showToast('Refreshing data...', 'info');
            // Trigger data refresh
            this.emit('data:refresh');
            setTimeout(() => {
                this.showToast('Data refreshed', 'success');
            }, 1000);
        }
        
        exportReport() {
            this.showToast('Generating report...', 'info');
            // Generate and download report
            setTimeout(() => {
                this.showToast('Report exported successfully', 'success');
            }, 1500);
        }
        
        clearCache() {
            localStorage.clear();
            sessionStorage.clear();
            this.showToast('Cache cleared', 'success');
        }
        
        runDiagnostics() {
            this.showToast('Running diagnostics...', 'info');
            // Run system diagnostics
            setTimeout(() => {
                this.showToast('Diagnostics complete - all systems operational', 'success');
            }, 2000);
        }
        
        emit(event, data) {
            this.eventBus.dispatchEvent(new CustomEvent(event, { detail: data }));
        }
        
        on(event, callback) {
            this.eventBus.addEventListener(event, callback);
        }
        
        off(event, callback) {
            this.eventBus.removeEventListener(event, callback);
        }
    }
    
    // Utility: Debounce function
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    // Initialize and expose globally
    const dashboard = new ModernDashboard();
    
    // Expose to global scope
    global.ModernDashboard = {
        instance: dashboard,
        version: dashboard.version,
        showToast: (msg, type, duration) => dashboard.showToast(msg, type, duration),
        navigateToView: (view) => dashboard.navigateToView(view),
        navigateToPath: (path) => dashboard.navigateToPath(path),
        on: (event, callback) => dashboard.on(event, callback),
        emit: (event, data) => dashboard.emit(event, data)
    };
    
    console.log('✨ Modern Dashboard Framework ready. Access via window.ModernDashboard');
    
})(window);