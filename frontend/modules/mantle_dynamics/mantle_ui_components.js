/**
 * Mantle UI Components - Task 1.3
 * Modern React-like component system for Mantle Dynamics Simulator
 * Fixes control panel alignment, slider responsiveness, touch support, and data layers
 * @author Prometheus Platform
 * @version 2.0.0
 */

(function(global) {
    'use strict';
    
    console.log('🌋 Mantle UI Components - Initializing modern control system...');
    
    // Component State Manager (React-like state management)
    class StateManager {
        constructor() {
            this.state = {};
            this.listeners = new Map();
            this.components = new Map();
        }
        
        setState(componentId, newState) {
            const oldState = this.state[componentId] || {};
            this.state[componentId] = { ...oldState, ...newState };
            this.notifyListeners(componentId);
        }
        
        getState(componentId) {
            return this.state[componentId] || {};
        }
        
        subscribe(componentId, callback) {
            if (!this.listeners.has(componentId)) {
                this.listeners.set(componentId, new Set());
            }
            this.listeners.get(componentId).add(callback);
        }
        
        unsubscribe(componentId, callback) {
            if (this.listeners.has(componentId)) {
                this.listeners.get(componentId).delete(callback);
            }
        }
        
        notifyListeners(componentId) {
            if (this.listeners.has(componentId)) {
                this.listeners.get(componentId).forEach(callback => {
                    callback(this.state[componentId]);
                });
            }
        }
        
        registerComponent(component) {
            this.components.set(component.id, component);
        }
        
        getComponent(componentId) {
            return this.components.get(componentId);
        }
    }
    
    // Global state manager instance
    const stateManager = new StateManager();
    
    // Base Component Class (React-like component)
    class Component {
        constructor(id, props = {}) {
            this.id = id;
            this.props = props;
            this.element = null;
            this.children = [];
            this.mounted = false;
            
            // Register with state manager
            stateManager.registerComponent(this);
            
            // Initialize state
            this.state = {};
            this.setState = this.setState.bind(this);
        }
        
        setState(newState) {
            const oldState = this.state;
            this.state = { ...oldState, ...newState };
            stateManager.setState(this.id, this.state);
            
            if (this.mounted) {
                this.update();
            }
        }
        
        mount(container) {
            if (typeof container === 'string') {
                container = document.querySelector(container);
            }
            
            if (!container) {
                console.error(`Cannot mount component ${this.id}: container not found`);
                return;
            }
            
            this.element = this.render();
            container.appendChild(this.element);
            this.mounted = true;
            
            // Call lifecycle method
            this.componentDidMount();
        }
        
        update() {
            if (!this.mounted || !this.element) return;
            
            const newElement = this.render();
            this.element.replaceWith(newElement);
            this.element = newElement;
            
            // Call lifecycle method
            this.componentDidUpdate();
        }
        
        unmount() {
            if (this.element && this.element.parentNode) {
                this.element.parentNode.removeChild(this.element);
            }
            this.mounted = false;
            
            // Call lifecycle method
            this.componentWillUnmount();
        }
        
        // Lifecycle methods (to be overridden)
        componentDidMount() {}
        componentDidUpdate() {}
        componentWillUnmount() {}
        
        // Render method (to be overridden)
        render() {
            const div = document.createElement('div');
            div.className = this.props.className || '';
            div.id = this.id;
            return div;
        }
        
        // Helper method to create elements
        createElement(tag, props = {}, children = []) {
            const element = document.createElement(tag);
            
            // Set properties
            Object.keys(props).forEach(key => {
                if (key === 'className') {
                    element.className = props[key];
                } else if (key === 'style' && typeof props[key] === 'object') {
                    Object.assign(element.style, props[key]);
                } else if (key.startsWith('on')) {
                    const eventName = key.substring(2).toLowerCase();
                    element.addEventListener(eventName, props[key]);
                } else if (key === 'dataset') {
                    Object.keys(props[key]).forEach(dataKey => {
                        element.dataset[dataKey] = props[key][dataKey];
                    });
                } else {
                    element[key] = props[key];
                }
            });
            
            // Add children
            children.forEach(child => {
                if (typeof child === 'string') {
                    element.appendChild(document.createTextNode(child));
                } else if (child instanceof Node) {
                    element.appendChild(child);
                }
            });
            
            return element;
        }
    }
    
    // Modern Slider Component with Touch Support
    class SliderComponent extends Component {
        constructor(id, props) {
            super(id, props);
            
            this.state = {
                value: props.value || props.min || 0,
                isDragging: false,
                isTouching: false
            };
            
            // Bind methods
            this.handleMouseDown = this.handleMouseDown.bind(this);
            this.handleMouseMove = this.handleMouseMove.bind(this);
            this.handleMouseUp = this.handleMouseUp.bind(this);
            this.handleTouchStart = this.handleTouchStart.bind(this);
            this.handleTouchMove = this.handleTouchMove.bind(this);
            this.handleTouchEnd = this.handleTouchEnd.bind(this);
            this.handleKeyDown = this.handleKeyDown.bind(this);
        }
        
        componentDidMount() {
            // Add global event listeners for drag
            document.addEventListener('mousemove', this.handleMouseMove);
            document.addEventListener('mouseup', this.handleMouseUp);
            document.addEventListener('touchmove', this.handleTouchMove, { passive: false });
            document.addEventListener('touchend', this.handleTouchEnd);
        }
        
        componentWillUnmount() {
            // Remove global event listeners
            document.removeEventListener('mousemove', this.handleMouseMove);
            document.removeEventListener('mouseup', this.handleMouseUp);
            document.removeEventListener('touchmove', this.handleTouchMove);
            document.removeEventListener('touchend', this.handleTouchEnd);
        }
        
        handleMouseDown(e) {
            e.preventDefault();
            this.setState({ isDragging: true });
            this.updateValue(e.clientX);
        }
        
        handleMouseMove(e) {
            if (this.state.isDragging) {
                e.preventDefault();
                this.updateValue(e.clientX);
            }
        }
        
        handleMouseUp(e) {
            if (this.state.isDragging) {
                e.preventDefault();
                this.setState({ isDragging: false });
            }
        }
        
        handleTouchStart(e) {
            e.preventDefault();
            this.setState({ isTouching: true });
            const touch = e.touches[0];
            this.updateValue(touch.clientX);
        }
        
        handleTouchMove(e) {
            if (this.state.isTouching) {
                e.preventDefault();
                const touch = e.touches[0];
                this.updateValue(touch.clientX);
            }
        }
        
        handleTouchEnd(e) {
            if (this.state.isTouching) {
                e.preventDefault();
                this.setState({ isTouching: false });
            }
        }
        
        handleKeyDown(e) {
            const step = parseFloat(this.props.step) || 1;
            let newValue = this.state.value;
            
            switch(e.key) {
                case 'ArrowLeft':
                case 'ArrowDown':
                    newValue -= step;
                    break;
                case 'ArrowRight':
                case 'ArrowUp':
                    newValue += step;
                    break;
                case 'Home':
                    newValue = this.props.min;
                    break;
                case 'End':
                    newValue = this.props.max;
                    break;
                default:
                    return;
            }
            
            e.preventDefault();
            this.setValue(newValue);
        }
        
        updateValue(clientX) {
            const track = this.element.querySelector('.slider-track');
            if (!track) return;
            
            const rect = track.getBoundingClientRect();
            const percent = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
            const value = this.props.min + percent * (this.props.max - this.props.min);
            
            this.setValue(value);
        }
        
        setValue(value) {
            // Apply step
            const step = parseFloat(this.props.step) || 1;
            value = Math.round(value / step) * step;
            
            // Clamp to range
            value = Math.max(this.props.min, Math.min(this.props.max, value));
            
            // Update state
            this.setState({ value });
            
            // Call onChange callback
            if (this.props.onChange) {
                this.props.onChange(value);
            }
            
            // Update display
            if (this.props.displayId) {
                const display = document.getElementById(this.props.displayId);
                if (display) {
                    display.textContent = this.formatValue(value);
                }
            }
        }
        
        formatValue(value) {
            if (this.props.format) {
                return this.props.format(value);
            }
            
            if (this.props.exponential) {
                return Math.pow(10, value).toExponential(1);
            }
            
            return value.toFixed(this.props.decimals || 0);
        }
        
        render() {
            const percent = ((this.state.value - this.props.min) / (this.props.max - this.props.min)) * 100;
            
            return this.createElement('div', {
                className: 'modern-slider-container',
                style: {
                    margin: '15px 0',
                    userSelect: 'none'
                }
            }, [
                // Label
                this.createElement('label', {
                    className: 'slider-label',
                    style: {
                        display: 'block',
                        marginBottom: '8px',
                        fontSize: '14px',
                        color: '#ccc',
                        fontWeight: '500'
                    }
                }, [
                    this.props.label,
                    this.props.displayId ? '' : `: ${this.formatValue(this.state.value)}`
                ]),
                
                // Slider track and thumb
                this.createElement('div', {
                    className: 'slider-track',
                    style: {
                        position: 'relative',
                        width: '100%',
                        height: '8px',
                        background: 'linear-gradient(to right, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.2) 100%)',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        overflow: 'visible'
                    },
                    onMouseDown: this.handleMouseDown,
                    onTouchStart: this.handleTouchStart
                }, [
                    // Progress fill
                    this.createElement('div', {
                        className: 'slider-fill',
                        style: {
                            position: 'absolute',
                            left: 0,
                            top: 0,
                            height: '100%',
                            width: `${percent}%`,
                            background: `linear-gradient(to right, ${this.props.color || '#FF6B35'}, ${this.props.colorEnd || '#f85125'})`,
                            borderRadius: '4px',
                            transition: this.state.isDragging || this.state.isTouching ? 'none' : 'width 0.2s ease'
                        }
                    }),
                    
                    // Thumb
                    this.createElement('div', {
                        className: 'slider-thumb',
                        tabIndex: 0,
                        style: {
                            position: 'absolute',
                            left: `${percent}%`,
                            top: '50%',
                            transform: 'translate(-50%, -50%)',
                            width: '24px',
                            height: '24px',
                            background: this.props.color || '#FF6B35',
                            borderRadius: '50%',
                            boxShadow: '0 2px 8px rgba(0,0,0,0.3), 0 0 0 2px rgba(255,255,255,0.2)',
                            cursor: 'grab',
                            transition: this.state.isDragging || this.state.isTouching ? 'none' : 'left 0.2s ease'
                        },
                        onKeyDown: this.handleKeyDown
                    })
                ]),
                
                // Min/Max labels
                this.props.showRange && this.createElement('div', {
                    style: {
                        display: 'flex',
                        justifyContent: 'space-between',
                        marginTop: '5px',
                        fontSize: '11px',
                        color: 'rgba(255,255,255,0.5)'
                    }
                }, [
                    this.createElement('span', {}, [this.formatValue(this.props.min)]),
                    this.createElement('span', {}, [this.formatValue(this.props.max)])
                ])
            ]);
        }
    }
    
    // Data Layer Checkbox Component
    class DataLayerCheckbox extends Component {
        constructor(id, props) {
            super(id, props);
            
            this.state = {
                checked: props.checked !== false,
                indeterminate: props.indeterminate || false
            };
            
            this.handleChange = this.handleChange.bind(this);
            this.handleKeyDown = this.handleKeyDown.bind(this);
        }
        
        handleChange(e) {
            const checked = e.target.checked;
            this.setState({ checked, indeterminate: false });
            
            if (this.props.onChange) {
                this.props.onChange(checked, this.props.layer);
            }
            
            // Update visualization if applicable
            this.updateVisualization(checked);
        }
        
        handleKeyDown(e) {
            if (e.key === ' ' || e.key === 'Enter') {
                e.preventDefault();
                this.setState({ 
                    checked: !this.state.checked,
                    indeterminate: false 
                });
                
                if (this.props.onChange) {
                    this.props.onChange(!this.state.checked, this.props.layer);
                }
                
                this.updateVisualization(!this.state.checked);
            }
        }
        
        updateVisualization(enabled) {
            // Dispatch custom event for visualization update
            const event = new CustomEvent('dataLayerToggle', {
                detail: {
                    layer: this.props.layer,
                    enabled: enabled
                }
            });
            document.dispatchEvent(event);
        }
        
        render() {
            return this.createElement('div', {
                className: 'data-layer-checkbox',
                style: {
                    display: 'flex',
                    alignItems: 'center',
                    padding: '8px',
                    margin: '5px 0',
                    background: 'rgba(255,255,255,0.05)',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    transition: 'background 0.2s ease'
                },
                onMouseEnter: (e) => {
                    e.currentTarget.style.background = 'rgba(255,255,255,0.1)';
                },
                onMouseLeave: (e) => {
                    e.currentTarget.style.background = 'rgba(255,255,255,0.05)';
                }
            }, [
                // Custom checkbox
                this.createElement('label', {
                    style: {
                        display: 'flex',
                        alignItems: 'center',
                        cursor: 'pointer',
                        width: '100%'
                    }
                }, [
                    this.createElement('input', {
                        type: 'checkbox',
                        checked: this.state.checked,
                        onChange: this.handleChange,
                        style: {
                            display: 'none'
                        }
                    }),
                    
                    // Custom checkbox visual
                    this.createElement('div', {
                        className: 'checkbox-visual',
                        tabIndex: 0,
                        style: {
                            width: '20px',
                            height: '20px',
                            marginRight: '10px',
                            border: '2px solid ' + (this.state.checked ? (this.props.color || '#FF6B35') : 'rgba(255,255,255,0.3)'),
                            borderRadius: '4px',
                            background: this.state.checked ? (this.props.color || '#FF6B35') : 'transparent',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            transition: 'all 0.2s ease'
                        },
                        onKeyDown: this.handleKeyDown
                    }, [
                        this.state.checked && this.createElement('span', {
                            style: {
                                color: 'white',
                                fontSize: '14px',
                                fontWeight: 'bold'
                            }
                        }, ['✓'])
                    ]),
                    
                    // Label text
                    this.createElement('span', {
                        style: {
                            flex: 1,
                            color: this.state.checked ? 'white' : 'rgba(255,255,255,0.7)',
                            transition: 'color 0.2s ease'
                        }
                    }, [this.props.label]),
                    
                    // Status indicator
                    this.props.showStatus && this.createElement('span', {
                        style: {
                            fontSize: '12px',
                            color: this.state.checked ? '#4CAF50' : 'rgba(255,255,255,0.5)'
                        }
                    }, [this.state.checked ? 'Active' : 'Inactive'])
                ])
            ]);
        }
    }
    
    // Visualization Mode Selector Component
    class VisualizationModeSelector extends Component {
        constructor(id, props) {
            super(id, props);
            
            this.state = {
                selectedMode: props.defaultMode || props.modes[0].value
            };
            
            this.handleModeSelect = this.handleModeSelect.bind(this);
        }
        
        handleModeSelect(mode) {
            this.setState({ selectedMode: mode });
            
            if (this.props.onModeChange) {
                this.props.onModeChange(mode);
            }
            
            // Update visualization
            if (typeof currentVisualizationMode !== 'undefined') {
                window.currentVisualizationMode = mode;
                if (typeof createMantleVisualization === 'function') {
                    createMantleVisualization();
                }
            }
        }
        
        render() {
            return this.createElement('div', {
                className: 'visualization-mode-selector',
                style: {
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: '10px',
                    margin: '10px 0'
                }
            }, 
            this.props.modes.map(mode => 
                this.createElement('button', {
                    className: `mode-button ${this.state.selectedMode === mode.value ? 'active' : ''}`,
                    style: {
                        padding: '10px 16px',
                        background: this.state.selectedMode === mode.value ? 
                            'linear-gradient(45deg, #FF6B35, #f85125)' : 
                            'rgba(255,255,255,0.1)',
                        border: '1px solid ' + (this.state.selectedMode === mode.value ? 
                            '#FF6B35' : 'rgba(255,255,255,0.2)'),
                        color: 'white',
                        borderRadius: '5px',
                        cursor: 'pointer',
                        transition: 'all 0.3s ease',
                        fontSize: '14px',
                        fontWeight: this.state.selectedMode === mode.value ? '600' : '400',
                        transform: this.state.selectedMode === mode.value ? 'scale(1.05)' : 'scale(1)'
                    },
                    onClick: () => this.handleModeSelect(mode.value),
                    onMouseEnter: (e) => {
                        if (this.state.selectedMode !== mode.value) {
                            e.target.style.transform = 'translateY(-2px)';
                            e.target.style.boxShadow = '0 4px 8px rgba(255,107,53,0.3)';
                        }
                    },
                    onMouseLeave: (e) => {
                        if (this.state.selectedMode !== mode.value) {
                            e.target.style.transform = 'scale(1)';
                            e.target.style.boxShadow = 'none';
                        }
                    }
                }, [mode.label])
            ));
        }
    }
    
    // Control Panel Component
    class ControlPanel extends Component {
        constructor(id, props) {
            super(id, props);
            
            this.state = {
                collapsed: false,
                activeSection: null
            };
            
            this.toggleCollapse = this.toggleCollapse.bind(this);
        }
        
        toggleCollapse() {
            this.setState({ collapsed: !this.state.collapsed });
        }
        
        render() {
            return this.createElement('div', {
                className: 'modern-control-panel',
                style: {
                    position: 'absolute',
                    top: '20px',
                    left: '20px',
                    background: 'rgba(0, 0, 0, 0.95)',
                    borderRadius: '10px',
                    padding: this.state.collapsed ? '15px' : '20px',
                    minWidth: this.state.collapsed ? '60px' : '320px',
                    maxHeight: this.state.collapsed ? '60px' : '90vh',
                    overflow: this.state.collapsed ? 'hidden' : 'auto',
                    backdropFilter: 'blur(20px)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    zIndex: 1000,
                    transition: 'all 0.3s ease',
                    boxShadow: '0 10px 40px rgba(0,0,0,0.5)'
                }
            }, [
                // Header
                this.createElement('div', {
                    style: {
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: this.state.collapsed ? '0' : '15px'
                    }
                }, [
                    !this.state.collapsed && this.createElement('h2', {
                        style: {
                            margin: 0,
                            fontSize: '18px',
                            color: '#FF6B35'
                        }
                    }, [this.props.title || 'Control Panel']),
                    
                    this.createElement('button', {
                        onClick: this.toggleCollapse,
                        style: {
                            background: 'transparent',
                            border: 'none',
                            color: 'white',
                            cursor: 'pointer',
                            fontSize: '20px',
                            padding: '5px',
                            transition: 'transform 0.3s ease'
                        },
                        onMouseEnter: (e) => {
                            e.target.style.transform = 'rotate(180deg)';
                        },
                        onMouseLeave: (e) => {
                            e.target.style.transform = 'rotate(0deg)';
                        }
                    }, [this.state.collapsed ? '☰' : '×'])
                ]),
                
                // Content
                !this.state.collapsed && this.createElement('div', {
                    className: 'panel-content'
                }, this.props.children || [])
            ]);
        }
    }
    
    // Initialize UI Components
    function initializeMantleUIComponents() {
        console.log('Initializing Mantle UI Components...');
        
        // Fix existing sliders
        fixExistingSliders();
        
        // Create modern sliders for key parameters
        createModernSliders();
        
        // Fix data layer checkboxes
        fixDataLayerCheckboxes();
        
        // Initialize visualization mode selector
        initializeVisualizationModeSelector();
        
        // Add touch gesture support
        addTouchGestureSupport();
        
        // Fix control panel layout
        fixControlPanelLayout();
        
        // Apply responsive design
        applyResponsiveDesign();
        
        console.log('✅ Mantle UI Components initialized successfully');
    }
    
    // Fix existing sliders with proper event delegation
    function fixExistingSliders() {
        const sliders = document.querySelectorAll('input[type="range"]');
        
        sliders.forEach(slider => {
            // Remove inline event handlers
            const newSlider = slider.cloneNode(true);
            slider.parentNode.replaceChild(newSlider, slider);
            
            // Add proper event delegation
            newSlider.addEventListener('input', function(e) {
                const value = parseFloat(e.target.value);
                const displayId = e.target.id.replace('-', '-value');
                const display = document.getElementById(displayId);
                
                if (display) {
                    // Format value appropriately
                    if (e.target.id === 'rayleigh-number') {
                        display.textContent = Math.pow(10, value).toExponential(1);
                    } else {
                        display.textContent = value;
                    }
                }
                
                // Trigger parameter update
                updateSimulationParameter(e.target.id, value);
            });
            
            // Add touch support
            addTouchToSlider(newSlider);
            
            // Style improvements
            Object.assign(newSlider.style, {
                width: '100%',
                cursor: 'pointer',
                WebkitAppearance: 'none',
                appearance: 'none',
                height: '8px',
                borderRadius: '4px',
                background: 'linear-gradient(to right, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.2) 100%)',
                outline: 'none'
            });
        });
    }
    
    // Add touch support to slider
    function addTouchToSlider(slider) {
        let touchActive = false;
        
        slider.addEventListener('touchstart', function(e) {
            touchActive = true;
            e.preventDefault();
            handleSliderTouch(e, slider);
        }, { passive: false });
        
        slider.addEventListener('touchmove', function(e) {
            if (touchActive) {
                e.preventDefault();
                handleSliderTouch(e, slider);
            }
        }, { passive: false });
        
        slider.addEventListener('touchend', function() {
            touchActive = false;
        });
    }
    
    // Handle slider touch events
    function handleSliderTouch(e, slider) {
        const touch = e.touches[0];
        const rect = slider.getBoundingClientRect();
        const percent = Math.max(0, Math.min(1, (touch.clientX - rect.left) / rect.width));
        const min = parseFloat(slider.min);
        const max = parseFloat(slider.max);
        const value = min + percent * (max - min);
        
        slider.value = value;
        slider.dispatchEvent(new Event('input', { bubbles: true }));
    }
    
    // Create modern sliders
    function createModernSliders() {
        // Replace Rayleigh Number slider
        const rayleighSlider = new SliderComponent('rayleigh-slider', {
            label: 'Rayleigh Number',
            min: 4,
            max: 8,
            step: 0.1,
            value: 6,
            exponential: true,
            displayId: 'rayleigh-value',
            color: '#FF6B35',
            showRange: true,
            onChange: (value) => {
                if (window.params) {
                    window.params.rayleighNumber = Math.pow(10, value);
                }
            }
        });
        
        // Replace Viscosity Contrast slider
        const viscositySlider = new SliderComponent('viscosity-slider', {
            label: 'Viscosity Contrast',
            min: 1,
            max: 1000,
            step: 10,
            value: 100,
            displayId: 'viscosity-value',
            color: '#2196F3',
            onChange: (value) => {
                if (window.params) {
                    window.params.viscosityContrast = value;
                }
            }
        });
        
        // Find container and mount if exists
        const controlGroup = document.querySelector('.control-group');
        if (controlGroup) {
            // Create container for modern sliders
            const modernContainer = document.createElement('div');
            modernContainer.className = 'modern-sliders';
            controlGroup.appendChild(modernContainer);
            
            // Mount sliders
            rayleighSlider.mount(modernContainer);
            viscositySlider.mount(modernContainer);
        }
    }
    
    // Fix data layer checkboxes
    function fixDataLayerCheckboxes() {
        const layers = [
            { id: 'temperature-layer', label: 'Temperature Field', layer: 'temperature', color: '#FF6B35' },
            { id: 'velocity-layer', label: 'Velocity Vectors', layer: 'velocity', color: '#2196F3' },
            { id: 'plumes-layer', label: 'Mantle Plumes', layer: 'plumes', color: '#4CAF50' },
            { id: 'streamlines-layer', label: 'Flow Streamlines', layer: 'streamlines', color: '#9C27B0' }
        ];
        
        // Create container for checkboxes
        const container = document.createElement('div');
        container.className = 'data-layer-controls';
        container.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            background: rgba(0,0,0,0.95);
            padding: 15px;
            border-radius: 10px;
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.1);
            z-index: 999;
        `;
        
        // Add title
        const title = document.createElement('h3');
        title.textContent = 'Data Layers';
        title.style.cssText = 'margin: 0 0 10px 0; color: #FF6B35; font-size: 16px;';
        container.appendChild(title);
        
        // Create checkboxes
        layers.forEach(layerConfig => {
            const checkbox = new DataLayerCheckbox(layerConfig.id, {
                label: layerConfig.label,
                layer: layerConfig.layer,
                color: layerConfig.color,
                checked: true,
                showStatus: true,
                onChange: (checked, layer) => {
                    console.log(`Layer ${layer} is now ${checked ? 'enabled' : 'disabled'}`);
                }
            });
            
            checkbox.mount(container);
        });
        
        // Add to page
        document.body.appendChild(container);
    }
    
    // Initialize visualization mode selector
    function initializeVisualizationModeSelector() {
        const modes = [
            { value: 'temperature', label: 'Temperature' },
            { value: 'velocity', label: 'Velocity Field' },
            { value: 'streamlines', label: 'Streamlines' },
            { value: 'plumes', label: 'Plume Tracking' },
            { value: 'composition', label: 'Composition' }
        ];
        
        const selector = new VisualizationModeSelector('viz-mode-selector', {
            modes: modes,
            defaultMode: 'temperature',
            onModeChange: (mode) => {
                console.log(`Visualization mode changed to: ${mode}`);
            }
        });
        
        // Find existing mode buttons and replace
        const existingContainer = document.querySelector('.visualization-mode');
        if (existingContainer) {
            existingContainer.innerHTML = '';
            selector.mount(existingContainer);
        }
    }
    
    // Add touch gesture support for 3D view
    function addTouchGestureSupport() {
        const canvas = document.querySelector('canvas');
        if (!canvas) return;
        
        let touchStartX = 0;
        let touchStartY = 0;
        let pinchStartDistance = 0;
        let isRotating = false;
        let isPinching = false;
        
        canvas.addEventListener('touchstart', function(e) {
            if (e.touches.length === 1) {
                // Single touch - rotation
                isRotating = true;
                touchStartX = e.touches[0].clientX;
                touchStartY = e.touches[0].clientY;
            } else if (e.touches.length === 2) {
                // Two touches - pinch zoom
                isPinching = true;
                const dx = e.touches[1].clientX - e.touches[0].clientX;
                const dy = e.touches[1].clientY - e.touches[0].clientY;
                pinchStartDistance = Math.sqrt(dx * dx + dy * dy);
            }
            e.preventDefault();
        }, { passive: false });
        
        canvas.addEventListener('touchmove', function(e) {
            if (isRotating && e.touches.length === 1) {
                const deltaX = e.touches[0].clientX - touchStartX;
                const deltaY = e.touches[0].clientY - touchStartY;
                
                if (window.scene) {
                    window.scene.rotation.y += deltaX * 0.01;
                    window.scene.rotation.x += deltaY * 0.01;
                }
                
                touchStartX = e.touches[0].clientX;
                touchStartY = e.touches[0].clientY;
            } else if (isPinching && e.touches.length === 2) {
                const dx = e.touches[1].clientX - e.touches[0].clientX;
                const dy = e.touches[1].clientY - e.touches[0].clientY;
                const distance = Math.sqrt(dx * dx + dy * dy);
                const scale = distance / pinchStartDistance;
                
                if (window.camera) {
                    window.camera.position.z *= (2 - scale);
                    window.camera.position.z = Math.max(5, Math.min(50, window.camera.position.z));
                }
                
                pinchStartDistance = distance;
            }
            e.preventDefault();
        }, { passive: false });
        
        canvas.addEventListener('touchend', function() {
            isRotating = false;
            isPinching = false;
        });
    }
    
    // Fix control panel layout
    function fixControlPanelLayout() {
        const panels = ['controls-panel', 'analysis-panel', 'simulation-controls'];
        
        panels.forEach(panelId => {
            const panel = document.getElementById(panelId);
            if (!panel) return;
            
            // Apply modern styles
            Object.assign(panel.style, {
                background: 'rgba(0, 0, 0, 0.95)',
                backdropFilter: 'blur(20px)',
                WebkitBackdropFilter: 'blur(20px)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                boxShadow: '0 10px 40px rgba(0,0,0,0.5)',
                transition: 'all 0.3s ease'
            });
            
            // Add hover effect
            panel.addEventListener('mouseenter', function() {
                this.style.boxShadow = '0 15px 50px rgba(255,107,53,0.2)';
            });
            
            panel.addEventListener('mouseleave', function() {
                this.style.boxShadow = '0 10px 40px rgba(0,0,0,0.5)';
            });
            
            // Make draggable
            makePanelDraggable(panel);
        });
    }
    
    // Make panel draggable
    function makePanelDraggable(panel) {
        let isDragging = false;
        let startX, startY, initialLeft, initialTop;
        
        const header = panel.querySelector('h2, h3');
        if (!header) return;
        
        header.style.cursor = 'move';
        header.style.userSelect = 'none';
        
        header.addEventListener('mousedown', startDrag);
        header.addEventListener('touchstart', startDrag, { passive: false });
        
        function startDrag(e) {
            isDragging = true;
            
            if (e.type === 'touchstart') {
                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
            } else {
                startX = e.clientX;
                startY = e.clientY;
            }
            
            initialLeft = panel.offsetLeft;
            initialTop = panel.offsetTop;
            
            document.addEventListener('mousemove', drag);
            document.addEventListener('touchmove', drag, { passive: false });
            document.addEventListener('mouseup', stopDrag);
            document.addEventListener('touchend', stopDrag);
            
            e.preventDefault();
        }
        
        function drag(e) {
            if (!isDragging) return;
            
            let clientX, clientY;
            
            if (e.type === 'touchmove') {
                clientX = e.touches[0].clientX;
                clientY = e.touches[0].clientY;
            } else {
                clientX = e.clientX;
                clientY = e.clientY;
            }
            
            const deltaX = clientX - startX;
            const deltaY = clientY - startY;
            
            panel.style.left = (initialLeft + deltaX) + 'px';
            panel.style.top = (initialTop + deltaY) + 'px';
            
            e.preventDefault();
        }
        
        function stopDrag() {
            isDragging = false;
            document.removeEventListener('mousemove', drag);
            document.removeEventListener('touchmove', drag);
            document.removeEventListener('mouseup', stopDrag);
            document.removeEventListener('touchend', stopDrag);
        }
    }
    
    // Apply responsive design
    function applyResponsiveDesign() {
        const style = document.createElement('style');
        style.textContent = `
            @media (max-width: 768px) {
                #controls-panel {
                    left: 10px !important;
                    right: 10px !important;
                    width: auto !important;
                    min-width: unset !important;
                }
                
                #analysis-panel {
                    display: none;
                }
                
                #simulation-controls {
                    bottom: 10px !important;
                    left: 10px !important;
                    right: 10px !important;
                    transform: none !important;
                    flex-wrap: wrap;
                }
                
                .control-group {
                    padding: 10px !important;
                }
                
                .slider-container {
                    margin: 8px 0 !important;
                }
                
                .button {
                    padding: 8px 12px !important;
                    font-size: 12px !important;
                }
            }
            
            @media (max-width: 480px) {
                #controls-panel h2 {
                    font-size: 16px !important;
                }
                
                .control-group h3 {
                    font-size: 14px !important;
                }
                
                .mode-button {
                    padding: 6px 10px !important;
                    font-size: 12px !important;
                }
            }
            
            /* Smooth animations */
            .panel, .control-group, .button, .slider-thumb {
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            /* Better scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: rgba(255,255,255,0.05);
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb {
                background: rgba(255,107,53,0.5);
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: rgba(255,107,53,0.7);
            }
        `;
        document.head.appendChild(style);
    }
    
    // Helper function to update simulation parameters
    function updateSimulationParameter(parameterId, value) {
        if (!window.params) return;
        
        const paramMap = {
            'rayleigh-number': () => window.params.rayleighNumber = Math.pow(10, value),
            'viscosity-contrast': () => window.params.viscosityContrast = value,
            'internal-heating': () => window.params.internalHeating = value,
            'core-temperature': () => window.params.coreTemperature = value,
            'surface-temperature': () => window.params.surfaceTemperature = value
        };
        
        if (paramMap[parameterId]) {
            paramMap[parameterId]();
            console.log(`Updated ${parameterId} to ${value}`);
        }
    }
    
    // Auto-initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeMantleUIComponents);
    } else {
        initializeMantleUIComponents();
    }
    
    // Export components and utilities for global access
    global.MantleUI = {
        Component,
        SliderComponent,
        DataLayerCheckbox,
        VisualizationModeSelector,
        ControlPanel,
        StateManager: stateManager,
        init: initializeMantleUIComponents
    };
    
    console.log('🎯 Mantle UI Components ready. Access via window.MantleUI');
    
})(window);