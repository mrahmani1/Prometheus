/**
 * Prometheus Components - Task 2.2
 * Reusable component system with drag-and-drop, advanced controls, and async loading
 * @author Prometheus Platform
 * @version 3.0.0
 */

(function(global) {
    'use strict';
    
    console.log('🧩 Prometheus Components - Initializing reusable component system...');
    
    // Component Registry
    class ComponentRegistry {
        constructor() {
            this.components = new Map();
            this.instances = new Map();
            this.eventBus = new EventTarget();
            this.dragManager = new DragManager();
            this.loadingStates = new Map();
        }
        
        register(name, componentClass) {
            this.components.set(name, componentClass);
            console.log(`✅ Component registered: ${name}`);
        }
        
        create(name, container, props = {}) {
            const ComponentClass = this.components.get(name);
            if (!ComponentClass) {
                throw new Error(`Component "${name}" not found`);
            }
            
            const instance = new ComponentClass(container, props);
            const instanceId = this.generateId();
            instance.id = instanceId;
            this.instances.set(instanceId, instance);
            
            // Setup drag and drop if enabled
            if (props.draggable) {
                this.dragManager.makeDraggable(instance);
            }
            
            return instance;
        }
        
        destroy(instanceId) {
            const instance = this.instances.get(instanceId);
            if (instance) {
                if (instance.destroy) instance.destroy();
                this.instances.delete(instanceId);
            }
        }
        
        generateId() {
            return 'comp_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        }
        
        emit(event, data) {
            this.eventBus.dispatchEvent(new CustomEvent(event, { detail: data }));
        }
        
        on(event, callback) {
            this.eventBus.addEventListener(event, callback);
        }
    }
    
    // Base Component Class
    class BaseComponent {
        constructor(container, props = {}) {
            this.container = typeof container === 'string' ? 
                document.getElementById(container) || document.querySelector(container) : 
                container;
            this.props = { ...this.defaultProps, ...props };
            this.state = {};
            this.element = null;
            this.destroyed = false;
            
            if (!this.container) {
                throw new Error('Container element not found');
            }
            
            this.render();
            this.bindEvents();
            
            if (this.props.onMount) {
                this.props.onMount(this);
            }
        }
        
        get defaultProps() {
            return {
                className: '',
                style: {},
                disabled: false,
                loading: false
            };
        }
        
        render() {
            // Override in subclasses
            this.element = document.createElement('div');
            this.element.className = 'prometheus-component ' + this.props.className;
            this.container.appendChild(this.element);
        }
        
        bindEvents() {
            // Override in subclasses
        }
        
        setState(newState) {
            this.state = { ...this.state, ...newState };
            if (!this.destroyed) {
                this.update();
            }
        }
        
        update() {
            // Override in subclasses
        }
        
        setLoading(loading) {
            this.setState({ loading });
            if (loading) {
                this.showLoadingSkeleton();
            } else {
                this.hideLoadingSkeleton();
            }
        }
        
        showLoadingSkeleton() {
            if (this.element && !this.element.querySelector('.loading-skeleton')) {
                const skeleton = document.createElement('div');
                skeleton.className = 'loading-skeleton';
                skeleton.innerHTML = this.getSkeletonHTML();
                this.element.appendChild(skeleton);
            }
        }
        
        hideLoadingSkeleton() {
            const skeleton = this.element?.querySelector('.loading-skeleton');
            if (skeleton) {
                skeleton.remove();
            }
        }
        
        getSkeletonHTML() {
            return `
                <div class="skeleton-pulse">
                    <div class="skeleton-line"></div>
                    <div class="skeleton-line"></div>
                    <div class="skeleton-line short"></div>
                </div>
            `;
        }
        
        emit(event, data) {
            ComponentRegistry.instance.emit(event, { component: this, ...data });
        }
        
        destroy() {
            if (this.destroyed) return;
            
            this.destroyed = true;
            if (this.element) {
                this.element.remove();
            }
            
            if (this.props.onDestroy) {
                this.props.onDestroy(this);
            }
        }
    }
    
    // Drag and Drop Manager
    class DragManager {
        constructor() {
            this.draggedElement = null;
            this.dropZones = new Set();
            this.ghostElement = null;
        }
        
        makeDraggable(component) {
            if (!component.element) return;
            
            component.element.draggable = true;
            component.element.addEventListener('dragstart', (e) => this.handleDragStart(e, component));
            component.element.addEventListener('dragend', (e) => this.handleDragEnd(e, component));
        }
        
        makeDropZone(element, onDrop) {
            element.addEventListener('dragover', this.handleDragOver.bind(this));
            element.addEventListener('drop', (e) => this.handleDrop(e, onDrop));
            element.classList.add('drop-zone');
            this.dropZones.add(element);
        }
        
        handleDragStart(e, component) {
            this.draggedElement = component.element;
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', component.id);
            
            // Create ghost element
            this.createGhostElement(component.element);
            
            // Add dragging class
            component.element.classList.add('dragging');
            
            ComponentRegistry.instance.emit('drag:start', { component });
        }
        
        handleDragEnd(e, component) {
            if (this.ghostElement) {
                this.ghostElement.remove();
                this.ghostElement = null;
            }
            
            component.element.classList.remove('dragging');
            this.draggedElement = null;
            
            ComponentRegistry.instance.emit('drag:end', { component });
        }
        
        handleDragOver(e) {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
        }
        
        handleDrop(e, onDrop) {
            e.preventDefault();
            const componentId = e.dataTransfer.getData('text/plain');
            const component = ComponentRegistry.instance.instances.get(componentId);
            
            if (component && onDrop) {
                onDrop(component, e);
            }
            
            ComponentRegistry.instance.emit('drop', { component, target: e.target });
        }
        
        createGhostElement(element) {
            this.ghostElement = element.cloneNode(true);
            this.ghostElement.style.opacity = '0.5';
            this.ghostElement.style.position = 'fixed';
            this.ghostElement.style.pointerEvents = 'none';
            this.ghostElement.style.zIndex = '10000';
            document.body.appendChild(this.ghostElement);
        }
    }
    
    // Panel Component
    class Panel extends BaseComponent {
        get defaultProps() {
            return {
                ...super.defaultProps,
                title: 'Panel',
                collapsible: true,
                collapsed: false,
                closable: true,
                resizable: false,
                minWidth: 200,
                minHeight: 100
            };
        }
        
        render() {
            this.element = document.createElement('div');
            this.element.className = `prometheus-panel ${this.props.className}`;
            
            this.element.innerHTML = `
                <div class="panel-header">
                    <div class="panel-title">${this.props.title}</div>
                    <div class="panel-actions">
                        ${this.props.collapsible ? '<button class="panel-btn collapse-btn">−</button>' : ''}
                        ${this.props.closable ? '<button class="panel-btn close-btn">×</button>' : ''}
                    </div>
                </div>
                <div class="panel-content">
                    ${this.props.content || ''}
                </div>
                ${this.props.resizable ? '<div class="panel-resizer"></div>' : ''}
            `;
            
            this.container.appendChild(this.element);
            this.headerElement = this.element.querySelector('.panel-header');
            this.contentElement = this.element.querySelector('.panel-content');
            
            if (this.props.collapsed) {
                this.collapse();
            }
            
            if (this.props.resizable) {
                this.setupResizing();
            }
        }
        
        bindEvents() {
            const collapseBtn = this.element.querySelector('.collapse-btn');
            const closeBtn = this.element.querySelector('.close-btn');
            
            if (collapseBtn) {
                collapseBtn.addEventListener('click', () => this.toggle());
            }
            
            if (closeBtn) {
                closeBtn.addEventListener('click', () => this.close());
            }
        }
        
        toggle() {
            if (this.state.collapsed) {
                this.expand();
            } else {
                this.collapse();
            }
        }
        
        collapse() {
            this.setState({ collapsed: true });
            this.element.classList.add('collapsed');
            this.contentElement.style.display = 'none';
            
            const btn = this.element.querySelector('.collapse-btn');
            if (btn) btn.textContent = '+';
            
            this.emit('panel:collapse');
        }
        
        expand() {
            this.setState({ collapsed: false });
            this.element.classList.remove('collapsed');
            this.contentElement.style.display = '';
            
            const btn = this.element.querySelector('.collapse-btn');
            if (btn) btn.textContent = '−';
            
            this.emit('panel:expand');
        }
        
        close() {
            this.emit('panel:close');
            this.destroy();
        }
        
        setContent(content) {
            this.contentElement.innerHTML = content;
        }
        
        setupResizing() {
            const resizer = this.element.querySelector('.panel-resizer');
            let isResizing = false;
            
            resizer.addEventListener('mousedown', (e) => {
                isResizing = true;
                document.addEventListener('mousemove', handleResize);
                document.addEventListener('mouseup', stopResize);
                e.preventDefault();
            });
            
            const handleResize = (e) => {
                if (!isResizing) return;
                
                const rect = this.element.getBoundingClientRect();
                const newWidth = e.clientX - rect.left;
                const newHeight = e.clientY - rect.top;
                
                if (newWidth > this.props.minWidth) {
                    this.element.style.width = newWidth + 'px';
                }
                if (newHeight > this.props.minHeight) {
                    this.element.style.height = newHeight + 'px';
                }
            };
            
            const stopResize = () => {
                isResizing = false;
                document.removeEventListener('mousemove', handleResize);
                document.removeEventListener('mouseup', stopResize);
            };
        }
    }
    
    // Advanced Slider Component
    class AdvancedSlider extends BaseComponent {
        get defaultProps() {
            return {
                ...super.defaultProps,
                min: 0,
                max: 100,
                value: 50,
                step: 1,
                orientation: 'horizontal',
                showTooltip: true,
                showTicks: false,
                tickCount: 5,
                formatValue: (v) => v,
                onChange: null
            };
        }
        
        render() {
            this.element = document.createElement('div');
            this.element.className = `prometheus-slider ${this.props.orientation} ${this.props.className}`;
            
            this.element.innerHTML = `
                <div class="slider-track">
                    <div class="slider-fill"></div>
                    <div class="slider-handle">
                        ${this.props.showTooltip ? '<div class="slider-tooltip"></div>' : ''}
                    </div>
                </div>
                ${this.props.showTicks ? '<div class="slider-ticks"></div>' : ''}
            `;
            
            this.container.appendChild(this.element);
            
            this.trackElement = this.element.querySelector('.slider-track');
            this.fillElement = this.element.querySelector('.slider-fill');
            this.handleElement = this.element.querySelector('.slider-handle');
            this.tooltipElement = this.element.querySelector('.slider-tooltip');
            
            this.updateValue(this.props.value);
            
            if (this.props.showTicks) {
                this.renderTicks();
            }
        }
        
        bindEvents() {
            let isDragging = false;
            
            this.handleElement.addEventListener('mousedown', (e) => {
                isDragging = true;
                document.addEventListener('mousemove', handleDrag);
                document.addEventListener('mouseup', stopDrag);
                e.preventDefault();
            });
            
            const handleDrag = (e) => {
                if (!isDragging) return;
                this.updateFromMouse(e);
            };
            
            const stopDrag = () => {
                isDragging = false;
                document.removeEventListener('mousemove', handleDrag);
                document.removeEventListener('mouseup', stopDrag);
            };
            
            this.trackElement.addEventListener('click', (e) => {
                if (e.target === this.handleElement) return;
                this.updateFromMouse(e);
            });
        }
        
        updateFromMouse(e) {
            const rect = this.trackElement.getBoundingClientRect();
            const { min, max, step, orientation } = this.props;
            
            let percent;
            if (orientation === 'horizontal') {
                percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            } else {
                percent = 1 - Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
            }
            
            const rawValue = min + (max - min) * percent;
            const steppedValue = Math.round(rawValue / step) * step;
            const clampedValue = Math.max(min, Math.min(max, steppedValue));
            
            this.setValue(clampedValue);
        }
        
        setValue(value) {
            this.props.value = value;
            this.updateValue(value);
            
            if (this.props.onChange) {
                this.props.onChange(value);
            }
            
            this.emit('slider:change', { value });
        }
        
        updateValue(value) {
            const { min, max, formatValue } = this.props;
            const percent = (value - min) / (max - min);
            
            if (this.props.orientation === 'horizontal') {
                this.handleElement.style.left = (percent * 100) + '%';
                this.fillElement.style.width = (percent * 100) + '%';
            } else {
                this.handleElement.style.bottom = (percent * 100) + '%';
                this.fillElement.style.height = (percent * 100) + '%';
            }
            
            if (this.tooltipElement) {
                this.tooltipElement.textContent = formatValue(value);
            }
        }
        
        renderTicks() {
            const ticksContainer = this.element.querySelector('.slider-ticks');
            const { min, max, tickCount } = this.props;
            
            for (let i = 0; i <= tickCount; i++) {
                const tick = document.createElement('div');
                tick.className = 'slider-tick';
                
                const value = min + (max - min) * (i / tickCount);
                const percent = (i / tickCount) * 100;
                
                if (this.props.orientation === 'horizontal') {
                    tick.style.left = percent + '%';
                } else {
                    tick.style.bottom = percent + '%';
                }
                
                tick.setAttribute('data-value', value);
                ticksContainer.appendChild(tick);
            }
        }
    }
    
    // Multi-Select Dropdown Component
    class MultiSelectDropdown extends BaseComponent {
        get defaultProps() {
            return {
                ...super.defaultProps,
                options: [],
                selectedValues: [],
                placeholder: 'Select options...',
                searchable: true,
                maxHeight: 200,
                onChange: null
            };
        }
        
        constructor(container, props) {
            super(container, props);
            this.isOpen = false;
            this.filteredOptions = [...this.props.options];
        }
        
        render() {
            this.element = document.createElement('div');
            this.element.className = `prometheus-multiselect ${this.props.className}`;
            
            this.element.innerHTML = `
                <div class="multiselect-trigger">
                    <div class="multiselect-display">
                        <div class="selected-tags"></div>
                        <span class="multiselect-placeholder">${this.props.placeholder}</span>
                    </div>
                    <div class="multiselect-arrow">▼</div>
                </div>
                <div class="multiselect-dropdown">
                    ${this.props.searchable ? '<input type="text" class="multiselect-search" placeholder="Search...">' : ''}
                    <div class="multiselect-options"></div>
                </div>
            `;
            
            this.container.appendChild(this.element);
            
            this.triggerElement = this.element.querySelector('.multiselect-trigger');
            this.dropdownElement = this.element.querySelector('.multiselect-dropdown');
            this.searchElement = this.element.querySelector('.multiselect-search');
            this.optionsElement = this.element.querySelector('.multiselect-options');
            this.tagsElement = this.element.querySelector('.selected-tags');
            this.placeholderElement = this.element.querySelector('.multiselect-placeholder');
            
            this.updateDisplay();
            this.renderOptions();
        }
        
        bindEvents() {
            this.triggerElement.addEventListener('click', () => this.toggle());
            
            if (this.searchElement) {
                this.searchElement.addEventListener('input', (e) => {
                    this.filterOptions(e.target.value);
                });
            }
            
            // Close dropdown when clicking outside
            document.addEventListener('click', (e) => {
                if (!this.element.contains(e.target)) {
                    this.close();
                }
            });
        }
        
        toggle() {
            if (this.isOpen) {
                this.close();
            } else {
                this.open();
            }
        }
        
        open() {
            this.isOpen = true;
            this.element.classList.add('open');
            this.dropdownElement.style.maxHeight = this.props.maxHeight + 'px';
            
            if (this.searchElement) {
                setTimeout(() => this.searchElement.focus(), 0);
            }
        }
        
        close() {
            this.isOpen = false;
            this.element.classList.remove('open');
            this.dropdownElement.style.maxHeight = '0';
            
            if (this.searchElement) {
                this.searchElement.value = '';
                this.filterOptions('');
            }
        }
        
        filterOptions(searchTerm) {
            const term = searchTerm.toLowerCase();
            this.filteredOptions = this.props.options.filter(option => 
                option.label.toLowerCase().includes(term)
            );
            this.renderOptions();
        }
        
        renderOptions() {
            this.optionsElement.innerHTML = '';
            
            this.filteredOptions.forEach(option => {
                const optionElement = document.createElement('div');
                optionElement.className = 'multiselect-option';
                
                const isSelected = this.props.selectedValues.includes(option.value);
                if (isSelected) {
                    optionElement.classList.add('selected');
                }
                
                optionElement.innerHTML = `
                    <input type="checkbox" ${isSelected ? 'checked' : ''}>
                    <span class="option-label">${option.label}</span>
                `;
                
                optionElement.addEventListener('click', () => {
                    this.toggleOption(option);
                });
                
                this.optionsElement.appendChild(optionElement);
            });
        }
        
        toggleOption(option) {
            const selectedValues = [...this.props.selectedValues];
            const index = selectedValues.indexOf(option.value);
            
            if (index >= 0) {
                selectedValues.splice(index, 1);
            } else {
                selectedValues.push(option.value);
            }
            
            this.props.selectedValues = selectedValues;
            this.updateDisplay();
            this.renderOptions();
            
            if (this.props.onChange) {
                this.props.onChange(selectedValues);
            }
            
            this.emit('multiselect:change', { selectedValues });
        }
        
        updateDisplay() {
            this.tagsElement.innerHTML = '';
            
            const selectedOptions = this.props.options.filter(opt => 
                this.props.selectedValues.includes(opt.value)
            );
            
            if (selectedOptions.length === 0) {
                this.placeholderElement.style.display = 'block';
            } else {
                this.placeholderElement.style.display = 'none';
                
                selectedOptions.forEach(option => {
                    const tag = document.createElement('div');
                    tag.className = 'selected-tag';
                    tag.innerHTML = `
                        <span>${option.label}</span>
                        <button class="tag-remove">×</button>
                    `;
                    
                    tag.querySelector('.tag-remove').addEventListener('click', (e) => {
                        e.stopPropagation();
                        this.toggleOption(option);
                    });
                    
                    this.tagsElement.appendChild(tag);
                });
            }
        }
    }
    
    // Date/Time Range Picker Component
    class DateTimeRangePicker extends BaseComponent {
        get defaultProps() {
            return {
                ...super.defaultProps,
                startDate: null,
                endDate: null,
                format: 'YYYY-MM-DD HH:mm',
                showTime: true,
                minDate: null,
                maxDate: null,
                onChange: null
            };
        }
        
        render() {
            this.element = document.createElement('div');
            this.element.className = `prometheus-datetime-range ${this.props.className}`;
            
            this.element.innerHTML = `
                <div class="datetime-inputs">
                    <div class="datetime-input-group">
                        <label>From:</label>
                        <input type="datetime-local" class="datetime-start" />
                    </div>
                    <div class="datetime-input-group">
                        <label>To:</label>
                        <input type="datetime-local" class="datetime-end" />
                    </div>
                </div>
                <div class="datetime-presets">
                    <button class="preset-btn" data-preset="1h">Last Hour</button>
                    <button class="preset-btn" data-preset="24h">Last 24h</button>
                    <button class="preset-btn" data-preset="7d">Last 7 days</button>
                    <button class="preset-btn" data-preset="30d">Last 30 days</button>
                </div>
            `;
            
            this.container.appendChild(this.element);
            
            this.startInput = this.element.querySelector('.datetime-start');
            this.endInput = this.element.querySelector('.datetime-end');
            
            this.initializeValues();
        }
        
        bindEvents() {
            this.startInput.addEventListener('change', () => {
                this.props.startDate = new Date(this.startInput.value);
                this.validateRange();
                this.onChange();
            });
            
            this.endInput.addEventListener('change', () => {
                this.props.endDate = new Date(this.endInput.value);
                this.validateRange();
                this.onChange();
            });
            
            this.element.querySelectorAll('.preset-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    this.applyPreset(btn.dataset.preset);
                });
            });
        }
        
        initializeValues() {
            if (this.props.startDate) {
                this.startInput.value = this.formatForInput(this.props.startDate);
            }
            if (this.props.endDate) {
                this.endInput.value = this.formatForInput(this.props.endDate);
            }
        }
        
        formatForInput(date) {
            return date.toISOString().slice(0, 16);
        }
        
        validateRange() {
            if (this.props.startDate && this.props.endDate) {
                if (this.props.startDate > this.props.endDate) {
                    // Swap dates
                    [this.props.startDate, this.props.endDate] = [this.props.endDate, this.props.startDate];
                    this.initializeValues();
                }
            }
        }
        
        applyPreset(preset) {
            const now = new Date();
            let startDate;
            
            switch (preset) {
                case '1h':
                    startDate = new Date(now.getTime() - 60 * 60 * 1000);
                    break;
                case '24h':
                    startDate = new Date(now.getTime() - 24 * 60 * 60 * 1000);
                    break;
                case '7d':
                    startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
                    break;
                case '30d':
                    startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
                    break;
                default:
                    return;
            }
            
            this.props.startDate = startDate;
            this.props.endDate = now;
            this.initializeValues();
            this.onChange();
        }
        
        onChange() {
            if (this.props.onChange) {
                this.props.onChange({
                    startDate: this.props.startDate,
                    endDate: this.props.endDate
                });
            }
            
            this.emit('datetime:change', {
                startDate: this.props.startDate,
                endDate: this.props.endDate
            });
        }
    }
    
    // Initialize component registry
    const registry = new ComponentRegistry();
    ComponentRegistry.instance = registry;
    
    // Register default components
    registry.register('Panel', Panel);
    registry.register('AdvancedSlider', AdvancedSlider);
    registry.register('MultiSelectDropdown', MultiSelectDropdown);
    registry.register('DateTimeRangePicker', DateTimeRangePicker);
    
    // Global API
    global.PrometheusComponents = {
        registry,
        BaseComponent,
        Panel,
        AdvancedSlider,
        MultiSelectDropdown,
        DateTimeRangePicker,
        DragManager,
        
        // Convenience methods
        create: (name, container, props) => registry.create(name, container, props),
        register: (name, componentClass) => registry.register(name, componentClass),
        destroy: (instanceId) => registry.destroy(instanceId),
        on: (event, callback) => registry.on(event, callback),
        emit: (event, data) => registry.emit(event, data)
    };
    
    console.log('✨ Prometheus Components ready. Access via window.PrometheusComponents');
    
})(window);