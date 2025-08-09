/**
 * Advanced Data Table - Task 2.3
 * High-performance data table with virtual scrolling, sorting, filtering, and export capabilities
 * @author Prometheus Platform
 * @version 3.0.0
 */

(function(global) {
    'use strict';
    
    console.log('📊 Advanced Data Table - Initializing high-performance table system...');
    
    class AdvancedDataTable {
        constructor(container, options = {}) {
            this.container = typeof container === 'string' ? 
                document.getElementById(container) || document.querySelector(container) : 
                container;
            
            if (!this.container) {
                throw new Error('Container element not found');
            }
            
            this.options = {
                // Data options
                data: [],
                columns: [],
                
                // Virtual scrolling
                virtualScrolling: true,
                rowHeight: 40,
                visibleRows: 20,
                bufferSize: 5,
                
                // Sorting
                sortable: true,
                multiSort: false,
                
                // Filtering
                filterable: true,
                globalFilter: true,
                columnFilters: true,
                
                // Selection
                selectable: true,
                multiSelect: true,
                selectAllEnabled: true,
                
                // Editing
                editable: false,
                editableColumns: [],
                
                // Export
                exportable: true,
                exportFormats: ['csv', 'json', 'excel'],
                
                // Pagination
                pagination: false,
                pageSize: 50,
                
                // Styling
                striped: true,
                bordered: true,
                hover: true,
                compact: false,
                
                // Events
                onRowClick: null,
                onRowSelect: null,
                onCellEdit: null,
                onSort: null,
                onFilter: null,
                onExport: null,
                
                ...options
            };
            
            this.state = {
                data: [...this.options.data],
                filteredData: [...this.options.data],
                sortedData: [...this.options.data],
                displayData: [],
                
                scrollTop: 0,
                startIndex: 0,
                endIndex: 0,
                
                selectedRows: new Set(),
                editingCell: null,
                
                sortColumns: [],
                filters: new Map(),
                globalFilterValue: '',
                
                loading: false,
                error: null
            };
            
            this.elements = {};
            this.observers = new Set();
            
            this.init();
        }
        
        init() {
            this.createStructure();
            this.setupVirtualScrolling();
            this.setupEventListeners();
            this.render();
            
            console.log('✅ Advanced Data Table initialized');
        }
        
        createStructure() {
            this.container.className = `advanced-data-table ${this.options.compact ? 'compact' : ''}`;
            
            this.container.innerHTML = `
                <div class="table-toolbar">
                    <div class="toolbar-left">
                        <div class="table-info">
                            <span class="row-count">0 rows</span>
                            <span class="selection-info" style="display: none;">0 selected</span>
                        </div>
                    </div>
                    
                    <div class="toolbar-center">
                        ${this.options.globalFilter ? `
                            <div class="global-filter">
                                <input type="text" placeholder="Search all columns..." class="filter-input">
                                <button class="clear-filter" title="Clear filter">×</button>
                            </div>
                        ` : ''}
                    </div>
                    
                    <div class="toolbar-right">
                        <div class="table-actions">
                            ${this.options.exportable ? `
                                <div class="export-dropdown">
                                    <button class="export-btn">📥 Export</button>
                                    <div class="export-menu">
                                        ${this.options.exportFormats.map(format => `
                                            <button class="export-format" data-format="${format}">
                                                ${format.toUpperCase()}
                                            </button>
                                        `).join('')}
                                    </div>
                                </div>
                            ` : ''}
                            
                            <button class="refresh-btn" title="Refresh data">⟲</button>
                            <button class="settings-btn" title="Table settings">⚙️</button>
                        </div>
                    </div>
                </div>
                
                <div class="table-container">
                    <div class="table-header">
                        <table class="data-table header-table">
                            <thead>
                                <!-- Header will be populated dynamically -->
                            </thead>
                        </table>
                    </div>
                    
                    <div class="table-body" style="height: ${this.options.rowHeight * this.options.visibleRows}px;">
                        <div class="virtual-spacer-top"></div>
                        <table class="data-table body-table">
                            <tbody>
                                <!-- Body will be populated dynamically -->
                            </tbody>
                        </table>
                        <div class="virtual-spacer-bottom"></div>
                    </div>
                </div>
                
                <div class="table-footer">
                    ${this.options.pagination ? `
                        <div class="pagination-controls">
                            <button class="page-btn first-page" disabled>⟪</button>
                            <button class="page-btn prev-page" disabled>⟨</button>
                            <span class="page-info">Page 1 of 1</span>
                            <button class="page-btn next-page" disabled>⟩</button>
                            <button class="page-btn last-page" disabled>⟫</button>
                            
                            <select class="page-size-select">
                                <option value="25">25 per page</option>
                                <option value="50" selected>50 per page</option>
                                <option value="100">100 per page</option>
                                <option value="500">500 per page</option>
                            </select>
                        </div>
                    ` : ''}
                    
                    <div class="table-status">
                        <span class="status-text">Ready</span>
                    </div>
                </div>
                
                <!-- Loading overlay -->
                <div class="loading-overlay" style="display: none;">
                    <div class="loading-spinner"></div>
                    <div class="loading-text">Loading data...</div>
                </div>
            `;
            
            // Cache important elements
            this.elements = {
                toolbar: this.container.querySelector('.table-toolbar'),
                rowCount: this.container.querySelector('.row-count'),
                selectionInfo: this.container.querySelector('.selection-info'),
                globalFilter: this.container.querySelector('.filter-input'),
                clearFilter: this.container.querySelector('.clear-filter'),
                exportBtn: this.container.querySelector('.export-btn'),
                exportMenu: this.container.querySelector('.export-menu'),
                refreshBtn: this.container.querySelector('.refresh-btn'),
                settingsBtn: this.container.querySelector('.settings-btn'),
                
                tableContainer: this.container.querySelector('.table-container'),
                tableHeader: this.container.querySelector('.table-header'),
                tableBody: this.container.querySelector('.table-body'),
                headerTable: this.container.querySelector('.header-table'),
                bodyTable: this.container.querySelector('.body-table'),
                thead: this.container.querySelector('.header-table thead'),
                tbody: this.container.querySelector('.body-table tbody'),
                
                virtualSpacerTop: this.container.querySelector('.virtual-spacer-top'),
                virtualSpacerBottom: this.container.querySelector('.virtual-spacer-bottom'),
                
                tableFooter: this.container.querySelector('.table-footer'),
                statusText: this.container.querySelector('.status-text'),
                loadingOverlay: this.container.querySelector('.loading-overlay')
            };
        }
        
        setupVirtualScrolling() {
            if (!this.options.virtualScrolling) return;
            
            this.elements.tableBody.addEventListener('scroll', () => {
                this.handleScroll();
            });
            
            // Sync header scroll with body
            this.elements.tableBody.addEventListener('scroll', () => {
                this.elements.tableHeader.scrollLeft = this.elements.tableBody.scrollLeft;
            });
        }
        
        setupEventListeners() {
            // Global filter
            if (this.elements.globalFilter) {
                this.elements.globalFilter.addEventListener('input', (e) => {
                    this.setGlobalFilter(e.target.value);
                });
                
                this.elements.clearFilter?.addEventListener('click', () => {
                    this.clearGlobalFilter();
                });
            }
            
            // Export functionality
            if (this.elements.exportBtn) {
                this.elements.exportBtn.addEventListener('click', () => {
                    this.toggleExportMenu();
                });
                
                this.elements.exportMenu?.querySelectorAll('.export-format').forEach(btn => {
                    btn.addEventListener('click', (e) => {
                        this.exportData(e.target.dataset.format);
                    });
                });
            }
            
            // Refresh button
            this.elements.refreshBtn?.addEventListener('click', () => {
                this.refresh();
            });
            
            // Settings button
            this.elements.settingsBtn?.addEventListener('click', () => {
                this.showSettings();
            });
            
            // Table interactions
            this.elements.tbody?.addEventListener('click', (e) => {
                this.handleTableClick(e);
            });
            
            this.elements.tbody?.addEventListener('dblclick', (e) => {
                this.handleTableDoubleClick(e);
            });
            
            // Header interactions
            this.elements.thead?.addEventListener('click', (e) => {
                this.handleHeaderClick(e);
            });
            
            // Keyboard shortcuts
            this.container.addEventListener('keydown', (e) => {
                this.handleKeyboard(e);
            });
            
            // Close export menu when clicking outside
            document.addEventListener('click', (e) => {
                if (!this.container.contains(e.target)) {
                    this.closeExportMenu();
                }
            });
        }
        
        render() {
            this.renderHeader();
            this.processData();
            this.renderBody();
            this.updateInfo();
        }
        
        renderHeader() {
            if (!this.options.columns.length) return;
            
            const headerRow = document.createElement('tr');
            
            // Selection column
            if (this.options.selectable) {
                const selectAllCell = document.createElement('th');
                selectAllCell.className = 'select-column';
                selectAllCell.innerHTML = this.options.selectAllEnabled ? 
                    '<input type="checkbox" class="select-all-checkbox">' : '';
                headerRow.appendChild(selectAllCell);
            }
            
            // Data columns
            this.options.columns.forEach((column, index) => {
                const th = document.createElement('th');
                th.className = 'column-header';
                th.dataset.columnIndex = index;
                th.dataset.columnKey = column.key;
                
                let headerContent = `<span class="column-title">${column.title || column.key}</span>`;
                
                // Sortable indicator
                if (this.options.sortable && column.sortable !== false) {
                    th.classList.add('sortable');
                    headerContent += '<span class="sort-indicator"></span>';
                }
                
                // Filter input
                if (this.options.columnFilters && column.filterable !== false) {
                    headerContent += `
                        <div class="column-filter">
                            <input type="text" class="column-filter-input" placeholder="Filter..." 
                                   data-column="${column.key}" onclick="event.stopPropagation()">
                        </div>
                    `;
                }
                
                th.innerHTML = headerContent;
                
                // Apply column styles
                if (column.width) {
                    th.style.width = column.width;
                }
                if (column.minWidth) {
                    th.style.minWidth = column.minWidth;
                }
                
                headerRow.appendChild(th);
            });
            
            this.elements.thead.innerHTML = '';
            this.elements.thead.appendChild(headerRow);
            
            // Setup column filter listeners
            if (this.options.columnFilters) {
                this.elements.thead.querySelectorAll('.column-filter-input').forEach(input => {
                    input.addEventListener('input', (e) => {
                        this.setColumnFilter(e.target.dataset.column, e.target.value);
                    });
                });
            }
            
            // Setup select all checkbox
            if (this.options.selectAllEnabled) {
                const selectAllCheckbox = this.elements.thead.querySelector('.select-all-checkbox');
                selectAllCheckbox?.addEventListener('change', (e) => {
                    this.selectAll(e.target.checked);
                });
            }
        }
        
        processData() {
            let data = [...this.state.data];
            
            // Apply global filter
            if (this.state.globalFilterValue) {
                data = this.applyGlobalFilter(data);
            }
            
            // Apply column filters
            if (this.state.filters.size > 0) {
                data = this.applyColumnFilters(data);
            }
            
            this.state.filteredData = data;
            
            // Apply sorting
            if (this.state.sortColumns.length > 0) {
                data = this.applySorting(data);
            }
            
            this.state.sortedData = data;
            
            // Calculate virtual scrolling indices
            if (this.options.virtualScrolling) {
                this.calculateVisibleRange();
                this.state.displayData = data.slice(this.state.startIndex, this.state.endIndex);
            } else {
                this.state.displayData = data;
            }
        }
        
        applyGlobalFilter(data) {
            const filterValue = this.state.globalFilterValue.toLowerCase();
            return data.filter(row => {
                return this.options.columns.some(column => {
                    const cellValue = this.getCellValue(row, column);
                    return String(cellValue).toLowerCase().includes(filterValue);
                });
            });
        }
        
        applyColumnFilters(data) {
            return data.filter(row => {
                return Array.from(this.state.filters.entries()).every(([columnKey, filterValue]) => {
                    if (!filterValue) return true;
                    
                    const column = this.options.columns.find(col => col.key === columnKey);
                    if (!column) return true;
                    
                    const cellValue = this.getCellValue(row, column);
                    return String(cellValue).toLowerCase().includes(filterValue.toLowerCase());
                });
            });
        }
        
        applySorting(data) {
            if (!this.state.sortColumns.length) return data;
            
            return [...data].sort((a, b) => {
                for (const sortCol of this.state.sortColumns) {
                    const column = this.options.columns.find(col => col.key === sortCol.key);
                    if (!column) continue;
                    
                    const aVal = this.getCellValue(a, column);
                    const bVal = this.getCellValue(b, column);
                    
                    let result = 0;
                    
                    if (column.sortFunction) {
                        result = column.sortFunction(aVal, bVal);
                    } else {
                        // Default sorting
                        if (aVal < bVal) result = -1;
                        else if (aVal > bVal) result = 1;
                    }
                    
                    if (result !== 0) {
                        return sortCol.direction === 'desc' ? -result : result;
                    }
                }
                return 0;
            });
        }
        
        calculateVisibleRange() {
            const scrollTop = this.elements.tableBody.scrollTop;
            const containerHeight = this.elements.tableBody.clientHeight;
            const rowHeight = this.options.rowHeight;
            
            this.state.startIndex = Math.max(0, 
                Math.floor(scrollTop / rowHeight) - this.options.bufferSize
            );
            
            const visibleRowCount = Math.ceil(containerHeight / rowHeight);
            this.state.endIndex = Math.min(this.state.sortedData.length,
                this.state.startIndex + visibleRowCount + (this.options.bufferSize * 2)
            );
            
            // Update spacers
            const topSpacerHeight = this.state.startIndex * rowHeight;
            const bottomSpacerHeight = (this.state.sortedData.length - this.state.endIndex) * rowHeight;
            
            this.elements.virtualSpacerTop.style.height = topSpacerHeight + 'px';
            this.elements.virtualSpacerBottom.style.height = bottomSpacerHeight + 'px';
        }
        
        renderBody() {
            const tbody = this.elements.tbody;
            tbody.innerHTML = '';
            
            this.state.displayData.forEach((row, displayIndex) => {
                const actualIndex = this.options.virtualScrolling ? 
                    this.state.startIndex + displayIndex : displayIndex;
                    
                const tr = this.createTableRow(row, actualIndex);
                tbody.appendChild(tr);
            });
            
            // Update row selection states
            this.updateRowSelectionDisplay();
        }
        
        createTableRow(rowData, rowIndex) {
            const tr = document.createElement('tr');
            tr.className = 'data-row';
            tr.dataset.rowIndex = rowIndex;
            
            // Apply row classes
            if (this.options.striped && rowIndex % 2 === 1) {
                tr.classList.add('striped');
            }
            
            if (this.state.selectedRows.has(rowIndex)) {
                tr.classList.add('selected');
            }
            
            // Selection column
            if (this.options.selectable) {
                const selectCell = document.createElement('td');
                selectCell.className = 'select-column';
                selectCell.innerHTML = '<input type="checkbox" class="row-checkbox">';
                tr.appendChild(selectCell);
            }
            
            // Data columns
            this.options.columns.forEach((column, columnIndex) => {
                const td = document.createElement('td');
                td.className = 'data-cell';
                td.dataset.columnIndex = columnIndex;
                td.dataset.columnKey = column.key;
                td.dataset.rowIndex = rowIndex;
                
                // Get and format cell value
                const cellValue = this.getCellValue(rowData, column);
                const formattedValue = this.formatCellValue(cellValue, column);
                
                // Make editable if enabled
                if (this.options.editable && this.isColumnEditable(column)) {
                    td.classList.add('editable');
                    td.setAttribute('contenteditable', 'false');
                }
                
                td.innerHTML = formattedValue;
                
                // Apply column styles
                if (column.align) {
                    td.style.textAlign = column.align;
                }
                if (column.width) {
                    td.style.width = column.width;
                }
                
                tr.appendChild(td);
            });
            
            return tr;
        }
        
        getCellValue(rowData, column) {
            if (column.valueGetter) {
                return column.valueGetter(rowData);
            }
            return rowData[column.key];
        }
        
        formatCellValue(value, column) {
            if (value === null || value === undefined) {
                return '<span class="null-value">—</span>';
            }
            
            if (column.formatter) {
                return column.formatter(value);
            }
            
            // Default formatting based on column type
            switch (column.type) {
                case 'number':
                    return typeof value === 'number' ? value.toLocaleString() : value;
                case 'currency':
                    return new Intl.NumberFormat('en-US', {
                        style: 'currency',
                        currency: 'USD'
                    }).format(value);
                case 'percent':
                    return (value * 100).toFixed(2) + '%';
                case 'date':
                    return new Date(value).toLocaleDateString();
                case 'datetime':
                    return new Date(value).toLocaleString();
                case 'boolean':
                    return value ? '✓' : '✗';
                default:
                    return String(value);
            }
        }
        
        handleScroll() {
            if (!this.options.virtualScrolling) return;
            
            this.processData();
            this.renderBody();
        }
        
        handleTableClick(e) {
            const row = e.target.closest('.data-row');
            if (!row) return;
            
            const rowIndex = parseInt(row.dataset.rowIndex);
            
            // Handle checkbox clicks
            if (e.target.classList.contains('row-checkbox')) {
                this.toggleRowSelection(rowIndex, e.target.checked);
                return;
            }
            
            // Handle cell clicks
            const cell = e.target.closest('.data-cell');
            if (cell) {
                if (this.options.onRowClick) {
                    this.options.onRowClick(this.getRowData(rowIndex), rowIndex);
                }
                
                // Select row if selectable
                if (this.options.selectable && !e.ctrlKey && !e.metaKey) {
                    if (this.options.multiSelect) {
                        this.toggleRowSelection(rowIndex);
                    } else {
                        this.selectRow(rowIndex, true);
                    }
                }
            }
        }
        
        handleTableDoubleClick(e) {
            const cell = e.target.closest('.data-cell');
            if (!cell || !this.options.editable) return;
            
            const columnKey = cell.dataset.columnKey;
            const column = this.options.columns.find(col => col.key === columnKey);
            
            if (this.isColumnEditable(column)) {
                this.startCellEdit(cell);
            }
        }
        
        handleHeaderClick(e) {
            const th = e.target.closest('.column-header');
            if (!th || !this.options.sortable) return;
            
            // Don't sort when clicking on filter inputs
            if (e.target.classList.contains('column-filter-input')) return;
            
            const columnKey = th.dataset.columnKey;
            const column = this.options.columns.find(col => col.key === columnKey);
            
            if (column && column.sortable !== false) {
                this.toggleSort(columnKey, e.ctrlKey || e.metaKey);
            }
        }
        
        handleKeyboard(e) {
            switch (e.key) {
                case 'Escape':
                    this.cancelCellEdit();
                    break;
                case 'Enter':
                    if (this.state.editingCell) {
                        this.finishCellEdit();
                    }
                    break;
                case 'Delete':
                    if (this.state.selectedRows.size > 0) {
                        this.deleteSelectedRows();
                    }
                    break;
                case 'a':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        this.selectAll(true);
                    }
                    break;
            }
        }
        
        // Public API methods
        setData(data) {
            this.state.data = [...data];
            this.render();
        }
        
        addRow(rowData) {
            this.state.data.push(rowData);
            this.render();
        }
        
        removeRow(index) {
            this.state.data.splice(index, 1);
            this.state.selectedRows.delete(index);
            this.render();
        }
        
        updateRow(index, rowData) {
            if (index >= 0 && index < this.state.data.length) {
                this.state.data[index] = { ...this.state.data[index], ...rowData };
                this.render();
            }
        }
        
        getSelectedRows() {
            return Array.from(this.state.selectedRows).map(index => ({
                index,
                data: this.state.data[index]
            }));
        }
        
        selectRow(index, selected = true) {
            if (selected) {
                if (!this.options.multiSelect) {
                    this.state.selectedRows.clear();
                }
                this.state.selectedRows.add(index);
            } else {
                this.state.selectedRows.delete(index);
            }
            
            this.updateRowSelectionDisplay();
            this.updateInfo();
            
            if (this.options.onRowSelect) {
                this.options.onRowSelect(this.getSelectedRows());
            }
        }
        
        toggleRowSelection(index, force = null) {
            const isSelected = force !== null ? force : !this.state.selectedRows.has(index);
            this.selectRow(index, isSelected);
        }
        
        selectAll(selected = true) {
            this.state.selectedRows.clear();
            
            if (selected) {
                this.state.sortedData.forEach((_, index) => {
                    this.state.selectedRows.add(index);
                });
            }
            
            this.updateRowSelectionDisplay();
            this.updateInfo();
            
            if (this.options.onRowSelect) {
                this.options.onRowSelect(this.getSelectedRows());
            }
        }
        
        updateRowSelectionDisplay() {
            // Update row checkboxes
            this.elements.tbody.querySelectorAll('.row-checkbox').forEach(checkbox => {
                const row = checkbox.closest('.data-row');
                const rowIndex = parseInt(row.dataset.rowIndex);
                checkbox.checked = this.state.selectedRows.has(rowIndex);
                
                if (checkbox.checked) {
                    row.classList.add('selected');
                } else {
                    row.classList.remove('selected');
                }
            });
            
            // Update select all checkbox
            const selectAllCheckbox = this.elements.thead?.querySelector('.select-all-checkbox');
            if (selectAllCheckbox) {
                const totalRows = this.state.sortedData.length;
                const selectedCount = this.state.selectedRows.size;
                
                selectAllCheckbox.checked = selectedCount > 0 && selectedCount === totalRows;
                selectAllCheckbox.indeterminate = selectedCount > 0 && selectedCount < totalRows;
            }
        }
        
        toggleSort(columnKey, multiSort = false) {
            if (!multiSort && !this.options.multiSort) {
                this.state.sortColumns = [];
            }
            
            const existingSort = this.state.sortColumns.find(s => s.key === columnKey);
            
            if (existingSort) {
                if (existingSort.direction === 'asc') {
                    existingSort.direction = 'desc';
                } else {
                    this.state.sortColumns = this.state.sortColumns.filter(s => s.key !== columnKey);
                }
            } else {
                this.state.sortColumns.push({ key: columnKey, direction: 'asc' });
            }
            
            this.updateSortIndicators();
            this.render();
            
            if (this.options.onSort) {
                this.options.onSort(this.state.sortColumns);
            }
        }
        
        updateSortIndicators() {
            this.elements.thead.querySelectorAll('.sort-indicator').forEach(indicator => {
                const th = indicator.closest('.column-header');
                const columnKey = th.dataset.columnKey;
                const sortColumn = this.state.sortColumns.find(s => s.key === columnKey);
                
                indicator.className = 'sort-indicator';
                
                if (sortColumn) {
                    indicator.classList.add(sortColumn.direction);
                    indicator.textContent = sortColumn.direction === 'asc' ? '▲' : '▼';
                    
                    if (this.state.sortColumns.length > 1) {
                        const index = this.state.sortColumns.indexOf(sortColumn);
                        indicator.textContent += (index + 1);
                    }
                } else {
                    indicator.textContent = '';
                }
            });
        }
        
        setGlobalFilter(value) {
            this.state.globalFilterValue = value;
            this.render();
            
            if (this.options.onFilter) {
                this.options.onFilter({ global: value, columns: Object.fromEntries(this.state.filters) });
            }
        }
        
        clearGlobalFilter() {
            this.elements.globalFilter.value = '';
            this.setGlobalFilter('');
        }
        
        setColumnFilter(columnKey, value) {
            if (value) {
                this.state.filters.set(columnKey, value);
            } else {
                this.state.filters.delete(columnKey);
            }
            
            this.render();
            
            if (this.options.onFilter) {
                this.options.onFilter({ 
                    global: this.state.globalFilterValue, 
                    columns: Object.fromEntries(this.state.filters) 
                });
            }
        }
        
        exportData(format) {
            const data = this.state.sortedData;
            const filename = `table_export_${new Date().toISOString().slice(0, 10)}`;
            
            switch (format) {
                case 'csv':
                    this.exportCSV(data, filename);
                    break;
                case 'json':
                    this.exportJSON(data, filename);
                    break;
                case 'excel':
                    this.exportExcel(data, filename);
                    break;
            }
            
            if (this.options.onExport) {
                this.options.onExport(format, data);
            }
        }
        
        exportCSV(data, filename) {
            const headers = this.options.columns.map(col => col.title || col.key);
            const rows = data.map(row => 
                this.options.columns.map(col => {
                    const value = this.getCellValue(row, col);
                    return this.escapeCSVValue(value);
                })
            );
            
            const csvContent = [headers, ...rows]
                .map(row => row.join(','))
                .join('\n');
            
            this.downloadFile(csvContent, `${filename}.csv`, 'text/csv');
        }
        
        exportJSON(data, filename) {
            const jsonContent = JSON.stringify(data, null, 2);
            this.downloadFile(jsonContent, `${filename}.json`, 'application/json');
        }
        
        exportExcel(data, filename) {
            // Simple Excel export using HTML table format
            const headers = this.options.columns.map(col => col.title || col.key);
            const rows = data.map(row => 
                this.options.columns.map(col => this.getCellValue(row, col))
            );
            
            const excelContent = `
                <table>
                    <tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>
                    ${rows.map(row => 
                        `<tr>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>`
                    ).join('')}
                </table>
            `;
            
            this.downloadFile(excelContent, `${filename}.xls`, 'application/vnd.ms-excel');
        }
        
        escapeCSVValue(value) {
            const stringValue = String(value || '');
            if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n')) {
                return `"${stringValue.replace(/"/g, '""')}"`;
            }
            return stringValue;
        }
        
        downloadFile(content, filename, mimeType) {
            const blob = new Blob([content], { type: mimeType });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
        }
        
        // Cell editing methods
        isColumnEditable(column) {
            return this.options.editable && (
                this.options.editableColumns.length === 0 || 
                this.options.editableColumns.includes(column.key)
            );
        }
        
        startCellEdit(cell) {
            if (this.state.editingCell) {
                this.finishCellEdit();
            }
            
            this.state.editingCell = {
                element: cell,
                originalValue: cell.textContent,
                rowIndex: parseInt(cell.dataset.rowIndex),
                columnKey: cell.dataset.columnKey
            };
            
            cell.setAttribute('contenteditable', 'true');
            cell.classList.add('editing');
            cell.focus();
            
            // Select all text
            const selection = window.getSelection();
            const range = document.createRange();
            range.selectNodeContents(cell);
            selection.removeAllRanges();
            selection.addRange(range);
        }
        
        finishCellEdit() {
            if (!this.state.editingCell) return;
            
            const { element, originalValue, rowIndex, columnKey } = this.state.editingCell;
            const newValue = element.textContent.trim();
            
            element.setAttribute('contenteditable', 'false');
            element.classList.remove('editing');
            
            if (newValue !== originalValue) {
                // Update data
                const column = this.options.columns.find(col => col.key === columnKey);
                const rowData = this.state.data[rowIndex];
                
                if (column && rowData) {
                    rowData[columnKey] = newValue;
                    
                    if (this.options.onCellEdit) {
                        this.options.onCellEdit(rowIndex, columnKey, newValue, originalValue);
                    }
                }
            }
            
            this.state.editingCell = null;
        }
        
        cancelCellEdit() {
            if (!this.state.editingCell) return;
            
            const { element, originalValue } = this.state.editingCell;
            
            element.textContent = originalValue;
            element.setAttribute('contenteditable', 'false');
            element.classList.remove('editing');
            
            this.state.editingCell = null;
        }
        
        // Utility methods
        updateInfo() {
            const totalRows = this.state.data.length;
            const filteredRows = this.state.sortedData.length;
            const selectedRows = this.state.selectedRows.size;
            
            this.elements.rowCount.textContent = 
                filteredRows < totalRows ? 
                `${filteredRows} of ${totalRows} rows` : 
                `${totalRows} rows`;
            
            if (selectedRows > 0) {
                this.elements.selectionInfo.style.display = '';
                this.elements.selectionInfo.textContent = `${selectedRows} selected`;
            } else {
                this.elements.selectionInfo.style.display = 'none';
            }
        }
        
        getRowData(index) {
            return this.state.data[index];
        }
        
        setLoading(loading) {
            this.state.loading = loading;
            this.elements.loadingOverlay.style.display = loading ? 'flex' : 'none';
        }
        
        refresh() {
            this.setLoading(true);
            
            // Simulate refresh delay
            setTimeout(() => {
                this.render();
                this.setLoading(false);
                this.elements.statusText.textContent = 'Data refreshed';
                
                setTimeout(() => {
                    this.elements.statusText.textContent = 'Ready';
                }, 2000);
            }, 500);
        }
        
        toggleExportMenu() {
            this.elements.exportMenu?.classList.toggle('visible');
        }
        
        closeExportMenu() {
            this.elements.exportMenu?.classList.remove('visible');
        }
        
        showSettings() {
            // Placeholder for settings modal
            console.log('Settings modal would open here');
        }
        
        deleteSelectedRows() {
            if (this.state.selectedRows.size === 0) return;
            
            const indicesToRemove = Array.from(this.state.selectedRows).sort((a, b) => b - a);
            indicesToRemove.forEach(index => {
                this.state.data.splice(index, 1);
            });
            
            this.state.selectedRows.clear();
            this.render();
        }
        
        destroy() {
            // Clean up event listeners and resources
            this.container.innerHTML = '';
            this.observers.clear();
        }
    }
    
    // Add stylesheet for table components
    const tableStyles = document.createElement('style');
    tableStyles.id = 'advanced-table-styles';
    tableStyles.textContent = `
        .advanced-data-table {
            background: var(--bg-secondary, #ffffff);
            border: 1px solid var(--border, #e0e0e0);
            border-radius: 8px;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        }
        
        .table-toolbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            background: var(--bg-tertiary, #f8f9fa);
            border-bottom: 1px solid var(--border, #e0e0e0);
        }
        
        .global-filter {
            position: relative;
            display: flex;
            align-items: center;
        }
        
        .filter-input {
            padding: 6px 12px;
            border: 1px solid var(--border, #ccc);
            border-radius: 4px;
            width: 200px;
        }
        
        .clear-filter {
            position: absolute;
            right: 8px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 16px;
            color: var(--text-secondary, #666);
        }
        
        .export-dropdown {
            position: relative;
        }
        
        .export-menu {
            position: absolute;
            top: 100%;
            right: 0;
            background: white;
            border: 1px solid var(--border, #ccc);
            border-radius: 4px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            z-index: 1000;
            display: none;
        }
        
        .export-menu.visible {
            display: block;
        }
        
        .export-format {
            display: block;
            width: 100%;
            padding: 8px 16px;
            border: none;
            background: none;
            text-align: left;
            cursor: pointer;
        }
        
        .export-format:hover {
            background: var(--bg-tertiary, #f8f9fa);
        }
        
        .table-container {
            position: relative;
            overflow: hidden;
        }
        
        .table-header {
            overflow: hidden;
            border-bottom: 1px solid var(--border, #e0e0e0);
        }
        
        .table-body {
            overflow: auto;
            position: relative;
        }
        
        .data-table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }
        
        .column-header {
            position: relative;
            padding: 12px;
            background: var(--bg-tertiary, #f8f9fa);
            font-weight: 600;
            text-align: left;
            border-right: 1px solid var(--border, #e0e0e0);
            user-select: none;
        }
        
        .column-header.sortable {
            cursor: pointer;
        }
        
        .column-header:hover {
            background: var(--bg-secondary, #f0f0f0);
        }
        
        .sort-indicator {
            margin-left: 8px;
            font-size: 12px;
            color: var(--accent, #007bff);
        }
        
        .column-filter {
            margin-top: 4px;
        }
        
        .column-filter-input {
            width: 100%;
            padding: 4px 8px;
            border: 1px solid var(--border, #ccc);
            border-radius: 3px;
            font-size: 12px;
        }
        
        .data-cell {
            padding: 12px;
            border-right: 1px solid var(--border, #e0e0e0);
            border-bottom: 1px solid var(--border, #e0e0e0);
        }
        
        .data-cell.editable {
            cursor: text;
        }
        
        .data-cell.editing {
            background: var(--accent, #007bff);
            color: white;
            outline: 2px solid var(--accent, #007bff);
        }
        
        .data-row:hover {
            background: var(--bg-tertiary, #f8f9fa);
        }
        
        .data-row.selected {
            background: rgba(0, 123, 255, 0.1);
        }
        
        .data-row.striped {
            background: var(--bg-tertiary, #f9f9f9);
        }
        
        .select-column {
            width: 40px;
            text-align: center;
        }
        
        .null-value {
            color: var(--text-secondary, #999);
            font-style: italic;
        }
        
        .table-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            background: var(--bg-tertiary, #f8f9fa);
            border-top: 1px solid var(--border, #e0e0e0);
        }
        
        .loading-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.9);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        
        .loading-spinner {
            width: 32px;
            height: 32px;
            border: 3px solid var(--border, #ccc);
            border-top: 3px solid var(--accent, #007bff);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    `;
    
    if (!document.getElementById('advanced-table-styles')) {
        document.head.appendChild(tableStyles);
    }
    
    // Export to global scope
    global.AdvancedDataTable = AdvancedDataTable;
    
    console.log('✨ Advanced Data Table ready. Access via window.AdvancedDataTable');
    
})(window);