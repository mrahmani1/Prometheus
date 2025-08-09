/**
 * Interactive Geochemical Visualization Suite
 * Advanced visualization tools for geochemical data analysis
 */

class GeochemicalVisualizer {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container ${containerId} not found`);
        }
        
        this.options = {
            width: options.width || 1000,
            height: options.height || 600,
            margin: { top: 40, right: 40, bottom: 60, left: 80 },
            colorScheme: options.colorScheme || 'viridis',
            interactive: options.interactive !== false,
            ...options
        };
        
        this.data = null;
        this.currentPlot = null;
        this.scales = {};
        this.colorScale = null;
        
        this.initializeContainer();
        this.createColorScales();
    }
    
    initializeContainer() {
        // Clear container
        this.container.innerHTML = '';
        
        // Create main SVG
        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', this.options.width)
            .attr('height', this.options.height)
            .attr('class', 'geochem-viz');
        
        // Create plot area
        this.plotArea = this.svg.append('g')
            .attr('transform', `translate(${this.options.margin.left}, ${this.options.margin.top})`);
        
        // Create tooltip
        this.tooltip = d3.select('body').append('div')
            .attr('class', 'geochem-tooltip')
            .style('opacity', 0)
            .style('position', 'absolute')
            .style('background', 'rgba(0, 0, 0, 0.8)')
            .style('color', 'white')
            .style('padding', '10px')
            .style('border-radius', '5px')
            .style('pointer-events', 'none');
        
        // Create control panel
        this.createControlPanel();
    }
    
    createColorScales() {
        const schemes = {
            viridis: d3.scaleSequential(d3.interpolateViridis),
            plasma: d3.scaleSequential(d3.interpolatePlasma),
            magma: d3.scaleSequential(d3.interpolateMagma),
            turbo: d3.scaleSequential(d3.interpolateTurbo),
            categorical: d3.scaleOrdinal(d3.schemeCategory10)
        };
        
        this.colorSchemes = schemes;
        this.colorScale = schemes[this.options.colorScheme];
    }
    
    createControlPanel() {
        const controlDiv = document.createElement('div');
        controlDiv.className = 'geochem-controls';
        controlDiv.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            z-index: 1000;
        `;
        
        this.container.style.position = 'relative';
        this.container.appendChild(controlDiv);
        this.controlPanel = controlDiv;
    }
    
    /**
     * Create Harker variation diagrams
     */
    createHarkerDiagram(data, xElement = 'SiO2', yElements = null) {
        if (!data || data.length === 0) {
            console.error('No data provided for Harker diagram');
            return;
        }
        
        this.data = data;
        this.currentPlot = 'harker';
        
        // Default elements for multi-plot
        if (!yElements) {
            yElements = ['Al2O3', 'Fe2O3', 'MgO', 'CaO', 'Na2O', 'K2O'];
        }
        
        // Clear previous plot
        this.plotArea.selectAll('*').remove();
        
        // Create subplot layout
        const nPlots = yElements.length;
        const plotsPerRow = Math.ceil(Math.sqrt(nPlots));
        const plotsPerCol = Math.ceil(nPlots / plotsPerRow);
        
        const subplotWidth = (this.options.width - this.options.margin.left - this.options.margin.right) / plotsPerRow - 20;
        const subplotHeight = (this.options.height - this.options.margin.top - this.options.margin.bottom) / plotsPerCol - 40;
        
        yElements.forEach((yElement, index) => {
            const row = Math.floor(index / plotsPerRow);
            const col = index % plotsPerRow;
            
            const x = col * (subplotWidth + 20);
            const y = row * (subplotHeight + 40);
            
            this.createSubplot(data, xElement, yElement, x, y, subplotWidth, subplotHeight);
        });
        
        // Add title
        this.svg.append('text')
            .attr('x', this.options.width / 2)
            .attr('y', 25)
            .attr('text-anchor', 'middle')
            .style('font-size', '16px')
            .style('font-weight', 'bold')
            .text(`Harker Diagrams - ${xElement} vs Major Elements`);
    }
    
    createSubplot(data, xElement, yElement, x, y, width, height) {
        const subplot = this.plotArea.append('g')
            .attr('transform', `translate(${x}, ${y})`);
        
        // Extract data for this subplot
        const plotData = data.filter(d => d[xElement] !== undefined && d[yElement] !== undefined)
                            .map(d => ({
                                x: +d[xElement],
                                y: +d[yElement],
                                sample: d.sample_id || d.id,
                                data: d
                            }));
        
        if (plotData.length === 0) return;
        
        // Create scales
        const xScale = d3.scaleLinear()
            .domain(d3.extent(plotData, d => d.x))
            .range([0, width]);
        
        const yScale = d3.scaleLinear()
            .domain(d3.extent(plotData, d => d.y))
            .range([height, 0]);
        
        // Create axes
        const xAxis = d3.axisBottom(xScale);
        const yAxis = d3.axisLeft(yScale);
        
        subplot.append('g')
            .attr('transform', `translate(0, ${height})`)
            .call(xAxis);
        
        subplot.append('g')
            .call(yAxis);
        
        // Add axis labels
        subplot.append('text')
            .attr('x', width / 2)
            .attr('y', height + 35)
            .attr('text-anchor', 'middle')
            .style('font-size', '12px')
            .text(`${xElement} (wt%)`);
        
        subplot.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('x', -height / 2)
            .attr('y', -35)
            .attr('text-anchor', 'middle')
            .style('font-size', '12px')
            .text(`${yElement} (wt%)`);
        
        // Add data points
        subplot.selectAll('.data-point')
            .data(plotData)
            .enter()
            .append('circle')
            .attr('class', 'data-point')
            .attr('cx', d => xScale(d.x))
            .attr('cy', d => yScale(d.y))
            .attr('r', 3)
            .style('fill', this.colorScale.range ? this.colorScale.range()[0] : this.colorScale(0.5))
            .style('stroke', '#333')
            .style('stroke-width', 1)
            .on('mouseover', (event, d) => this.showTooltip(event, d, xElement, yElement))
            .on('mouseout', () => this.hideTooltip());
        
        // Add trend line if requested
        if (plotData.length > 2) {
            const regression = this.calculateLinearRegression(plotData);
            if (regression) {
                const lineData = [
                    { x: d3.min(plotData, d => d.x), y: regression.slope * d3.min(plotData, d => d.x) + regression.intercept },
                    { x: d3.max(plotData, d => d.x), y: regression.slope * d3.max(plotData, d => d.x) + regression.intercept }
                ];
                
                const line = d3.line()
                    .x(d => xScale(d.x))
                    .y(d => yScale(d.y));
                
                subplot.append('path')
                    .datum(lineData)
                    .attr('d', line)
                    .style('stroke', '#ff6b6b')
                    .style('stroke-width', 2)
                    .style('stroke-dasharray', '5,5')
                    .style('fill', 'none');
            }
        }
    }
    
    /**
     * Create spider/radar diagrams for multi-element comparison
     */
    createSpiderDiagram(data, elements = null, normalizationValues = null) {
        if (!data || data.length === 0) {
            console.error('No data provided for spider diagram');
            return;
        }
        
        this.data = data;
        this.currentPlot = 'spider';
        
        // Default elements (REE + trace elements)
        if (!elements) {
            elements = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'];
        }
        
        // Default normalization (chondrite values)
        if (!normalizationValues) {
            normalizationValues = {
                'La': 0.31, 'Ce': 0.808, 'Pr': 0.122, 'Nd': 0.6, 'Sm': 0.195, 'Eu': 0.0735,
                'Gd': 0.259, 'Tb': 0.0474, 'Dy': 0.322, 'Ho': 0.0718, 'Er': 0.21, 'Tm': 0.0324,
                'Yb': 0.209, 'Lu': 0.0322
            };
        }
        
        // Clear previous plot
        this.plotArea.selectAll('*').remove();
        
        const centerX = (this.options.width - this.options.margin.left - this.options.margin.right) / 2;
        const centerY = (this.options.height - this.options.margin.top - this.options.margin.bottom) / 2;
        const radius = Math.min(centerX, centerY) - 50;
        
        // Prepare normalized data
        const normalizedData = data.map(sample => {
            const normalized = { sample_id: sample.sample_id || sample.id };
            elements.forEach(element => {
                const value = sample[element];
                const normValue = normalizationValues[element];
                if (value !== undefined && normValue) {
                    normalized[element] = value / normValue;
                }
            });
            return normalized;
        });
        
        // Create logarithmic scale
        const allValues = normalizedData.flatMap(sample => 
            elements.map(element => sample[element]).filter(v => v !== undefined && v > 0)
        );
        
        const minValue = d3.min(allValues);
        const maxValue = d3.max(allValues);
        
        const logScale = d3.scaleLog()
            .domain([Math.max(minValue, 0.01), maxValue])
            .range([0, radius]);
        
        // Create radial axes
        elements.forEach((element, i) => {
            const angle = (i * 2 * Math.PI) / elements.length - Math.PI / 2;
            const x1 = centerX;
            const y1 = centerY;
            const x2 = centerX + radius * Math.cos(angle);
            const y2 = centerY + radius * Math.sin(angle);
            
            // Axis line
            this.plotArea.append('line')
                .attr('x1', x1)
                .attr('y1', y1)
                .attr('x2', x2)
                .attr('y2', y2)
                .style('stroke', '#ddd')
                .style('stroke-width', 1);
            
            // Element label
            const labelX = centerX + (radius + 20) * Math.cos(angle);
            const labelY = centerY + (radius + 20) * Math.sin(angle);
            
            this.plotArea.append('text')
                .attr('x', labelX)
                .attr('y', labelY)
                .attr('text-anchor', 'middle')
                .attr('dominant-baseline', 'middle')
                .style('font-size', '12px')
                .text(element);
        });
        
        // Create concentric circles for scale
        const logTicks = [0.1, 1, 10, 100];
        logTicks.forEach(tick => {
            if (tick >= minValue && tick <= maxValue) {
                const r = logScale(tick);
                
                this.plotArea.append('circle')
                    .attr('cx', centerX)
                    .attr('cy', centerY)
                    .attr('r', r)
                    .style('fill', 'none')
                    .style('stroke', '#eee')
                    .style('stroke-width', 1);
                
                this.plotArea.append('text')
                    .attr('x', centerX + r)
                    .attr('y', centerY - 5)
                    .style('font-size', '10px')
                    .style('fill', '#666')
                    .text(tick);
            }
        });
        
        // Plot data lines
        const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
        
        normalizedData.forEach((sample, sampleIndex) => {
            const pathData = [];
            
            elements.forEach((element, i) => {
                const value = sample[element];
                if (value !== undefined && value > 0) {
                    const angle = (i * 2 * Math.PI) / elements.length - Math.PI / 2;
                    const r = logScale(value);
                    const x = centerX + r * Math.cos(angle);
                    const y = centerY + r * Math.sin(angle);
                    pathData.push({ x, y, value, element });
                }
            });
            
            if (pathData.length > 1) {
                // Create path
                const line = d3.line()
                    .x(d => d.x)
                    .y(d => d.y)
                    .curve(d3.curveLinearClosed);
                
                this.plotArea.append('path')
                    .datum(pathData)
                    .attr('d', line)
                    .style('fill', 'none')
                    .style('stroke', colorScale(sampleIndex))
                    .style('stroke-width', 2)
                    .style('opacity', 0.8);
                
                // Add data points
                this.plotArea.selectAll(`.point-${sampleIndex}`)
                    .data(pathData)
                    .enter()
                    .append('circle')
                    .attr('class', `point-${sampleIndex}`)
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y)
                    .attr('r', 3)
                    .style('fill', colorScale(sampleIndex))
                    .style('stroke', 'white')
                    .style('stroke-width', 1)
                    .on('mouseover', (event, d) => {
                        this.tooltip
                            .style('opacity', 1)
                            .html(`Sample: ${sample.sample_id}<br>
                                   Element: ${d.element}<br>
                                   Value: ${d.value.toFixed(2)}<br>
                                   Normalized: ${d.value.toFixed(2)}x chondrite`)
                            .style('left', (event.pageX + 10) + 'px')
                            .style('top', (event.pageY - 10) + 'px');
                    })
                    .on('mouseout', () => this.hideTooltip());
            }
        });
        
        // Add legend
        this.createLegend(normalizedData.map((d, i) => ({
            label: d.sample_id,
            color: colorScale(i)
        })));
        
        // Add title
        this.svg.append('text')
            .attr('x', this.options.width / 2)
            .attr('y', 25)
            .attr('text-anchor', 'middle')
            .style('font-size', '16px')
            .style('font-weight', 'bold')
            .text('REE Spider Diagram (Chondrite Normalized)');
    }
    
    /**
     * Create classification diagrams (TAS, AFM, etc.)
     */
    createTASDiagram(data) {
        if (!data || data.length === 0) {
            console.error('No data provided for TAS diagram');
            return;
        }
        
        this.data = data;
        this.currentPlot = 'tas';
        
        // Clear previous plot
        this.plotArea.selectAll('*').remove();
        
        const plotWidth = this.options.width - this.options.margin.left - this.options.margin.right;
        const plotHeight = this.options.height - this.options.margin.top - this.options.margin.bottom;
        
        // Create scales
        const xScale = d3.scaleLinear()
            .domain([35, 85])
            .range([0, plotWidth]);
        
        const yScale = d3.scaleLinear()
            .domain([0, 20])
            .range([plotHeight, 0]);
        
        // Create axes
        const xAxis = d3.axisBottom(xScale);
        const yAxis = d3.axisLeft(yScale);
        
        this.plotArea.append('g')
            .attr('transform', `translate(0, ${plotHeight})`)
            .call(xAxis);
        
        this.plotArea.append('g')
            .call(yAxis);
        
        // Add axis labels
        this.plotArea.append('text')
            .attr('x', plotWidth / 2)
            .attr('y', plotHeight + 40)
            .attr('text-anchor', 'middle')
            .style('font-size', '14px')
            .text('SiO₂ (wt%)');
        
        this.plotArea.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('x', -plotHeight / 2)
            .attr('y', -50)
            .attr('text-anchor', 'middle')
            .style('font-size', '14px')
            .text('Na₂O + K₂O (wt%)');
        
        // Add TAS classification fields
        this.addTASFields(xScale, yScale);
        
        // Plot data points
        const plotData = data.filter(d => d.SiO2 !== undefined && d.Na2O !== undefined && d.K2O !== undefined)
                            .map(d => ({
                                x: +d.SiO2,
                                y: (+d.Na2O || 0) + (+d.K2O || 0),
                                sample: d.sample_id || d.id,
                                data: d
                            }));
        
        this.plotArea.selectAll('.data-point')
            .data(plotData)
            .enter()
            .append('circle')
            .attr('class', 'data-point')
            .attr('cx', d => xScale(d.x))
            .attr('cy', d => yScale(d.y))
            .attr('r', 4)
            .style('fill', '#4682b4')
            .style('stroke', '#333')
            .style('stroke-width', 1)
            .style('opacity', 0.8)
            .on('mouseover', (event, d) => {
                this.tooltip
                    .style('opacity', 1)
                    .html(`Sample: ${d.sample}<br>
                           SiO₂: ${d.x.toFixed(1)}%<br>
                           Na₂O+K₂O: ${d.y.toFixed(1)}%`)
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 10) + 'px');
            })
            .on('mouseout', () => this.hideTooltip());
        
        // Add title
        this.svg.append('text')
            .attr('x', this.options.width / 2)
            .attr('y', 25)
            .attr('text-anchor', 'middle')
            .style('font-size', '16px')
            .style('font-weight', 'bold')
            .text('TAS Classification Diagram');
    }
    
    addTASFields(xScale, yScale) {
        // Define TAS field boundaries
        const fields = [
            { name: 'Picro-basalt', path: [[35, 0], [45, 0], [45, 3], [35, 3]] },
            { name: 'Basalt', path: [[45, 0], [52, 0], [52, 5], [45, 3]] },
            { name: 'Basaltic-andesite', path: [[52, 0], [57, 0], [57, 5.9], [52, 5]] },
            { name: 'Andesite', path: [[57, 0], [63, 0], [63, 7], [57, 5.9]] },
            { name: 'Dacite', path: [[63, 0], [77, 0], [77, 8.8], [69, 8], [63, 7]] },
            { name: 'Rhyolite', path: [[77, 0], [85, 0], [85, 20], [77, 8.8]] }
        ];
        
        fields.forEach(field => {
            const pathString = field.path.map((point, index) => {
                const x = xScale(point[0]);
                const y = yScale(point[1]);
                return (index === 0 ? 'M' : 'L') + x + ',' + y;
            }).join('') + 'Z';
            
            this.plotArea.append('path')
                .attr('d', pathString)
                .style('fill', 'none')
                .style('stroke', '#666')
                .style('stroke-width', 1)
                .style('stroke-dasharray', '3,3');
            
            // Add field labels
            const centroidX = field.path.reduce((sum, p) => sum + p[0], 0) / field.path.length;
            const centroidY = field.path.reduce((sum, p) => sum + p[1], 0) / field.path.length;
            
            this.plotArea.append('text')
                .attr('x', xScale(centroidX))
                .attr('y', yScale(centroidY))
                .attr('text-anchor', 'middle')
                .style('font-size', '10px')
                .style('fill', '#666')
                .style('font-style', 'italic')
                .text(field.name);
        });
    }
    
    /**
     * Create ternary diagrams
     */
    createTernaryDiagram(data, elements = ['SiO2', 'Al2O3', 'FeO']) {
        if (!data || data.length === 0) {
            console.error('No data provided for ternary diagram');
            return;
        }
        
        this.data = data;
        this.currentPlot = 'ternary';
        
        // Clear previous plot
        this.plotArea.selectAll('*').remove();
        
        const plotWidth = this.options.width - this.options.margin.left - this.options.margin.right;
        const plotHeight = this.options.height - this.options.margin.top - this.options.margin.bottom;
        
        const size = Math.min(plotWidth, plotHeight);
        const triangleHeight = (Math.sqrt(3) / 2) * size * 0.8;
        const centerX = plotWidth / 2;
        const centerY = plotHeight / 2;
        
        // Triangle vertices
        const vertices = {
            top: { x: centerX, y: centerY - triangleHeight / 2 },
            left: { x: centerX - size * 0.4, y: centerY + triangleHeight / 2 },
            right: { x: centerX + size * 0.4, y: centerY + triangleHeight / 2 }
        };
        
        // Draw triangle
        const trianglePath = `M${vertices.top.x},${vertices.top.y} L${vertices.left.x},${vertices.left.y} L${vertices.right.x},${vertices.right.y}Z`;
        
        this.plotArea.append('path')
            .attr('d', trianglePath)
            .style('fill', 'none')
            .style('stroke', '#333')
            .style('stroke-width', 2);
        
        // Add grid lines
        this.addTernaryGrid(vertices);
        
        // Add labels
        this.plotArea.append('text')
            .attr('x', vertices.top.x)
            .attr('y', vertices.top.y - 20)
            .attr('text-anchor', 'middle')
            .style('font-size', '14px')
            .style('font-weight', 'bold')
            .text(elements[0]);
        
        this.plotArea.append('text')
            .attr('x', vertices.left.x - 20)
            .attr('y', vertices.left.y + 20)
            .attr('text-anchor', 'middle')
            .style('font-size', '14px')
            .style('font-weight', 'bold')
            .text(elements[1]);
        
        this.plotArea.append('text')
            .attr('x', vertices.right.x + 20)
            .attr('y', vertices.right.y + 20)
            .attr('text-anchor', 'middle')
            .style('font-size', '14px')
            .style('font-weight', 'bold')
            .text(elements[2]);
        
        // Plot data
        const plotData = data.filter(d => 
            elements.every(elem => d[elem] !== undefined)
        ).map(d => {
            const values = elements.map(elem => +d[elem] || 0);
            const sum = values.reduce((a, b) => a + b, 0);
            const normalized = sum > 0 ? values.map(v => v / sum) : [1/3, 1/3, 1/3];
            
            const coords = this.ternaryToCartesian(normalized, vertices);
            
            return {
                x: coords.x,
                y: coords.y,
                sample: d.sample_id || d.id,
                values: normalized,
                data: d
            };
        });
        
        this.plotArea.selectAll('.ternary-point')
            .data(plotData)
            .enter()
            .append('circle')
            .attr('class', 'ternary-point')
            .attr('cx', d => d.x)
            .attr('cy', d => d.y)
            .attr('r', 4)
            .style('fill', '#e74c3c')
            .style('stroke', '#333')
            .style('stroke-width', 1)
            .style('opacity', 0.8)
            .on('mouseover', (event, d) => {
                this.tooltip
                    .style('opacity', 1)
                    .html(`Sample: ${d.sample}<br>
                           ${elements[0]}: ${(d.values[0] * 100).toFixed(1)}%<br>
                           ${elements[1]}: ${(d.values[1] * 100).toFixed(1)}%<br>
                           ${elements[2]}: ${(d.values[2] * 100).toFixed(1)}%`)
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 10) + 'px');
            })
            .on('mouseout', () => this.hideTooltip());
        
        // Add title
        this.svg.append('text')
            .attr('x', this.options.width / 2)
            .attr('y', 25)
            .attr('text-anchor', 'middle')
            .style('font-size', '16px')
            .style('font-weight', 'bold')
            .text(`Ternary Diagram: ${elements.join(' - ')}`);
    }
    
    addTernaryGrid(vertices) {
        // Add grid lines at 10% intervals
        for (let i = 1; i < 10; i++) {
            const fraction = i / 10;
            
            // Horizontal lines (constant top component)
            const leftPoint = {
                x: vertices.left.x + fraction * (vertices.right.x - vertices.left.x),
                y: vertices.left.y
            };
            const rightPoint = {
                x: vertices.top.x + (1 - fraction) * (vertices.right.x - vertices.top.x) / 2,
                y: vertices.top.y + (1 - fraction) * (vertices.left.y - vertices.top.y)
            };
            
            this.plotArea.append('line')
                .attr('x1', leftPoint.x)
                .attr('y1', leftPoint.y)
                .attr('x2', rightPoint.x)
                .attr('y2', rightPoint.y)
                .style('stroke', '#ddd')
                .style('stroke-width', 0.5);
        }
    }
    
    ternaryToCartesian(ternaryCoords, vertices) {
        const [a, b, c] = ternaryCoords;
        
        const x = vertices.right.x * a + vertices.left.x * b + vertices.top.x * c;
        const y = vertices.right.y * a + vertices.left.y * b + vertices.top.y * c;
        
        return { x, y };
    }
    
    /**
     * Create bivariate plots with density estimation
     */
    createBivariatePlot(data, xElement, yElement, options = {}) {
        if (!data || data.length === 0) {
            console.error('No data provided for bivariate plot');
            return;
        }
        
        this.data = data;
        this.currentPlot = 'bivariate';
        
        // Clear previous plot
        this.plotArea.selectAll('*').remove();
        
        const plotWidth = this.options.width - this.options.margin.left - this.options.margin.right;
        const plotHeight = this.options.height - this.options.margin.top - this.options.margin.bottom;
        
        // Prepare data
        const plotData = data.filter(d => d[xElement] !== undefined && d[yElement] !== undefined)
                            .map(d => ({
                                x: +d[xElement],
                                y: +d[yElement],
                                sample: d.sample_id || d.id,
                                data: d
                            }));
        
        if (plotData.length === 0) return;
        
        // Create scales
        const xScale = d3.scaleLinear()
            .domain(d3.extent(plotData, d => d.x))
            .range([0, plotWidth]);
        
        const yScale = d3.scaleLinear()
            .domain(d3.extent(plotData, d => d.y))
            .range([plotHeight, 0]);
        
        // Create axes
        const xAxis = d3.axisBottom(xScale);
        const yAxis = d3.axisLeft(yScale);
        
        this.plotArea.append('g')
            .attr('transform', `translate(0, ${plotHeight})`)
            .call(xAxis);
        
        this.plotArea.append('g')
            .call(yAxis);
        
        // Add axis labels
        this.plotArea.append('text')
            .attr('x', plotWidth / 2)
            .attr('y', plotHeight + 40)
            .attr('text-anchor', 'middle')
            .style('font-size', '14px')
            .text(xElement);
        
        this.plotArea.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('x', -plotHeight / 2)
            .attr('y', -50)
            .attr('text-anchor', 'middle')
            .style('font-size', '14px')
            .text(yElement);
        
        // Add density contours if requested
        if (options.showDensity && plotData.length > 10) {
            this.addDensityContours(plotData, xScale, yScale, plotWidth, plotHeight);
        }
        
        // Plot data points
        this.plotArea.selectAll('.data-point')
            .data(plotData)
            .enter()
            .append('circle')
            .attr('class', 'data-point')
            .attr('cx', d => xScale(d.x))
            .attr('cy', d => yScale(d.y))
            .attr('r', 3)
            .style('fill', options.pointColor || '#3498db')
            .style('stroke', '#333')
            .style('stroke-width', 1)
            .style('opacity', 0.7)
            .on('mouseover', (event, d) => this.showTooltip(event, d, xElement, yElement))
            .on('mouseout', () => this.hideTooltip());
        
        // Add correlation line
        if (options.showCorrelation) {
            const regression = this.calculateLinearRegression(plotData);
            if (regression) {
                this.addRegressionLine(regression, xScale, yScale, plotData);
            }
        }
    }
    
    addDensityContours(data, xScale, yScale, width, height) {
        // Simple 2D kernel density estimation
        const gridSize = 20;
        const bandwidth = 0.5;
        
        const xStep = width / gridSize;
        const yStep = height / gridSize;
        
        const densityGrid = [];
        
        for (let i = 0; i < gridSize; i++) {
            densityGrid[i] = [];
            for (let j = 0; j < gridSize; j++) {
                const x = i * xStep;
                const y = j * yStep;
                
                let density = 0;
                data.forEach(d => {
                    const dx = (xScale(d.x) - x) / width;
                    const dy = (yScale(d.y) - y) / height;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    density += Math.exp(-distance * distance / (2 * bandwidth * bandwidth));
                });
                
                densityGrid[i][j] = density;
            }
        }
        
        // Create contour paths (simplified)
        const contourLevels = [0.1, 0.3, 0.5, 0.7, 0.9];
        const contourColors = ['#f0f0f0', '#d0d0d0', '#b0b0b0', '#909090', '#707070'];
        
        // This would typically use a proper contouring algorithm
        // For now, just show the concept with simple threshold areas
    }
    
    addRegressionLine(regression, xScale, yScale, data) {
        const lineData = [
            { x: d3.min(data, d => d.x), y: regression.slope * d3.min(data, d => d.x) + regression.intercept },
            { x: d3.max(data, d => d.x), y: regression.slope * d3.max(data, d => d.x) + regression.intercept }
        ];
        
        const line = d3.line()
            .x(d => xScale(d.x))
            .y(d => yScale(d.y));
        
        this.plotArea.append('path')
            .datum(lineData)
            .attr('d', line)
            .style('stroke', '#e74c3c')
            .style('stroke-width', 2)
            .style('stroke-dasharray', '5,5')
            .style('fill', 'none');
        
        // Add regression statistics
        this.plotArea.append('text')
            .attr('x', 10)
            .attr('y', 20)
            .style('font-size', '12px')
            .text(`R² = ${regression.r2.toFixed(3)}`);
    }
    
    // Utility functions
    calculateLinearRegression(data) {
        const n = data.length;
        if (n < 2) return null;
        
        const sumX = data.reduce((sum, d) => sum + d.x, 0);
        const sumY = data.reduce((sum, d) => sum + d.y, 0);
        const sumXY = data.reduce((sum, d) => sum + d.x * d.y, 0);
        const sumXX = data.reduce((sum, d) => sum + d.x * d.x, 0);
        const sumYY = data.reduce((sum, d) => sum + d.y * d.y, 0);
        
        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;
        
        // Calculate R²
        const meanY = sumY / n;
        const ssTotal = data.reduce((sum, d) => sum + Math.pow(d.y - meanY, 2), 0);
        const ssRes = data.reduce((sum, d) => sum + Math.pow(d.y - (slope * d.x + intercept), 2), 0);
        const r2 = 1 - (ssRes / ssTotal);
        
        return { slope, intercept, r2 };
    }
    
    showTooltip(event, d, xLabel, yLabel) {
        this.tooltip
            .style('opacity', 1)
            .html(`Sample: ${d.sample}<br>
                   ${xLabel}: ${d.x.toFixed(2)}<br>
                   ${yLabel}: ${d.y.toFixed(2)}`)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px');
    }
    
    hideTooltip() {
        this.tooltip.style('opacity', 0);
    }
    
    createLegend(items) {
        const legend = this.svg.append('g')
            .attr('class', 'legend')
            .attr('transform', `translate(${this.options.width - 150}, 50)`);
        
        items.forEach((item, i) => {
            const legendItem = legend.append('g')
                .attr('transform', `translate(0, ${i * 20})`);
            
            legendItem.append('rect')
                .attr('width', 15)
                .attr('height', 15)
                .style('fill', item.color);
            
            legendItem.append('text')
                .attr('x', 20)
                .attr('y', 12)
                .style('font-size', '12px')
                .text(item.label);
        });
    }
    
    exportPlot(format = 'svg') {
        if (format === 'svg') {
            const svgData = new XMLSerializer().serializeToString(this.svg.node());
            const blob = new Blob([svgData], { type: 'image/svg+xml' });
            const url = URL.createObjectURL(blob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = `geochem_plot_${this.currentPlot}.svg`;
            link.click();
        } else if (format === 'png') {
            // Convert SVG to PNG using canvas
            const svgData = new XMLSerializer().serializeToString(this.svg.node());
            const img = new Image();
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                
                const link = document.createElement('a');
                link.href = canvas.toDataURL('image/png');
                link.download = `geochem_plot_${this.currentPlot}.png`;
                link.click();
            };
            
            img.src = 'data:image/svg+xml;base64,' + btoa(svgData);
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GeochemicalVisualizer;
}

// Example usage
/*
// Initialize visualizer
const viz = new GeochemicalVisualizer('geochem-container', {
    width: 1200,
    height: 800,
    colorScheme: 'viridis'
});

// Sample data
const geochemData = [
    { sample_id: 'S001', SiO2: 65.2, Al2O3: 15.1, Fe2O3: 4.8, MgO: 2.1, CaO: 3.9, Na2O: 3.2, K2O: 2.8 },
    { sample_id: 'S002', SiO2: 72.1, Al2O3: 13.8, Fe2O3: 2.1, MgO: 0.9, CaO: 2.1, Na2O: 3.8, K2O: 4.1 }
];

// Create visualizations
viz.createHarkerDiagram(geochemData);
viz.createTASDiagram(geochemData);
viz.createTernaryDiagram(geochemData, ['SiO2', 'Al2O3', 'FeO']);
*/