# Prometheus AI Platform - Frontend

Modern React/TypeScript frontend for the Prometheus AI Platform, built with advanced geological visualization capabilities and comprehensive state management.

## 🚀 Features

### Core Technologies
- **React 18+** with TypeScript for type-safe development
- **Vite** for lightning-fast development and optimized builds
- **Material-UI (MUI)** with custom geological design system
- **Redux Toolkit** with RTK Query for advanced state management
- **React Router v6** with intelligent code splitting
- **i18next** for comprehensive internationalization

### Advanced Capabilities
- **Real-time WebSocket integration** for live geological data
- **Progressive Web App (PWA)** with offline capabilities
- **Advanced error boundaries** with geological domain-aware recovery
- **Comprehensive accessibility** (WCAG 2.1 AA compliant)
- **Multi-language support** with geological terminology
- **Advanced caching strategies** for geological datasets
- **Performance optimization** with lazy loading and code splitting

### Geological Domain Features
- **Geological data types** with comprehensive TypeScript interfaces
- **Scientific visualization** color schemes and projections
- **Real-time monitoring** for seismic, volcanic, and deformation data
- **Advanced coordinate systems** and geological projections
- **Domain-specific error handling** and validation
- **Geological unit conversions** and calculations

## 📁 Project Structure

```
frontend-react/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── common/         # Common components (LoadingSpinner, ErrorFallback)
│   │   └── layout/         # Layout components (MainLayout, AuthLayout)
│   ├── pages/              # Page components
│   ├── store/              # Redux store configuration
│   │   ├── api/           # RTK Query API slices
│   │   └── slices/        # Redux slices
│   ├── types/              # TypeScript type definitions
│   │   ├── geological.ts   # Geological data types
│   │   └── api.ts         # API types
│   ├── theme/              # Material-UI theme configuration
│   ├── i18n/               # Internationalization
│   │   └── locales/       # Translation files
│   ├── utils/              # Utility functions and constants
│   ├── hooks/              # Custom React hooks
│   └── test/               # Test utilities and setup
├── .storybook/             # Storybook configuration
├── public/                 # Static assets
└── dist/                   # Build output
```

## 🛠️ Development

### Prerequisites
- Node.js 18.0.0+
- npm 9.0.0+

### Installation
```bash
cd frontend-react
npm install
```

### Development Server
```bash
npm run dev
```
Starts the development server at `http://localhost:3000`

### Building
```bash
npm run build
```
Creates optimized production build in `dist/`

### Testing
```bash
# Run tests once
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with UI
npm run test:ui

# Run tests with coverage
npm run test:coverage
```

### Code Quality
```bash
# Lint code
npm run lint

# Fix linting issues
npm run lint:fix

# Format code
npm run format

# Check formatting
npm run format:check

# Type checking
npm run type-check
```

### Storybook
```bash
# Start Storybook
npm run storybook

# Build Storybook
npm run build-storybook
```

## 🎨 Design System

### Geological Color Schemes
The application includes specialized color schemes for geological data:

- **Seismic**: Magnitude, depth, and frequency visualizations
- **Volcanic**: Temperature, gas composition, and alert levels
- **Deformation**: Displacement, velocity, and strain
- **Geological**: Age, lithology, and structural features
- **Resources**: Prospectivity, grade, and confidence

### Accessibility Features
- WCAG 2.1 AA compliance
- High contrast mode support
- Reduced motion preferences
- Screen reader optimization
- Keyboard navigation
- Focus management

### Responsive Design
- Mobile-first approach
- Tablet optimization
- Desktop enhancement
- Touch-friendly interfaces
- Adaptive layouts

## 🌐 Internationalization

Supported languages:
- English (en) - Complete
- Spanish (es) - Partial
- French (fr) - Partial
- German (de) - Partial
- Japanese (ja) - Partial

### Adding New Languages
1. Create translation file in `src/i18n/locales/[lang].json`
2. Add language to `src/i18n/index.ts`
3. Update language selector component

## 📊 State Management

### Redux Store Structure
- **auth**: Authentication and user management
- **ui**: UI preferences and layout state
- **geological**: Geological data and analysis state
- **visualization**: 3D/4D visualization settings
- **monitoring**: Real-time monitoring and alerts
- **dashboard**: Dashboard configuration
- **notifications**: Global notification system

### RTK Query APIs
- **Seismic Analysis**: Event detection and waveform analysis
- **Volcanic Monitoring**: Real-time volcanic data and forecasting
- **Deformation Analysis**: InSAR and GPS data processing
- **Resource Assessment**: Prospectivity mapping and economics
- **Model Management**: ML model lifecycle and predictions

## 🔧 Configuration

### Environment Variables
```bash
VITE_API_BASE_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000/ws
VITE_MAPBOX_TOKEN=your_mapbox_token
```

### Build Configuration
- **Vite**: Modern build tool with HMR
- **TypeScript**: Strict type checking
- **ESLint**: Code quality and geological domain rules
- **Prettier**: Code formatting
- **Vitest**: Fast unit testing

## 🚀 Deployment

### Production Build
```bash
npm run build
npm run preview
```

### Docker
```bash
docker build -t prometheus-ai-frontend .
docker run -p 3000:3000 prometheus-ai-frontend
```

### PWA Features
- Service worker for offline functionality
- App manifest for installation
- Background sync for data updates
- Push notifications for alerts

## 🧪 Testing Strategy

### Unit Tests
- Component testing with React Testing Library
- Hook testing with custom utilities
- Store testing with Redux mock store
- Utility function testing

### Integration Tests
- API integration with MSW
- User workflow testing
- Cross-browser compatibility
- Accessibility testing

### Visual Testing
- Storybook for component documentation
- Visual regression testing
- Responsive design testing
- Theme testing (light/dark/high-contrast)

## 📈 Performance

### Optimization Features
- Code splitting by route and feature
- Lazy loading of heavy components
- Image optimization and lazy loading
- Bundle analysis and optimization
- Memory leak prevention
- Efficient re-rendering strategies

### Monitoring
- Performance metrics collection
- Error boundary reporting
- User interaction tracking
- Real-time performance monitoring

## 🔒 Security

### Security Features
- Content Security Policy (CSP)
- XSS protection
- CSRF protection
- Secure authentication flow
- Input validation and sanitization
- Secure data transmission

## 📚 Documentation

### Component Documentation
- Storybook stories for all components
- TypeScript interfaces and JSDoc
- Usage examples and best practices
- Accessibility guidelines

### API Documentation
- RTK Query endpoint documentation
- Type definitions for all APIs
- Error handling examples
- Real-time data flow diagrams

## 🤝 Contributing

### Development Workflow
1. Create feature branch from `main`
2. Implement changes with tests
3. Run quality checks (`npm run lint`, `npm test`)
4. Submit pull request with description
5. Code review and approval
6. Merge to main

### Code Standards
- TypeScript strict mode
- ESLint geological domain rules
- Prettier formatting
- Comprehensive test coverage
- Accessibility compliance
- Performance optimization

## 📄 License

This project is part of the Prometheus AI Platform and follows the same licensing terms.

## 🆘 Support

For technical support and questions:
- Check the documentation in `/docs`
- Review Storybook component examples
- Consult the TypeScript interfaces
- Check the Redux DevTools for state debugging