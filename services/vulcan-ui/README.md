# Vulcan UI

React-based web application framework for the Prometheus AI Geodynamic Platform. This is the user interface component of the Vulcan software suite, providing intuitive access to geological modeling, real-time monitoring, and data visualization capabilities.

## Features

- **Responsive Web Application**: Built with React 18 and TypeScript for modern, type-safe development
- **User Authentication & Authorization**: JWT-based authentication with role-based access control
- **Navigation & Layout**: Collapsible sidebar navigation optimized for geological workflows
- **State Management**: Redux Toolkit for complex geological data visualization state
- **Material-UI Components**: Professional UI components with geological workflow customizations
- **Real-time Data Integration**: WebSocket connections for live monitoring data
- **3D/4D Visualization**: Three.js integration for geological process visualization
- **No-code Workflow Builder**: Drag-and-drop interface for geological analysis workflows

## Technology Stack

- **Frontend Framework**: React 18 with TypeScript
- **State Management**: Redux Toolkit with RTK Query
- **UI Library**: Material-UI (MUI) v5
- **Routing**: React Router v6
- **Build Tool**: Vite
- **Testing**: Vitest + React Testing Library
- **3D Graphics**: Three.js with React Three Fiber
- **Maps**: Leaflet with React Leaflet
- **Charts**: Recharts for data visualization

## Getting Started

### Prerequisites

- Node.js 18.0.0 or higher
- npm or yarn package manager

### Installation

1. Clone the repository and navigate to the Vulcan UI directory:
```bash
cd services/vulcan-ui
```

2. Install dependencies:
```bash
npm install
```

3. Copy environment configuration:
```bash
cp .env.example .env
```

4. Update the `.env` file with your configuration:
```bash
# API Configuration
VITE_API_BASE_URL=http://localhost:8080/api
VITE_WS_BASE_URL=ws://localhost:8080/ws

# Feature flags and external service keys
VITE_MAPBOX_ACCESS_TOKEN=your_token_here
```

### Development

Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:3000`.

### Building for Production

Build the application:
```bash
npm run build
```

Preview the production build:
```bash
npm run preview
```

### Testing

Run tests:
```bash
npm test
```

Run tests in watch mode:
```bash
npm run test:watch
```

### Code Quality

Run ESLint:
```bash
npm run lint
```

Type checking:
```bash
npm run type-check
```

## Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── auth/           # Authentication components
│   ├── common/         # Common utility components
│   └── layout/         # Layout components (sidebar, topbar)
├── pages/              # Page components
│   ├── auth/           # Authentication pages
│   ├── monitoring/     # Real-time monitoring pages
│   ├── simulations/    # Geological simulation pages
│   ├── workflows/      # Workflow builder pages
│   ├── visualization/  # Data visualization pages
│   └── data/           # Data management pages
├── store/              # Redux store configuration
│   └── slices/         # Redux slices for different domains
├── services/           # API service functions
├── types/              # TypeScript type definitions
├── utils/              # Utility functions
└── test/               # Test utilities and setup
```

## Key Components

### Authentication System
- JWT-based authentication with automatic token refresh
- Role-based access control for geological workflows
- Protected routes with permission checking

### Layout System
- Responsive sidebar navigation with geological workflow organization
- Collapsible sidebar for different screen sizes
- Top bar with user profile, notifications, and system status

### State Management
- Redux Toolkit for complex geological data state
- Separate slices for auth, UI, geological data, and workflows
- Async thunks for API integration

### Geological Data Types
- Comprehensive TypeScript interfaces for geological entities
- Seismic events, deformation measurements, geochemical samples
- Volcanic risk assessments and model predictions

## API Integration

The application integrates with the Prometheus AI platform backend services:

- **Authentication Service**: User login, registration, and profile management
- **Geological Data Service**: CRUD operations for geological data
- **Workflow Service**: Workflow creation, execution, and management
- **Real-time Monitoring**: WebSocket connections for live data streams

## Docker Deployment

Build the Docker image:
```bash
docker build -t vulcan-ui .
```

Run the container:
```bash
docker run -p 80:80 vulcan-ui
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_BASE_URL` | Backend API base URL | `http://localhost:8080/api` |
| `VITE_WS_BASE_URL` | WebSocket base URL | `ws://localhost:8080/ws` |
| `VITE_ENABLE_3D_VISUALIZATION` | Enable 3D visualization features | `true` |
| `VITE_ENABLE_REAL_TIME_MONITORING` | Enable real-time monitoring | `true` |
| `VITE_MAPBOX_ACCESS_TOKEN` | Mapbox API token for maps | - |

## Contributing

1. Follow the established code style and TypeScript conventions
2. Write tests for new components and features
3. Update documentation for any API changes
4. Ensure all tests pass before submitting changes

## License

This project is part of the Prometheus AI Geodynamic Platform.