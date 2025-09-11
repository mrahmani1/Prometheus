// Specialized Geological Interface Components
export { VolcanicMonitoringDashboard } from './VolcanicMonitoringDashboard';
export { SeismicAnalysisInterface } from './SeismicAnalysisInterface';
export { MantleDynamics3D } from './MantleDynamics3D';

// Re-export geological types
export type {
  VolcanicState,
  SeismicEvent,
  SeismicEventType,
  MomentTensorSolution,
  SourceMechanism,
  AlertLevel,
  ThermalAnomaly,
  GasComposition,
  CalderaClassification,
  SupereruptionRiskAssessment,
  Point3D
} from '../../types/geological';