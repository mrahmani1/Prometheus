/**
 * Dashboard configuration and layout management slice
 */

import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface DashboardState {
  layouts: {
    [layoutId: string]: {
      id: string;
      name: string;
      widgets: Array<{
        id: string;
        type: string;
        position: { x: number; y: number };
        size: { width: number; height: number };
        config: Record<string, any>;
        minimized: boolean;
      }>;
      isDefault: boolean;
    };
  };
  activeLayout: string;
  editMode: boolean;
  gridSize: number;
  snapToGrid: boolean;
}

const initialState: DashboardState = {
  layouts: {
    default: {
      id: 'default',
      name: 'Default Layout',
      widgets: [],
      isDefault: true,
    },
  },
  activeLayout: 'default',
  editMode: false,
  gridSize: 20,
  snapToGrid: true,
};

const dashboardSlice = createSlice({
  name: 'dashboard',
  initialState,
  reducers: {
    setActiveLayout: (state, action: PayloadAction<string>) => {
      state.activeLayout = action.payload;
    },
    setEditMode: (state, action: PayloadAction<boolean>) => {
      state.editMode = action.payload;
    },
    addWidget: (state, action: PayloadAction<DashboardState['layouts'][string]['widgets'][0]>) => {
      const layout = state.layouts[state.activeLayout];
      if (layout) {
        layout.widgets.push(action.payload);
      }
    },
    removeWidget: (state, action: PayloadAction<string>) => {
      const layout = state.layouts[state.activeLayout];
      if (layout) {
        layout.widgets = layout.widgets.filter(w => w.id !== action.payload);
      }
    },
    updateWidget: (state, action: PayloadAction<{ id: string; updates: Partial<DashboardState['layouts'][string]['widgets'][0]> }>) => {
      const layout = state.layouts[state.activeLayout];
      if (layout) {
        const widget = layout.widgets.find(w => w.id === action.payload.id);
        if (widget) {
          Object.assign(widget, action.payload.updates);
        }
      }
    },
  },
});

export const {
  setActiveLayout,
  setEditMode,
  addWidget,
  removeWidget,
  updateWidget,
} = dashboardSlice.actions;

export default dashboardSlice.reducer;