/**
 * Immersive Education System for Geological Training and Learning
 * 
 * This system provides comprehensive educational experiences in VR/AR,
 * including guided tours, interactive lessons, assessment tools, and
 * adaptive learning pathways for geological education.
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';

// Educational content interfaces
interface EducationalModule {
  id: string;
  title: string;
  description: string;
