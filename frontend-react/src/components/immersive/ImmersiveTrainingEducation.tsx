import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { VRButton, XR, Controllers, Hands } from '@react-three/xr';
import { Text, Html, Line, Sphere, Box, Plane } from '@react-three/drei';
import * as THREE from 'three';
import { useDispatch, useSelector } from 'react-redux';
import { 
  TrainingModule, 
  EducationalContent, 
  LearningProgress,
  InteractiveLesson,
  AssessmentResult
} from '../../types/immersive';
import { 
  updateTrainingProgress,
  completeLesson,
  addAssessmentResult,
  updateEducationalContent
} from '../../store/slices/immersiveSlice';

interface ImmersiveTrainingEducationProps {
  trainingModules?: TrainingModule[];
  currentModule?: string;
  userLevel?: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  enableAssessments?: boolean;
  enableCollaboration?: boolean;
  onProgressUpdate?: (progress: LearningProgress) => void;
}

interface TrainingScenario {
  id: string;
  title: string;
  description: string;
  difficulty: number;
  duration: number;
  objectives: string[];
  geologicalConcepts: string[];
  interactiveElements: InteractiveElement[];
  assessment?: Assessment;
}

interface InteractiveElement {
  id: string;
  type: 'rock_sample' | 'geological_structure' | 'tool' | 'measurement' | 'quiz';
  position: THREE.Vector3;
  data: any;
  interactions: string[];
  feedback: string[];
}

interface Assessment {
  id: string;
  type: 'practical' | 'theoretical' | 'identification' | 'measurement';
  questions: AssessmentQuestion[];
  passingScore: number;
  timeLimit?: number;
}

interface AssessmentQuestion {
  id: string;
  type: 'multiple_choice' | 'identification' | 'measurement' | 'sequence';
  question: string;
  options?: string[];
  correctAnswer: any;
  explanation: string;
  points: number;
}

const ImmersiveTrainingEducation: React.FC<ImmersiveTrainingEducationProps> = ({
  trainingModules = [],
  currentModule,
  userLevel = 'beginner',
  enableAssessments = true,
  enableCollaboration = false,
  onProgressUpdate
}) => {
  const dispatch = useDispatch();
  const trainingState = useSelector((state: any) => state.immersive.trainingProgress);
  const [activeScenario, setActiveScenario] = useState<TrainingScenario | null>(null);
  const [currentLesson, setCurrentLesson] = useState<InteractiveLesson | null>(null);
  const [learningProgress, setLearningProgress] = useState<LearningProgress>({
    userId: 'current_user',
    completedLessons: [],
    currentModule: currentModule || '',
    skillLevel: userLevel,
    totalScore: 0,
    assessmentResults: [],
    timeSpent: 0,
    achievements: []
  });
  const [isAssessmentActive, setIsAssessmentActive] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState<AssessmentQuestion | null>(null);
  const [userAnswers, setUserAnswers] = useState<Map<string, any>>(new Map());
  const [selectedElement, setSelectedElement] = useState<InteractiveElement | null>(null);

  // Predefined training scenarios for geological education
  const trainingScenarios: TrainingScenario[] = [
    {
      id: 'rock_identification',
      title: 'Rock Identification Basics',
      description: 'Learn to identify common rock types using visual and physical properties',
      difficulty: 1,
      duration: 15,
      objectives: [
        'Identify igneous, sedimentary, and metamorphic rocks',
        'Understand rock formation processes',
        'Use physical properties for identification'
      ],
      geologicalConcepts: ['rock_cycle', 'mineral_composition', 'texture', 'color'],
      interactiveElements: [
        {
          id: 'granite_sample',
          type: 'rock_sample',
          position: new THREE.Vector3(-1, 0, 0),
          data: {
            name: 'Granite',
            type: 'igneous',
            properties: {
              color: 'light_colored',
              texture: 'coarse_grained',
              hardness: 6,
              composition: ['quartz', 'feldspar', 'mica']
            }
          },
          interactions: ['examine', 'measure_hardness', 'identify_minerals'],
          feedback: [
            'Notice the coarse-grained texture indicating slow cooling',
            'The light color suggests felsic composition',
            'Quartz crystals are glassy and hard'
          ]
        },
        {
          id: 'limestone_sample',
          type: 'rock_sample',
          position: new THREE.Vector3(0, 0, 0),
          data: {
            name: 'Limestone',
            type: 'sedimentary',
            properties: {
              color: 'light_gray',
              texture: 'fine_grained',
              hardness: 3,
              composition: ['calcite'],
              reaction: 'fizzes_with_acid'
            }
          },
          interactions: ['examine', 'acid_test', 'scratch_test'],
          feedback: [
            'Limestone fizzes when acid is applied',
            'Soft enough to scratch with a coin',
            'Often contains fossil fragments'
          ]
        },
        {
          id: 'slate_sample',
          type: 'rock_sample',
          position: new THREE.Vector3(1, 0, 0),
          data: {
            name: 'Slate',
            type: 'metamorphic',
            properties: {
              color: 'dark_gray',
              texture: 'fine_grained',
              hardness: 4,
              foliation: 'present',
              composition: ['clay_minerals', 'mica']
            }
          },
          interactions: ['examine', 'test_foliation', 'measure_hardness'],
          feedback: [
            'Notice the flat, parallel surfaces (foliation)',
            'Formed from shale under pressure and heat',
            'Splits easily along foliation planes'
          ]
        }
      ],
      assessment: {
        id: 'rock_id_quiz',
        type: 'identification',
        questions: [
          {
            id: 'q1',
            type: 'identification',
            question: 'Identify this rock sample based on its properties',
            correctAnswer: 'granite',
            explanation: 'Granite is identified by its coarse-grained texture and light color',
            points: 10
          }
        ],
        passingScore: 70,
        timeLimit: 300
      }
    },
    {
      id: 'geological_structures',
      title: 'Understanding Geological Structures',
      description: 'Explore folds, faults, and other geological structures in 3D',
      difficulty: 2,
      duration: 20,
      objectives: [
        'Identify different types of folds and faults',
        'Understand stress and deformation',
        'Interpret geological cross-sections'
      ],
      geologicalConcepts: ['stress', 'strain', 'deformation', 'structural_geology'],
      interactiveElements: [
        {
          id: 'anticline_fold',
          type: 'geological_structure',
          position: new THREE.Vector3(-2, 0, 0),
          data: {
            name: 'Anticline',
            type: 'fold',
            properties: {
              shape: 'upward_arch',
              stress_type: 'compression',
              oldest_rocks: 'center'
            }
          },
          interactions: ['examine_layers', 'measure_dip', 'trace_fold_axis'],
          feedback: [
            'Anticlines arch upward with oldest rocks in the center',
            'Formed by compressive stress',
            'Often associated with oil and gas traps'
          ]
        }
      ]
    },
    {
      id: 'field_techniques',
      title: 'Field Geology Techniques',
      description: 'Master essential field geology skills and measurements',
      difficulty: 3,
      duration: 25,
      objectives: [
        'Use geological compass and clinometer',
        'Measure strike and dip',
        'Create field sketches and notes'
      ],
      geologicalConcepts: ['strike_dip', 'field_mapping', 'structural_measurements'],
      interactiveElements: [
        {
          id: 'compass_tool',
          type: 'tool',
          position: new THREE.Vector3(0, 1, -0.5),
          data: {
            name: 'Geological Compass',
            type: 'measurement_tool',
            functions: ['measure_strike', 'measure_dip', 'take_bearing']
          },
          interactions: ['pick_up', 'align_with_bedding', 'read_measurement'],
          feedback: [
            'Align the compass edge with the bedding plane',
            'Level the compass for accurate readings',
            'Record both strike and dip measurements'
          ]
        }
      ]
    }
  ];

  useEffect(() => {
    // Initialize with first scenario based on user level
    const appropriateScenarios = trainingScenarios.filter(scenario => {
      switch (userLevel) {
        case 'beginner': return scenario.difficulty <= 2;
        case 'intermediate': return scenario.difficulty <= 3;
        case 'advanced': return scenario.difficulty <= 4;
        case 'expert': return true;
        default: return scenario.difficulty <= 2;
      }
    });
    
    if (appropriateScenarios.length > 0) {
      setActiveScenario(appropriateScenarios[0]);
    }
  }, [userLevel]);

  const startLesson = useCallback((scenario: TrainingScenario) => {
    const lesson: InteractiveLesson = {
      id: scenario.id,
      title: scenario.title,
      content: scenario.description,
      objectives: scenario.objectives,
      interactiveElements: scenario.interactiveElements,
      duration: scenario.duration,
      completed: false
    };
    
    setCurrentLesson(lesson);
    setActiveScenario(scenario);
    
    dispatch(updateEducationalContent({
      currentLesson: lesson.id,
      startTime: Date.now()
    }));
  }, [dispatch]);

  const completeCurrentLesson = useCallback(() => {
    if (currentLesson) {
      const completedLesson = { ...currentLesson, completed: true };
      
      setLearningProgress(prev => ({
        ...prev,
        completedLessons: [...prev.completedLessons, completedLesson.id],
        timeSpent: prev.timeSpent + completedLesson.duration
      }));
      
      dispatch(completeLesson({
        lessonId: completedLesson.id,
        completionTime: Date.now(),
        score: calculateLessonScore()
      }));
      
      onProgressUpdate?.(learningProgress);
      
      // Start assessment if enabled
      if (enableAssessments && activeScenario?.assessment) {
        startAssessment(activeScenario.assessment);
      }
    }
  }, [currentLesson, activeScenario, learningProgress, enableAssessments, dispatch, onProgressUpdate]);

  const calculateLessonScore = (): number => {
    // Calculate score based on interactions and time
    const baseScore = 80;
    const interactionBonus = selectedElement ? 10 : 0;
    const timeBonus = currentLesson && currentLesson.duration > 0 ? 
      Math.max(0, 10 - (Date.now() - (trainingState.startTime || 0)) / 60000) : 0;
    
    return Math.min(100, baseScore + interactionBonus + timeBonus);
  };

  const startAssessment = useCallback((assessment: Assessment) => {
    setIsAssessmentActive(true);
    setCurrentQuestion(assessment.questions[0]);
    setUserAnswers(new Map());
  }, []);

  const submitAnswer = useCallback((questionId: string, answer: any) => {
    setUserAnswers(prev => new Map(prev.set(questionId, answer)));
    
    // Move to next question or complete assessment
    if (activeScenario?.assessment) {
      const questions = activeScenario.assessment.questions;
      const currentIndex = questions.findIndex(q => q.id === questionId);
      
      if (currentIndex < questions.length - 1) {
        setCurrentQuestion(questions[currentIndex + 1]);
      } else {
        completeAssessment();
      }
    }
  }, [activeScenario]);

  const completeAssessment = useCallback(() => {
    if (activeScenario?.assessment) {
      const assessment = activeScenario.assessment;
      let totalScore = 0;
      let maxScore = 0;
      
      assessment.questions.forEach(question => {
        maxScore += question.points;
        const userAnswer = userAnswers.get(question.id);
        
        if (userAnswer === question.correctAnswer) {
          totalScore += question.points;
        }
      });
      
      const percentage = (totalScore / maxScore) * 100;
      const passed = percentage >= assessment.passingScore;
      
      const result: AssessmentResult = {
        id: `assessment_${Date.now()}`,
        assessmentId: assessment.id,
        userId: learningProgress.userId,
        score: percentage,
        passed,
        answers: Array.from(userAnswers.entries()),
        completedAt: new Date(),
        timeSpent: 0 // Would track actual time
      };
      
      setLearningProgress(prev => ({
        ...prev,
        assessmentResults: [...prev.assessmentResults, result],
        totalScore: prev.totalScore + percentage
      }));
      
      dispatch(addAssessmentResult(result));
      setIsAssessmentActive(false);
      setCurrentQuestion(null);
    }
  }, [activeScenario, userAnswers, learningProgress, dispatch]);

  const handleElementInteraction = useCallback((element: InteractiveElement, interactionType: string) => {
    setSelectedElement(element);
    
    // Provide feedback based on interaction
    const feedback = element.feedback[Math.floor(Math.random() * element.feedback.length)];
    
    // Update learning progress
    setLearningProgress(prev => ({
      ...prev,
      timeSpent: prev.timeSpent + 1
    }));
    
    console.log(`Interaction: ${interactionType} with ${element.data.name}`);
    console.log(`Feedback: ${feedback}`);
  }, []);

  return (
    <div className="immersive-training-education">
      <div className="training-controls">
        <h3>Geological Training & Education</h3>
        <div className="progress-info">
          <p>Current Level: {userLevel}</p>
          <p>Completed Lessons: {learningProgress.completedLessons.length}</p>
          <p>Total Score: {learningProgress.totalScore.toFixed(1)}</p>
          <p>Time Spent: {Math.floor(learningProgress.timeSpent / 60)}h {learningProgress.timeSpent % 60}m</p>
        </div>
        
        <div className="scenario-selection">
          <h4>Available Training Scenarios</h4>
          {trainingScenarios.map(scenario => (
            <button
              key={scenario.id}
              onClick={() => startLesson(scenario)}
              disabled={scenario.difficulty > getDifficultyLimit(userLevel)}
              className={activeScenario?.id === scenario.id ? 'active' : ''}
            >
              {scenario.title} (Level {scenario.difficulty})
            </button>
          ))}
        </div>
        
        {currentLesson && (
          <div className="current-lesson">
            <h4>{currentLesson.title}</h4>
            <p>{currentLesson.content}</p>
            <div className="objectives">
              <strong>Objectives:</strong>
              <ul>
                {currentLesson.objectives.map((objective, index) => (
                  <li key={index}>{objective}</li>
                ))}
              </ul>
            </div>
            <button onClick={completeCurrentLesson}>
              Complete Lesson
            </button>
          </div>
        )}
      </div>

      <VRButton />

      <Canvas style={{ height: '100vh' }}>
        <XR>
          <ambientLight intensity={0.6} />
          <pointLight position={[10, 10, 10]} />
          <directionalLight position={[5, 5, 5]} intensity={0.5} />
          
          <Controllers />
          <Hands />
          
          <TrainingScene
            scenario={activeScenario}
            selectedElement={selectedElement}
            onElementInteraction={handleElementInteraction}
            userLevel={userLevel}
          />
          
          <TrainingInterface
            currentLesson={currentLesson}
            learningProgress={learningProgress}
            selectedElement={selectedElement}
          />
          
          {isAssessmentActive && currentQuestion && (
            <AssessmentInterface
              question={currentQuestion}
              onSubmitAnswer={submitAnswer}
            />
          )}
        </XR>
      </Canvas>
    </div>
  );
};

function getDifficultyLimit(userLevel: string): number {
  switch (userLevel) {
    case 'beginner': return 2;
    case 'intermediate': return 3;
    case 'advanced': return 4;
    case 'expert': return 5;
    default: return 2;
  }
}

interface TrainingSceneProps {
  scenario: TrainingScenario | null;
  selectedElement: InteractiveElement | null;
  onElementInteraction: (element: InteractiveElement, interactionType: string) => void;
  userLevel: string;
}

const TrainingScene: React.FC<TrainingSceneProps> = ({
  scenario,
  selectedElement,
  onElementInteraction,
  userLevel
}) => {
  if (!scenario) return null;

  return (
    <>
      {/* Render interactive elements */}
      {scenario.interactiveElements.map(element => (
        <InteractiveElementComponent
          key={element.id}
          element={element}
          isSelected={selectedElement?.id === element.id}
          onInteraction={onElementInteraction}
          userLevel={userLevel}
        />
      ))}
      
      {/* Environment setup */}
      <EnvironmentSetup scenario={scenario} />
      
      {/* Learning aids */}
      <LearningAids scenario={scenario} />
    </>
  );
};

const InteractiveElementComponent: React.FC<{
  element: InteractiveElement;
  isSelected: boolean;
  onInteraction: (element: InteractiveElement, interactionType: string) => void;
  userLevel: string;
}> = ({ element, isSelected, onInteraction, userLevel }) => {
  const meshRef = useRef<THREE.Group>(null);
  const [hovered, setHovered] = useState(false);

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.scale.setScalar(isSelected ? 1.2 : hovered ? 1.1 : 1.0);
    }
  });

  const getElementGeometry = () => {
    switch (element.type) {
      case 'rock_sample':
        return <Box args={[0.3, 0.3, 0.3]} />;
      case 'geological_structure':
        return <Plane args={[1, 2]} />;
      case 'tool':
        return <Cylinder args={[0.05, 0.05, 0.3, 8]} />;
      case 'measurement':
        return <Sphere args={[0.1, 8, 8]} />;
      default:
        return <Box args={[0.2, 0.2, 0.2]} />;
    }
  };

  const getElementColor = () => {
    if (isSelected) return 0xFFFF00;
    if (hovered) return 0x00FFFF;
    
    switch (element.type) {
      case 'rock_sample': return getRockColor(element.data);
      case 'geological_structure': return 0x8B4513;
      case 'tool': return 0x888888;
      case 'measurement': return 0x00FF00;
      default: return 0xFFFFFF;
    }
  };

  const getRockColor = (data: any): number => {
    switch (data.type) {
      case 'igneous': return 0xFF6B6B;
      case 'sedimentary': return 0xFFE66D;
      case 'metamorphic': return 0x4ECDC4;
      default: return 0x888888;
    }
  };

  return (
    <group
      ref={meshRef}
      position={element.position}
      onPointerEnter={() => setHovered(true)}
      onPointerLeave={() => setHovered(false)}
      onClick={() => onInteraction(element, 'examine')}
    >
      <mesh>
        {getElementGeometry()}
        <meshStandardMaterial
          color={getElementColor()}
          emissive={isSelected ? 0x222222 : 0x000000}
        />
      </mesh>
      
      <Text
        position={[0, 0.4, 0]}
        fontSize={0.05}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {element.data.name}
      </Text>
      
      {hovered && (
        <Text
          position={[0, 0.3, 0]}
          fontSize={0.03}
          color="lightblue"
          anchorX="center"
          anchorY="middle"
        >
          Click to examine
        </Text>
      )}
      
      {isSelected && (
        <InteractionMenu
          element={element}
          onInteraction={onInteraction}
        />
      )}
    </group>
  );
};

const InteractionMenu: React.FC<{
  element: InteractiveElement;
  onInteraction: (element: InteractiveElement, interactionType: string) => void;
}> = ({ element, onInteraction }) => {
  return (
    <group position={[0.5, 0, 0]}>
      {element.interactions.map((interaction, index) => (
        <group key={interaction} position={[0, -index * 0.15, 0]}>
          <Box
            args={[0.4, 0.1, 0.02]}
            onClick={() => onInteraction(element, interaction)}
          >
            <meshStandardMaterial color={0x444444} />
          </Box>
          
          <Text
            position={[0, 0, 0.02]}
            fontSize={0.03}
            color="white"
            anchorX="center"
            anchorY="middle"
          >
            {interaction.replace('_', ' ')}
          </Text>
        </group>
      ))}
    </group>
  );
};

const EnvironmentSetup: React.FC<{ scenario: TrainingScenario }> = ({ scenario }) => {
  return (
    <>
      {/* Ground plane */}
      <Plane args={[20, 20]} rotation={[-Math.PI / 2, 0, 0]} position={[0, -1, 0]}>
        <meshStandardMaterial color={0x8B7355} />
      </Plane>
      
      {/* Lighting setup for geological visualization */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 10, 5]} intensity={0.6} castShadow />
      
      {/* Scenario-specific environment */}
      {scenario.id === 'field_techniques' && (
        <FieldEnvironment />
      )}
    </>
  );
};

const FieldEnvironment: React.FC = () => {
  return (
    <>
      {/* Outcrop */}
      <Box args={[3, 2, 1]} position={[3, 0, 0]}>
        <meshStandardMaterial color={0x8B4513} />
      </Box>
      
      {/* Vegetation */}
      {Array.from({ length: 10 }, (_, i) => (
        <Cylinder
          key={i}
          args={[0.05, 0.05, 0.5, 8]}
          position={[
            (Math.random() - 0.5) * 10,
            -0.75,
            (Math.random() - 0.5) * 10
          ]}
        >
          <meshStandardMaterial color={0x228B22} />
        </Cylinder>
      ))}
    </>
  );
};

const LearningAids: React.FC<{ scenario: TrainingScenario }> = ({ scenario }) => {
  return (
    <group position={[0, 3, -3]}>
      <Text
        position={[0, 0.5, 0]}
        fontSize={0.1}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {scenario.title}
      </Text>
      
      <Text
        position={[0, 0.2, 0]}
        fontSize={0.06}
        color="lightblue"
        anchorX="center"
        anchorY="middle"
        maxWidth={3}
      >
        {scenario.description}
      </Text>
      
      <group position={[0, -0.2, 0]}>
        <Text
          position={[0, 0.1, 0]}
          fontSize={0.05}
          color="yellow"
          anchorX="center"
          anchorY="middle"
        >
          Key Concepts:
        </Text>
        
        {scenario.geologicalConcepts.slice(0, 3).map((concept, index) => (
          <Text
            key={concept}
            position={[0, -index * 0.08, 0]}
            fontSize={0.04}
            color="lightgreen"
            anchorX="center"
            anchorY="middle"
          >
            • {concept.replace('_', ' ')}
          </Text>
        ))}
      </group>
    </group>
  );
};

const TrainingInterface: React.FC<{
  currentLesson: InteractiveLesson | null;
  learningProgress: LearningProgress;
  selectedElement: InteractiveElement | null;
}> = ({ currentLesson, learningProgress, selectedElement }) => {
  return (
    <group position={[-3, 2, -2]}>
      <Text
        position={[0, 0.5, 0]}
        fontSize={0.1}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        Training Progress
      </Text>
      
      <Text
        position={[0, 0.2, 0]}
        fontSize={0.06}
        color="lightblue"
        anchorX="center"
        anchorY="middle"
      >
        Level: {learningProgress.skillLevel}
      </Text>
      
      <Text
        position={[0, 0, 0]}
        fontSize={0.06}
        color="lightgreen"
        anchorX="center"
        anchorY="middle"
      >
        Completed: {learningProgress.completedLessons.length}
      </Text>
      
      <Text
        position={[0, -0.2, 0]}
        fontSize={0.06}
        color="yellow"
        anchorX="center"
        anchorY="middle"
      >
        Score: {learningProgress.totalScore.toFixed(1)}
      </Text>
      
      {selectedElement && (
        <group position={[0, -0.6, 0]}>
          <Text
            position={[0, 0.1, 0]}
            fontSize={0.05}
            color="cyan"
            anchorX="center"
            anchorY="middle"
          >
            Selected: {selectedElement.data.name}
          </Text>
          
          <Text
            position={[0, -0.1, 0]}
            fontSize={0.04}
            color="white"
            anchorX="center"
            anchorY="middle"
          >
            Type: {selectedElement.type.replace('_', ' ')}
          </Text>
        </group>
      )}
    </group>
  );
};

const AssessmentInterface: React.FC<{
  question: AssessmentQuestion;
  onSubmitAnswer: (questionId: string, answer: any) => void;
}> = ({ question, onSubmitAnswer }) => {
  const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null);

  return (
    <group position={[3, 2, -2]}>
      <Text
        position={[0, 0.5, 0]}
        fontSize={0.08}
        color="yellow"
        anchorX="center"
        anchorY="middle"
      >
        Assessment
      </Text>
      
      <Text
        position={[0, 0.2, 0]}
        fontSize={0.05}
        color="white"
        anchorX="center"
        anchorY="middle"
        maxWidth={2}
      >
        {question.question}
      </Text>
      
      {question.options && (
        <group position={[0, -0.2, 0]}>
          {question.options.map((option, index) => (
            <group key={option} position={[0, -index * 0.15, 0]}>
              <Box
                args={[1.5, 0.1, 0.02]}
                onClick={() => {
                  setSelectedAnswer(option);
                  onSubmitAnswer(question.id, option);
                }}
              >
                <meshStandardMaterial
                  color={selectedAnswer === option ? 0x00FF00 : 0x444444}
                />
              </Box>
              
              <Text
                position={[0, 0, 0.02]}
                fontSize={0.04}
                color="white"
                anchorX="center"
                anchorY="middle"
              >
                {option}
              </Text>
            </group>
          ))}
        </group>
      )}
    </group>
  );
};

export default ImmersiveTrainingEducation;