/**
 * AI Predictions Service - Integration with Python ML Server
 * Real machine learning predictions using scikit-learn and TensorFlow
 */

import { aiClient, PatientMLData, MLRiskPrediction, MLBedOptimization, MLPatientFlow } from './aiClient';
import { storage } from '../storage';

export interface RiskPrediction {
  patientId: string;
  riskScore: number;
  confidence: number;
  factors: string[];
  recommendation: string;
}

export interface BedOptimization {
  patientId: string;
  currentBed: string;
  recommendedBed: string;
  reason: string;
  priority: number;
}

async function convertPatientToMLFormat(patientId: string): Promise<PatientMLData | null> {
  const patient = await storage.getPatient(patientId);
  if (!patient) return null;

  // Get latest vital signs
  const vitalSigns = await storage.getVitalSignsForPatient(patientId);
  const latestVitals = vitalSigns[vitalSigns.length - 1];

  // Get latest lab results
  const labResults = await storage.getLabResultsForPatient(patientId);
  const latestLabs = labResults[labResults.length - 1];

  return {
    patient_id: patientId,
    age: patient.age,
    gender: patient.gender,
    heart_rate: latestVitals?.heartRate || undefined,
    blood_pressure_systolic: latestVitals?.bloodPressureSystolic || undefined,
    blood_pressure_diastolic: latestVitals?.bloodPressureDiastolic || undefined,
    temperature: latestVitals?.temperature || undefined,
    respiratory_rate: latestVitals?.respiratoryRate || undefined,
    oxygen_saturation: latestVitals?.oxygenSaturation || undefined,
    glucose_level: latestLabs?.glucoseLevel || undefined,
    white_blood_cell_count: latestLabs?.whiteBloodCellCount || undefined,
    creatinine: latestLabs?.creatinine || undefined,
    comorbidities: patient.comorbidities || undefined,
    admission_date: patient.admissionDate.toISOString()
  };
}

export async function predictDeteriorationRisk(patientId: string): Promise<RiskPrediction> {
  try {
    console.log(`🤖 Running ML risk prediction for patient ${patientId}`);
    
    const mlPatientData = await convertPatientToMLFormat(patientId);
    if (!mlPatientData) {
      throw new Error(`Patient ${patientId} not found`);
    }

    const mlPrediction = await aiClient.predictPatientRisk(mlPatientData);
    
    console.log(`✅ ML prediction completed: Risk=${mlPrediction.risk_score}, Confidence=${mlPrediction.confidence}`);

    // Store prediction
    await storage.createPrediction({
      patientId,
      predictionType: "risk_deterioration",
      confidence: mlPrediction.confidence,
      prediction: { 
        riskScore: mlPrediction.risk_score, 
        factors: mlPrediction.factors, 
        recommendation: mlPrediction.recommendation,
        modelAccuracy: mlPrediction.model_accuracy
      },
      timestamp: new Date()
    });

    return {
      patientId: mlPrediction.patient_id,
      riskScore: mlPrediction.risk_score,
      confidence: mlPrediction.confidence,
      factors: mlPrediction.factors,
      recommendation: mlPrediction.recommendation
    };
  } catch (error) {
    console.error(`❌ Error in ML risk prediction for ${patientId}:`, error.message);
    
    // Return error state instead of fallback
    return {
      patientId,
      riskScore: 0,
      confidence: 0,
      factors: [`ML Error: ${error.message}`],
      recommendation: "Unable to generate AI prediction - manual assessment required"
    };
  }
}

export async function optimizeBedAllocation(): Promise<BedOptimization[]> {
  try {
    console.log('🤖 Running ML bed optimization');
    
    const patients = await storage.getPatients();
    const mlPatients: PatientMLData[] = [];

    // Convert all patients to ML format
    for (const patient of patients) {
      const mlPatient = await convertPatientToMLFormat(patient.patientId);
      if (mlPatient) {
        mlPatients.push(mlPatient);
      }
    }

    if (mlPatients.length === 0) {
      return [];
    }

    const mlOptimizations = await aiClient.optimizeBedAllocation(mlPatients);
    
    console.log(`✅ ML bed optimization completed for ${mlOptimizations.length} patients`);

    return mlOptimizations.map(opt => ({
      patientId: opt.patient_id,
      currentBed: opt.current_bed,
      recommendedBed: opt.recommended_bed,
      reason: opt.reason,
      priority: opt.priority
    }));
  } catch (error) {
    console.error('❌ Error in ML bed optimization:', error.message);
    return [];
  }
}

export async function generatePatientFlowPrediction(): Promise<{ admissions: number[], discharges: number[] }> {
  try {
    console.log('🤖 Running ML patient flow prediction');
    
    const mlFlowPredictions = await aiClient.predictPatientFlow();
    
    const admissions = mlFlowPredictions.map(pred => pred.predicted_admissions);
    const discharges = mlFlowPredictions.map(pred => pred.predicted_discharges);
    
    console.log(`✅ ML flow prediction completed - 24h forecast generated`);

    return { admissions, discharges };
  } catch (error) {
    console.error('❌ Error in ML flow prediction:', error.message);
    
    // Return empty arrays on error
    return {
      admissions: new Array(24).fill(0),
      discharges: new Array(24).fill(0)
    };
  }
}

// Initialize AI client connection on startup
export async function initializeAIService(): Promise<void> {
  console.log('🚀 Initializing AI Service...');
  
  const connected = await aiClient.checkConnection();
  if (connected) {
    console.log('✅ AI Service ready - Python ML server connected');
    
    // Get model status
    const status = await aiClient.getModelStatus();
    console.log('📊 ML Models Status:', status);
  } else {
    console.warn('⚠️  AI Service starting without ML server - using fallback predictions');
  }
}
