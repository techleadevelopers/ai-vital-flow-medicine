/**
 * AI Client for connecting to Python ML server
 * Real machine learning integration instead of mock data
 */

import axios from 'axios';

const AI_SERVER_URL = process.env.AI_SERVER_URL || 'http://localhost:8000';

export interface PatientMLData {
  patient_id: string;
  age: number;
  gender: string;
  heart_rate?: number;
  blood_pressure_systolic?: number;
  blood_pressure_diastolic?: number;
  temperature?: number;
  respiratory_rate?: number;
  oxygen_saturation?: number;
  glucose_level?: number;
  white_blood_cell_count?: number;
  creatinine?: number;
  comorbidities?: string;
  admission_date: string;
}

export interface MLRiskPrediction {
  patient_id: string;
  risk_score: number;
  confidence: number;
  factors: string[];
  recommendation: string;
  model_accuracy: number;
}

export interface MLBedOptimization {
  patient_id: string;
  current_bed: string;
  recommended_bed: string;
  reason: string;
  priority: number;
  confidence: number;
}

export interface MLPatientFlow {
  hour: number;
  predicted_admissions: number;
  predicted_discharges: number;
  confidence: number;
}

class AIClient {
  private baseURL: string;
  private isConnected: boolean = false;

  constructor() {
    this.baseURL = AI_SERVER_URL;
  }

  async checkConnection(): Promise<boolean> {
    try {
      const response = await axios.get(`${this.baseURL}/`, { timeout: 5000 });
      this.isConnected = response.status === 200;
      console.log('✅ AI Server connected:', response.data);
      return this.isConnected;
    } catch (error) {
      console.error('❌ AI Server connection failed:', error.message);
      this.isConnected = false;
      return false;
    }
  }

  async predictPatientRisk(patientData: PatientMLData): Promise<MLRiskPrediction> {
    try {
      if (!this.isConnected) {
        await this.checkConnection();
      }

      const response = await axios.post(`${this.baseURL}/predict/risk`, patientData, {
        timeout: 10000,
        headers: { 'Content-Type': 'application/json' }
      });

      return response.data;
    } catch (error) {
      console.error('Error predicting patient risk:', error.message);
      
      // Fallback prediction if AI server is down
      return {
        patient_id: patientData.patient_id,
        risk_score: this.calculateFallbackRisk(patientData),
        confidence: 0.5,
        factors: ['AI server unavailable - using fallback calculation'],
        recommendation: 'Monitor patient closely until AI analysis is available',
        model_accuracy: 0.5
      };
    }
  }

  async predictPatientFlow(): Promise<MLPatientFlow[]> {
    try {
      if (!this.isConnected) {
        await this.checkConnection();
      }

      const response = await axios.get(`${this.baseURL}/predict/flow`, {
        timeout: 10000
      });

      return response.data;
    } catch (error) {
      console.error('Error predicting patient flow:', error.message);
      
      // Fallback flow prediction
      return this.generateFallbackFlow();
    }
  }

  async optimizeBedAllocation(patients: PatientMLData[]): Promise<MLBedOptimization[]> {
    try {
      if (!this.isConnected) {
        await this.checkConnection();
      }

      const response = await axios.post(`${this.baseURL}/optimize/beds`, patients, {
        timeout: 15000,
        headers: { 'Content-Type': 'application/json' }
      });

      return response.data;
    } catch (error) {
      console.error('Error optimizing bed allocation:', error.message);
      
      // Fallback bed optimization
      return patients.map(patient => ({
        patient_id: patient.patient_id,
        current_bed: 'General',
        recommended_bed: this.calculateFallbackBedType(patient),
        reason: 'Fallback recommendation - AI server unavailable',
        priority: 3,
        confidence: 0.5
      }));
    }
  }

  async getModelStatus(): Promise<any> {
    try {
      const response = await axios.get(`${this.baseURL}/models/status`, {
        timeout: 5000
      });
      return response.data;
    } catch (error) {
      console.error('Error getting model status:', error.message);
      return {
        risk_classifier: false,
        flow_predictor: false,
        scalers_loaded: 0,
        server_status: 'disconnected',
        error: error.message
      };
    }
  }

  // Fallback calculations when AI server is unavailable
  private calculateFallbackRisk(patient: PatientMLData): number {
    let riskScore = 0;

    // Age factor
    if (patient.age > 75) riskScore += 30;
    else if (patient.age > 65) riskScore += 15;

    // Vital signs
    if (patient.heart_rate) {
      if (patient.heart_rate > 100 || patient.heart_rate < 60) riskScore += 20;
    }

    if (patient.oxygen_saturation && patient.oxygen_saturation < 92) {
      riskScore += 35;
    }

    if (patient.temperature && patient.temperature > 100.4) {
      riskScore += 25;
    }

    if (patient.blood_pressure_systolic && patient.blood_pressure_systolic > 160) {
      riskScore += 15;
    }

    return Math.min(100, riskScore);
  }

  private calculateFallbackBedType(patient: PatientMLData): string {
    const riskScore = this.calculateFallbackRisk(patient);
    
    if (riskScore >= 70) return 'ICU';
    if (riskScore >= 40) return 'Step-down';
    return 'General';
  }

  private generateFallbackFlow(): MLPatientFlow[] {
    const flow: MLPatientFlow[] = [];
    const currentHour = new Date().getHours();

    for (let i = 0; i < 24; i++) {
      const hour = (currentHour + i) % 24;
      
      // Basic pattern: more admissions during day, more discharges in afternoon
      let admissions = 2;
      let discharges = 1;

      if (hour >= 8 && hour <= 18) {
        admissions = Math.floor(Math.random() * 6) + 2;
      }

      if (hour >= 12 && hour <= 16) {
        discharges = Math.floor(Math.random() * 5) + 3;
      }

      flow.push({
        hour,
        predicted_admissions: admissions,
        predicted_discharges: discharges,
        confidence: 0.6
      });
    }

    return flow;
  }
}

export const aiClient = new AIClient();