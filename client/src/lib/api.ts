import { apiRequest } from "./queryClient";

export interface DashboardStats {
  totalPatients: number;
  highRiskPatients: number;
  availableBeds: number;
  icuBeds: number;
  generalBeds: number;
  aiAccuracy: number;
}

export interface PatientFlowData {
  admissions: number[];
  discharges: number[];
}

export interface RiskPrediction {
  patientId: string;
  riskScore: number;
  confidence: number;
  factors: string[];
  recommendation: string;
}

export interface ClinicalInsight {
  type: 'staffing' | 'trend' | 'risk' | 'optimization';
  title: string;
  content: string;
  priority: 'low' | 'medium' | 'high';
}

export interface Activity {
  id: number;
  type: string;
  description: string;
  patientId?: string;
  timestamp: string;
  severity: string;
}

export const api = {
  // Dashboard
  getDashboardStats: async (): Promise<DashboardStats> => {
    const res = await apiRequest("GET", "/api/dashboard/stats");
    return res.json();
  },

  getPatientFlow: async (): Promise<PatientFlowData> => {
    const res = await apiRequest("GET", "/api/predictions/patient-flow");
    return res.json();
  },

  // Patients
  getPatients: async () => {
    const res = await apiRequest("GET", "/api/patients");
    return res.json();
  },

  getPatient: async (id: string) => {
    const res = await apiRequest("GET", `/api/patients/${id}`);
    return res.json();
  },

  getPatientVitalSigns: async (id: string) => {
    const res = await apiRequest("GET", `/api/patients/${id}/vital-signs`);
    return res.json();
  },

  // Beds
  getBeds: async () => {
    const res = await apiRequest("GET", "/api/beds");
    return res.json();
  },

  // Predictions
  getRiskPredictions: async (): Promise<RiskPrediction[]> => {
    const res = await apiRequest("GET", "/api/predictions/risk");
    return res.json();
  },

  getPatientRiskPrediction: async (patientId: string): Promise<RiskPrediction> => {
    const res = await apiRequest("GET", `/api/predictions/risk/${patientId}`);
    return res.json();
  },

  getBedOptimizations: async () => {
    const res = await apiRequest("GET", "/api/predictions/bed-optimization");
    return res.json();
  },

  // AI Insights
  getClinicalInsights: async (): Promise<ClinicalInsight[]> => {
    const res = await apiRequest("POST", "/api/llm/clinical-insights");
    return res.json();
  },

  getPatientSummary: async (patientId: string): Promise<{ summary: string }> => {
    const res = await apiRequest("POST", `/api/llm/summarize-patient/${patientId}`);
    return res.json();
  },

  // Activities
  getActivities: async (limit?: number): Promise<Activity[]> => {
    const url = limit ? `/api/activities?limit=${limit}` : "/api/activities";
    const res = await apiRequest("GET", url);
    return res.json();
  },
};
