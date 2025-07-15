import { 
  patients, beds, vitalSigns, labResults, predictions, activities,
  type Patient, type InsertPatient, 
  type Bed, type InsertBed,
  type VitalSigns, type InsertVitalSigns,
  type LabResults, type InsertLabResults,
  type Prediction, type InsertPrediction,
  type Activity, type InsertActivity
} from "@shared/schema";

export interface IStorage {
  // Patients
  getPatients(): Promise<Patient[]>;
  getPatient(id: string): Promise<Patient | undefined>;
  createPatient(patient: InsertPatient): Promise<Patient>;
  updatePatient(id: string, patient: Partial<InsertPatient>): Promise<Patient>;
  
  // Beds
  getBeds(): Promise<Bed[]>;
  getBed(id: number): Promise<Bed | undefined>;
  createBed(bed: InsertBed): Promise<Bed>;
  updateBed(id: number, bed: Partial<InsertBed>): Promise<Bed>;
  
  // Vital Signs
  getVitalSignsForPatient(patientId: string): Promise<VitalSigns[]>;
  createVitalSigns(vitalSigns: InsertVitalSigns): Promise<VitalSigns>;
  
  // Lab Results
  getLabResultsForPatient(patientId: string): Promise<LabResults[]>;
  createLabResults(labResults: InsertLabResults): Promise<LabResults>;
  
  // Predictions
  getPredictions(): Promise<Prediction[]>;
  getPredictionsForPatient(patientId: string): Promise<Prediction[]>;
  createPrediction(prediction: InsertPrediction): Promise<Prediction>;
  
  // Activities
  getRecentActivities(limit?: number): Promise<Activity[]>;
  createActivity(activity: InsertActivity): Promise<Activity>;
}

export class MemStorage implements IStorage {
  private patients: Map<string, Patient>;
  private beds: Map<number, Bed>;
  private vitalSigns: Map<number, VitalSigns>;
  private labResults: Map<number, LabResults>;
  private predictions: Map<number, Prediction>;
  private activities: Map<number, Activity>;
  private currentPatientId: number;
  private currentBedId: number;
  private currentVitalSignsId: number;
  private currentLabResultsId: number;
  private currentPredictionId: number;
  private currentActivityId: number;

  constructor() {
    this.patients = new Map();
    this.beds = new Map();
    this.vitalSigns = new Map();
    this.labResults = new Map();
    this.predictions = new Map();
    this.activities = new Map();
    this.currentPatientId = 1;
    this.currentBedId = 1;
    this.currentVitalSignsId = 1;
    this.currentLabResultsId = 1;
    this.currentPredictionId = 1;
    this.currentActivityId = 1;
    
    this.seedData();
  }

  private seedData() {
    // Seed some initial data
    const samplePatients: InsertPatient[] = [
      {
        patientId: "P1847",
        name: "Maria Santos",
        age: 65,
        gender: "Female",
        admissionDate: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
        roomNumber: "204",
        bedId: 1,
        diagnosis: "Pneumonia",
        comorbidities: "Diabetes, Hypertension",
        status: "active",
        riskScore: 87
      },
      {
        patientId: "P1823",
        name: "John Wilson",
        age: 42,
        gender: "Male",
        admissionDate: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
        roomNumber: "156",
        bedId: 2,
        diagnosis: "Post-surgical recovery",
        comorbidities: "None",
        status: "active",
        riskScore: 64
      },
      {
        patientId: "P1901",
        name: "Emily Chen",
        age: 28,
        gender: "Female",
        admissionDate: new Date(Date.now() - 12 * 60 * 60 * 1000),
        roomNumber: "189",
        bedId: 3,
        diagnosis: "Appendectomy",
        comorbidities: "None",
        status: "active",
        riskScore: 23
      }
    ];

    samplePatients.forEach(patient => {
      this.createPatient(patient);
    });

    // Seed beds
    const bedTypes = ["ICU", "General", "Emergency"];
    for (let i = 1; i <= 50; i++) {
      const type = bedTypes[Math.floor(Math.random() * bedTypes.length)];
      const isOccupied = i <= 3;
      this.createBed({
        bedNumber: `B${i.toString().padStart(3, '0')}`,
        roomNumber: `${100 + i}`,
        type,
        status: isOccupied ? "occupied" : "available",
        patientId: isOccupied ? `P${1820 + i}` : undefined
      });
    }

    // Seed activities
    const activities: InsertActivity[] = [
      {
        type: "admission",
        description: "New patient admitted to Room 204",
        patientId: "P1847",
        timestamp: new Date(Date.now() - 2 * 60 * 1000),
        severity: "info"
      },
      {
        type: "alert",
        description: "AI detected high risk in Patient #P1847",
        patientId: "P1847",
        timestamp: new Date(Date.now() - 5 * 60 * 1000),
        severity: "warning"
      },
      {
        type: "discharge",
        description: "Patient discharged from ICU to General Ward",
        patientId: "P1823",
        timestamp: new Date(Date.now() - 12 * 60 * 1000),
        severity: "info"
      }
    ];

    activities.forEach(activity => {
      this.createActivity(activity);
    });
  }

  // Patients
  async getPatients(): Promise<Patient[]> {
    return Array.from(this.patients.values());
  }

  async getPatient(patientId: string): Promise<Patient | undefined> {
    return this.patients.get(patientId);
  }

  async createPatient(insertPatient: InsertPatient): Promise<Patient> {
    const id = this.currentPatientId++;
    const patient: Patient = { 
      ...insertPatient, 
      id,
      status: insertPatient.status || "active",
      dischargeDate: insertPatient.dischargeDate || null,
      roomNumber: insertPatient.roomNumber || null,
      bedId: insertPatient.bedId || null,
      diagnosis: insertPatient.diagnosis || null,
      comorbidities: insertPatient.comorbidities || null,
      riskScore: insertPatient.riskScore || null
    };
    this.patients.set(insertPatient.patientId, patient);
    return patient;
  }

  async updatePatient(patientId: string, updateData: Partial<InsertPatient>): Promise<Patient> {
    const existing = this.patients.get(patientId);
    if (!existing) {
      throw new Error("Patient not found");
    }
    const updated = { ...existing, ...updateData };
    this.patients.set(patientId, updated);
    return updated;
  }

  // Beds
  async getBeds(): Promise<Bed[]> {
    return Array.from(this.beds.values());
  }

  async getBed(id: number): Promise<Bed | undefined> {
    return this.beds.get(id);
  }

  async createBed(insertBed: InsertBed): Promise<Bed> {
    const id = this.currentBedId++;
    const bed: Bed = { 
      ...insertBed, 
      id,
      status: insertBed.status || "available",
      patientId: insertBed.patientId || null
    };
    this.beds.set(id, bed);
    return bed;
  }

  async updateBed(id: number, updateData: Partial<InsertBed>): Promise<Bed> {
    const existing = this.beds.get(id);
    if (!existing) {
      throw new Error("Bed not found");
    }
    const updated = { ...existing, ...updateData };
    this.beds.set(id, updated);
    return updated;
  }

  // Vital Signs
  async getVitalSignsForPatient(patientId: string): Promise<VitalSigns[]> {
    return Array.from(this.vitalSigns.values())
      .filter(vs => vs.patientId === patientId)
      .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
  }

  async createVitalSigns(insertVitalSigns: InsertVitalSigns): Promise<VitalSigns> {
    const id = this.currentVitalSignsId++;
    const vitalSigns: VitalSigns = { 
      ...insertVitalSigns, 
      id,
      heartRate: insertVitalSigns.heartRate || null,
      bloodPressureSystolic: insertVitalSigns.bloodPressureSystolic || null,
      bloodPressureDiastolic: insertVitalSigns.bloodPressureDiastolic || null,
      temperature: insertVitalSigns.temperature || null,
      respiratoryRate: insertVitalSigns.respiratoryRate || null,
      oxygenSaturation: insertVitalSigns.oxygenSaturation || null
    };
    this.vitalSigns.set(id, vitalSigns);
    return vitalSigns;
  }

  // Lab Results
  async getLabResultsForPatient(patientId: string): Promise<LabResults[]> {
    return Array.from(this.labResults.values())
      .filter(lr => lr.patientId === patientId)
      .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
  }

  async createLabResults(insertLabResults: InsertLabResults): Promise<LabResults> {
    const id = this.currentLabResultsId++;
    const labResults: LabResults = { 
      ...insertLabResults, 
      id,
      glucoseLevel: insertLabResults.glucoseLevel || null,
      whiteBloodCellCount: insertLabResults.whiteBloodCellCount || null,
      creatinine: insertLabResults.creatinine || null
    };
    this.labResults.set(id, labResults);
    return labResults;
  }

  // Predictions
  async getPredictions(): Promise<Prediction[]> {
    return Array.from(this.predictions.values())
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }

  async getPredictionsForPatient(patientId: string): Promise<Prediction[]> {
    return Array.from(this.predictions.values())
      .filter(p => p.patientId === patientId)
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }

  async createPrediction(insertPrediction: InsertPrediction): Promise<Prediction> {
    const id = this.currentPredictionId++;
    const prediction: Prediction = { 
      ...insertPrediction, 
      id,
      confidence: insertPrediction.confidence || null,
      prediction: insertPrediction.prediction || null
    };
    this.predictions.set(id, prediction);
    return prediction;
  }

  // Activities
  async getRecentActivities(limit: number = 10): Promise<Activity[]> {
    return Array.from(this.activities.values())
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, limit);
  }

  async createActivity(insertActivity: InsertActivity): Promise<Activity> {
    const id = this.currentActivityId++;
    const activity: Activity = { 
      ...insertActivity, 
      id,
      patientId: insertActivity.patientId || null,
      severity: insertActivity.severity || null
    };
    this.activities.set(id, activity);
    return activity;
  }
}

export const storage = new MemStorage();
