import { pgTable, text, serial, integer, boolean, timestamp, real, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const patients = pgTable("patients", {
  id: serial("id").primaryKey(),
  patientId: text("patient_id").notNull().unique(),
  name: text("name").notNull(),
  age: integer("age").notNull(),
  gender: text("gender").notNull(),
  admissionDate: timestamp("admission_date").notNull(),
  dischargeDate: timestamp("discharge_date"),
  roomNumber: text("room_number"),
  bedId: integer("bed_id"),
  diagnosis: text("diagnosis"),
  comorbidities: text("comorbidities"),
  status: text("status").notNull().default("active"), // active, discharged, transferred
  riskScore: integer("risk_score").default(0), // 0-100
});

export const beds = pgTable("beds", {
  id: serial("id").primaryKey(),
  bedNumber: text("bed_number").notNull().unique(),
  roomNumber: text("room_number").notNull(),
  type: text("type").notNull(), // ICU, General, Emergency
  status: text("status").notNull().default("available"), // occupied, available, maintenance
  patientId: text("patient_id"),
});

export const vitalSigns = pgTable("vital_signs", {
  id: serial("id").primaryKey(),
  patientId: text("patient_id").notNull(),
  timestamp: timestamp("timestamp").notNull(),
  heartRate: integer("heart_rate"),
  bloodPressureSystolic: integer("blood_pressure_systolic"),
  bloodPressureDiastolic: integer("blood_pressure_diastolic"),
  temperature: real("temperature"),
  respiratoryRate: integer("respiratory_rate"),
  oxygenSaturation: integer("oxygen_saturation"),
});

export const labResults = pgTable("lab_results", {
  id: serial("id").primaryKey(),
  patientId: text("patient_id").notNull(),
  timestamp: timestamp("timestamp").notNull(),
  glucoseLevel: real("glucose_level"),
  whiteBloodCellCount: real("white_blood_cell_count"),
  creatinine: real("creatinine"),
});

export const predictions = pgTable("predictions", {
  id: serial("id").primaryKey(),
  patientId: text("patient_id").notNull(),
  predictionType: text("prediction_type").notNull(), // risk_deterioration, length_of_stay
  confidence: real("confidence"),
  prediction: jsonb("prediction"), // flexible JSON for different prediction types
  timestamp: timestamp("timestamp").notNull(),
});

export const activities = pgTable("activities", {
  id: serial("id").primaryKey(),
  type: text("type").notNull(), // admission, discharge, alert, prediction
  description: text("description").notNull(),
  patientId: text("patient_id"),
  timestamp: timestamp("timestamp").notNull(),
  severity: text("severity").default("info"), // info, warning, error
});

// Insert schemas
export const insertPatientSchema = createInsertSchema(patients).omit({
  id: true,
});

export const insertBedSchema = createInsertSchema(beds).omit({
  id: true,
});

export const insertVitalSignsSchema = createInsertSchema(vitalSigns).omit({
  id: true,
});

export const insertLabResultsSchema = createInsertSchema(labResults).omit({
  id: true,
});

export const insertPredictionSchema = createInsertSchema(predictions).omit({
  id: true,
});

export const insertActivitySchema = createInsertSchema(activities).omit({
  id: true,
});

// Types
export type Patient = typeof patients.$inferSelect;
export type InsertPatient = z.infer<typeof insertPatientSchema>;
export type Bed = typeof beds.$inferSelect;
export type InsertBed = z.infer<typeof insertBedSchema>;
export type VitalSigns = typeof vitalSigns.$inferSelect;
export type InsertVitalSigns = z.infer<typeof insertVitalSignsSchema>;
export type LabResults = typeof labResults.$inferSelect;
export type InsertLabResults = z.infer<typeof insertLabResultsSchema>;
export type Prediction = typeof predictions.$inferSelect;
export type InsertPrediction = z.infer<typeof insertPredictionSchema>;
export type Activity = typeof activities.$inferSelect;
export type InsertActivity = z.infer<typeof insertActivitySchema>;
