import { GoogleGenAI } from '@google/genai';

const genai = new GoogleGenAI(process.env.GEMINI_API_KEY || 'default_key');

export interface ClinicalInsight {
  type: 'staffing' | 'trend' | 'risk' | 'optimization';
  title: string;
  content: string;
  priority: 'low' | 'medium' | 'high';
}

export async function generateClinicalInsights(
  patientData: any[],
  bedOccupancy: any,
  riskPredictions: any[]
): Promise<ClinicalInsight[]> {
  try {
    const prompt = `As an AI healthcare assistant, analyze the following hospital data and provide 2-3 clinical insights:

Patient Data Summary:
- Total patients: ${patientData.length}
- High risk patients: ${riskPredictions.filter(p => p.confidence > 0.7).length}
- Average risk score: ${riskPredictions.reduce((sum, p) => sum + (p.confidence || 0), 0) / riskPredictions.length}

Bed Occupancy:
- ICU occupancy: ${bedOccupancy.icuOccupancy}%
- General ward occupancy: ${bedOccupancy.generalOccupancy}%
- Emergency occupancy: ${bedOccupancy.emergencyOccupancy}%

Please provide insights in JSON format with the following structure:
{
  "insights": [
    {
      "type": "staffing|trend|risk|optimization",
      "title": "Brief title",
      "content": "Detailed recommendation or insight",
      "priority": "low|medium|high"
    }
  ]
}`;

    const response = await genai.models.generateContent({
      model: "gemini-2.5-flash",
      config: {
        systemInstruction: "You are an expert healthcare AI assistant specializing in hospital operations and patient care optimization.",
        responseMimeType: "application/json",
        responseSchema: {
          type: "object",
          properties: {
            insights: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  type: { type: "string" },
                  title: { type: "string" },
                  content: { type: "string" },
                  priority: { type: "string" }
                }
              }
            }
          }
        }
      },
      contents: prompt
    });

    const result = JSON.parse(response.text || '{"insights": []}');
    return result.insights || [];
  } catch (error) {
    console.error("Error generating clinical insights:", error);
    return [
      {
        type: 'trend',
        title: 'System Analysis',
        content: 'Clinical insights temporarily unavailable. Please check system configuration.',
        priority: 'medium'
      }
    ];
  }
}

export async function summarizePatientCondition(
  patientData: any,
  vitalSigns: any[],
  labResults: any[]
): Promise<string> {
  try {
    const prompt = `Summarize the clinical condition of this patient based on the following data:

Patient: ${patientData.name}, Age: ${patientData.age}, Gender: ${patientData.gender}
Diagnosis: ${patientData.diagnosis}
Comorbidities: ${patientData.comorbidities}
Current Risk Score: ${patientData.riskScore}%

Recent Vital Signs: ${JSON.stringify(vitalSigns.slice(-5))}
Recent Lab Results: ${JSON.stringify(labResults.slice(-3))}

Provide a concise clinical summary and recommendations in 2-3 sentences.`;

    const response = await genai.models.generateContent({
      model: "gemini-2.5-flash",
      config: {
        systemInstruction: "You are a clinical AI assistant. Provide clear, professional medical summaries."
      },
      contents: prompt
    });

    return response.text || "Clinical summary unavailable.";
  } catch (error) {
    console.error("Error summarizing patient condition:", error);
    return "Clinical summary temporarily unavailable. Please check system configuration.";
  }
}