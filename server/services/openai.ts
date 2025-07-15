import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || process.env.OPENAI_API_KEY_ENV_VAR || "default_key"
});

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

    const response = await openai.chat.completions.create({
      model: "gpt-4o", // the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
      messages: [
        {
          role: "system",
          content: "You are an expert healthcare AI assistant specializing in hospital operations and patient care optimization."
        },
        {
          role: "user",
          content: prompt
        }
      ],
      response_format: { type: "json_object" },
    });

    const result = JSON.parse(response.choices[0].message.content || '{"insights": []}');
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

    const response = await openai.chat.completions.create({
      model: "gpt-4o", // the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
      messages: [
        {
          role: "system",
          content: "You are a clinical AI assistant. Provide clear, professional medical summaries."
        },
        {
          role: "user",
          content: prompt
        }
      ],
    });

    return response.choices[0].message.content || "Clinical summary unavailable.";
  } catch (error) {
    console.error("Error summarizing patient condition:", error);
    return "Clinical summary temporarily unavailable. Please check system configuration.";
  }
}
