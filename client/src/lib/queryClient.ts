// src/lib/queryClient.ts

import { QueryClient } from "@tanstack/react-query";
// 1. Importe TODOS os seus dados mockados (incluindo os de gráficos)
import * as mock from './mock-data';

/**
 * Função auxiliar para criar uma Resposta HTTP falsa a partir dos dados do mock.
 */
const createMockResponse = (data: any, delay: number = 300): Promise<Response> => {
  return new Promise<Response>(resolve => 
    setTimeout(() => {
      const response = new Response(JSON.stringify(data), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
      resolve(response);
    }, delay)
  );
};

/**
 * Sua função apiRequest, agora com a rota de analytics integrada.
 */
export async function apiRequest(
  method: string,
  url: string,
  data?: unknown | undefined,
): Promise<Response> {
  console.log(`[MOCK API] Rota interceptada: ${method} ${url}`);

  switch (true) {
    // --- ROTAS DO DASHBOARD ---
    case url === "/api/dashboard/stats":
      const stats = { totalPatients: mock.mockPatients.length, highRiskPatients: 1, availableBeds: 15, aiAccuracy: 94.5 };
      return createMockResponse(stats);

    // --- NOVA ROTA PARA ANALYTICS ---
    case url === "/api/analytics/all":
        const analyticsData = {
            // Estes dados devem existir no seu arquivo mock-data.ts
            trends: mock.mockTrendData,
            resources: mock.mockResourceData,
            performance: mock.mockPerformanceData,
        };
        return createMockResponse(analyticsData);

    // --- ROTAS DE PREDIÇÕES (EXISTENTES) ---
    case url === "/api/predictions/patient-flow":
      const flow = { admissions: [5, 8, 12, 10, 7, 9, 11], discharges: [4, 6, 10, 8, 8, 7, 9] };
      return createMockResponse(flow);

    // --- ROTAS DE PACIENTES (EXISTENTES) ---
    case url === "/api/patients":
      return createMockResponse(mock.mockPatients);
    
    case url.startsWith("/api/patients/"):
      const patientId = url.split('/')[3];
      if (url.endsWith("/vital-signs")) {
        const vitalSigns = { heartRate: [88, 90, 85], bloodPressure: ["120/80", "122/81", "118/79"] };
        return createMockResponse(vitalSigns);
      }
      const patient = mock.mockPatients.find(p => p.id === patientId);
      return createMockResponse(patient);

    // --- ROTAS DE LEITOS (EXISTENTES) ---
    case url === "/api/beds":
      return createMockResponse(mock.mockBeds);

    // --- ROTAS DE PREDIÇÕES DE IA (EXISTENTES) ---
    case url === "/api/predictions/risk":
      return createMockResponse(Object.values(mock.mockRiskPredictions));
    
    case url.startsWith("/api/predictions/risk/"):
      const riskPatientId = url.split('/')[4];
      const prediction = mock.mockRiskPredictions[riskPatientId as keyof typeof mock.mockRiskPredictions];
      return createMockResponse(prediction);
    
    case url === "/api/predictions/bed-optimization":
      const optimization = [{ patientId: "P002", recommendation: "Mover para leito de observação" }];
      return createMockResponse(optimization);
      
    // --- ROTAS DE INSIGHTS DO LLM (EXISTENTES) ---
    case url === "/api/llm/clinical-insights":
      return createMockResponse([mock.mockClinicalInsights]);

    case url.startsWith("/api/llm/summarize-patient/"):
      const summary = { summary: "Este é um resumo gerado pela IA mockada sobre o paciente." };
      return createMockResponse(summary);

    // --- ROTA DE ATIVIDADES (EXISTENTES) ---
    case url.startsWith("/api/activities"):
      const activities = [{ id: 1, description: "Alerta de risco alto para João Silva" }];
      return createMockResponse(activities);

    // --- CASO NENHUMA ROTA SEJA ENCONTRADA ---
    default:
      return new Promise<Response>(resolve => 
          setTimeout(() => resolve(new Response(`[MOCK API] Rota não encontrada: ${url}`, { status: 404 })), 100)
      );
  }
}

// O RESTO DO ARQUIVO CONTINUA EXATAMENTE IGUAL
async function throwIfResNotOk(res: Response) {
  if (!res.ok) {
    const text = (await res.text()) || res.statusText;
    throw new Error(`${res.status}: ${text}`);
  }
}

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchInterval: false,
      refetchOnWindowFocus: false,
      staleTime: Infinity,
      retry: false,
    },
    mutations: {
      retry: false,
    },
  },
});