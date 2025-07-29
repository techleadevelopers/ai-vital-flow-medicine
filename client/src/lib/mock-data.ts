// src/lib/mock-data.ts

// Tipos importados (podem vir de um arquivo de schema compartilhado)
export interface Patient {
  id: string;
  name: string;
  age: number;
  gender: 'Masculino' | 'Feminino';
  diagnosis: string;
  bed: string;
  admissionDate: string;
}

export interface Bed {
  id: string;
  type: 'UTI' | 'Geral' | 'Emergência';
  status: 'Ocupado' | 'Disponível' | 'Manutenção';
  patientId?: string;
}

// ... outros tipos como VitalSigns, Activity, etc.

// --- DADOS MOCKADOS ---

export const mockPatients: Patient[] = [
  { id: 'P001', name: 'João Silva', age: 68, gender: 'Masculino', diagnosis: 'Insuficiência Cardíaca', bed: 'UTI-101', admissionDate: '2025-07-20' },
  { id: 'P002', name: 'Maria Oliveira', age: 75, gender: 'Feminino', diagnosis: 'Pneumonia', bed: 'GER-203', admissionDate: '2025-07-22' },
  { id: 'P003', name: 'Carlos Pereira', age: 55, gender: 'Masculino', diagnosis: 'Fratura de Fêmur', bed: 'EME-01', admissionDate: '2025-07-25' },
];

export const mockBeds: Bed[] = [
    { id: 'UTI-101', type: 'UTI', status: 'Ocupado', patientId: 'P001' },
    { id: 'UTI-102', type: 'UTI', status: 'Disponível' },
    { id: 'GER-203', type: 'Geral', status: 'Ocupado', patientId: 'P002' },
];

// --- DADOS MOCKADOS DA IA (O que o backend Python retornaria) ---

export const mockRiskPredictions = {
  'P001': {
    patientId: 'P001',
    riskScore: 85.5,
    confidence: 0.92,
    level: 'Alto',
    factors: ['Saturação de O2 baixa', 'Histórico de DAC', 'Pressão arterial elevada'],
    recommendation: 'Monitoramento contínuo na UTI e avaliação cardiológica imediata.',
  },
  'P002': {
    patientId: 'P002',
    riskScore: 45.0,
    confidence: 0.88,
    level: 'Médio',
    factors: ['Idade avançada', 'Infecção ativa'],
    recommendation: 'Monitorar sinais vitais a cada 4 horas.',
  },
  'P003': {
    patientId: 'P003',
    riskScore: 15.2,
    confidence: 0.98,
    level: 'Baixo',
    factors: ['Pós-operatório estável'],
    recommendation: 'Manter observação de rotina.',
  },
};

export const mockClinicalInsights = {
    type: 'Eficiência Operacional',
    title: 'Otimização do Fluxo de Admissão',
    content: 'Identificamos um padrão de atraso nas admissões de cardiologia entre 14h e 16h. Recomenda-se alocar uma equipe de enfermagem adicional nesse período para acelerar a triagem e liberação de leitos.',
    priority: 'Média',
};

// --- DADOS MOCKADOS PARA GRÁFICOS DE ANALYTICS ---

// Para o gráfico de "Análise de Tendências"
export const mockTrendData = [
  { name: 'Jan', "Admissões": 40, "Altas": 24, "Ocupação Média": 65 },
  { name: 'Fev', "Admissões": 30, "Altas": 13, "Ocupação Média": 70 },
  { name: 'Mar', "Admissões": 50, "Altas": 38, "Ocupação Média": 75 },
  { name: 'Abr', "Admissões": 47, "Altas": 39, "Ocupação Média": 80 },
  { name: 'Mai', "Admissões": 55, "Altas": 48, "Ocupação Média": 85 },
  { name: 'Jun', "Admissões": 58, "Altas": 45, "Ocupação Média": 88 },
  { name: 'Jul', "Admissões": 62, "Altas": 55, "Ocupação Média": 90 },
];

// Para o gráfico de "Distribuição de Recursos" (ex: um gráfico de barras)
export const mockResourceData = [
    { name: 'UTI', "Leitos Ocupados": 18, "Leitos Disponíveis": 6 },
    { name: 'Geral', "Leitos Ocupados": 65, "Leitos Disponíveis": 15 },
    { name: 'Emergência', "Leitos Ocupados": 12, "Leitos Disponíveis": 4 },
    { name: 'Maternidade', "Leitos Ocupados": 22, "Leitos Disponíveis": 8 },
];

// Para o gráfico de "Performance Histórica"
export const mockPerformanceData = [
    { month: 'Janeiro', "Satisfação Paciente": 4.2, "Tempo Médio Espera": 25 },
    { month: 'Fevereiro', "Satisfação Paciente": 4.3, "Tempo Médio Espera": 22 },
    { month: 'Março', "Satisfação Paciente": 4.5, "Tempo Médio Espera": 20 },
    { month: 'Abril', "Satisfação Paciente": 4.4, "Tempo Médio Espera": 21 },
    { month: 'Maio', "Satisfação Paciente": 4.7, "Tempo Médio Espera": 15 },
    { month: 'Junho', "Satisfação Paciente": 4.8, "Tempo Médio Espera": 12 },
];