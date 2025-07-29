# main.py - VitalFlow AI v5.0 Server
# Esta versão integra os modelos avançados do advanced_models.py (VitalFlow AI v5.0)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
import uvicorn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import os
import tensorflow as tf
from tensorflow import keras
import logging
import warnings
warnings.filterwarnings('ignore')

# Importa os modelos avançados e a configuração da nova arquitetura V5
from advanced_models import (
    causal_ai, digital_twin, rl_optimizer, gnn_module, federated_orchestrator,
    initialize_and_train_all_models_v5, config as vitalflow_config # Renomeia 'config' para evitar conflito
)
# Importa classes específicas para uso nos modelos Pydantic de resposta
from advanced_models import IndividualCausalEffect, FormalVerificationReport

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel('ERROR') # Suprime avisos do TensorFlow

app = FastAPI(
    title="VitalFlow AI Server - v5.0 (IA Hospitalar Futurística)",
    description="API de IA em tempo real, Ética, Explicável e Adaptativa para Operações Hospitalares, alimentada pelos modelos VitalFlow v5.0.",
    version="5.0.0"
)

# Middleware CORS para integração com frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite todas as origens para desenvolvimento. Ajuste para produção.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelos Pydantic para a API V5 ---

# Reutiliza IndividualCausalEffect do advanced_models, adaptando para resposta
class IndividualCausalEffectResponse(BaseModel):
    treatment: str
    outcome: str
    point_estimate: float
    confidence_interval: Tuple[float, float]
    patient_segment: Dict[str, Any]
    counterfactual_explanation: Optional[Dict[str, Any]] = None

class VitalSignData(BaseModel):
    heart_rate: Optional[float] = None
    blood_pressure_systolic: Optional[float] = None
    blood_pressure_diastolic: Optional[float] = None
    temperature: Optional[float] = None
    respiratory_rate: Optional[float] = None
    oxygen_saturation: Optional[float] = None
    glucose_level: Optional[float] = None
    white_blood_cell_count: Optional[float] = None
    creatinine: Optional[float] = None

class PatientDataV5(BaseModel):
    patient_id: str
    age: int
    gender: str
    admission_date: str
    comorbidities: Optional[str] = None
    vital_signs: Optional[VitalSignData] = Field(default_factory=VitalSignData) # Garante que seja sempre um objeto VitalSignData

class RiskPredictionV5(BaseModel):
    patient_id: str
    overall_risk_score: float = Field(..., description="Pontuação de risco agregada de 0 a 100.")
    risk_factors_identified: List[str] = Field(..., description="Fatores chave que contribuem para o risco.")
    recommendation: str = Field(..., description="Recomendação acionável gerada pela IA.")
    causal_analysis: Optional[IndividualCausalEffectResponse] = Field(None, description="Insights causais detalhados e contrafactuais.")
    model_confidence: float = Field(..., description="Nível de confiança da predição (0-1).")

class PatientFlowScenario(BaseModel):
    scenario_id: str = Field(..., description="ID único para este cenário simulado.")
    hour_predictions: List[Dict[str, Any]] = Field(..., description="Lista de predições horárias para admissões, altas, ocupação, etc.")
    plausibility_score: float = Field(..., description="Pontuação indicando a plausibilidade deste cenário (0-1).")
    key_metrics: Dict[str, Any] = Field(..., description="Métricas de resumo para o cenário (ex: pico de ocupação, tempo médio de espera).")

class BedOptimizationV5(BaseModel):
    patient_id: str
    recommended_action: str = Field(..., description="Ação específica recomendada para alocação de leitos.")
    reasoning: str = Field(..., description="Explicação para a recomendação, derivada do agente de RL.")
    priority: int = Field(..., description="Nível de prioridade para a ação (1=mais alta).")
    confidence: float = Field(..., description="Confiança na recomendação (0-1).")
    expected_impact: Dict[str, float] = Field(..., description="Impacto esperado quantificado da ação (ex: {'reducao_mortalidade': 0.05}).")

class EquipmentData(BaseModel):
    equipment_id: str
    sensor_data: List[float] # Exemplo: [temperatura, pressão, vibração, tempo_de_uso, dias_desde_ultima_manutencao]
    sequence_length: int = Field(..., description="Comprimento da sequência de dados do sensor para predição.")

class EquipmentFailurePrediction(BaseModel):
    equipment_id: str
    failure_probability: float = Field(..., description="Probabilidade prevista de falha (0-1).")
    recommendation: str = Field(..., description="Recomendação acionável para manutenção.")
    confidence: float = Field(..., description="Confiança na predição (0-1).")

class NetworkAnalysisRequest(BaseModel):
    graph_snapshot: Dict[str, Any] = Field(..., description="Instantâneo atual da rede hospitalar (nós e arestas).")
    analysis_type: str = Field(..., description="Tipo de análise solicitada (ex: 'propagacao_infeccao', 'gargalo_recursos').")

class NetworkAnalysisReportResponse(BaseModel):
    analysis_type: str
    identified_risks: List[str]
    recommendations: List[str]
    timestamp: str

class FormalVerificationReportSummary(BaseModel): # Simplificado para resposta da API
    policy_name: str
    property_checked: str
    is_safe: bool
    details: str
    timestamp: str

# --- Instâncias Globais de Modelos de IA Avançados ---
# Estas são importadas diretamente do advanced_models.py
# causal_ai, digital_twin, rl_optimizer, gnn_module, federated_orchestrator, vitalflow_config

@app.on_event("startup")
async def startup_event():
    """Inicializa todos os modelos avançados da VitalFlow AI na inicialização."""
    logger.info("🚀 Iniciando VitalFlow AI Server v5.0...")
    
    try:
        # Esta função orquestra o treinamento/carregamento de todos os modelos V5
        await initialize_and_train_all_models_v5()
        logger.info("✅ Todos os modelos VitalFlow AI v5.0 inicializados e prontos!")
    except Exception as e:
        logger.error(f"❌ Falha ao inicializar os modelos VitalFlow AI v5.0: {e}", exc_info=True)
        # Em um sistema de produção, isso pode impedir o servidor de iniciar ou marcar os modelos como não saudáveis.
        raise RuntimeError(f"Erro crítico durante a inicialização do modelo: {e}")

@app.get("/")
async def root():
    return {
        "message": "VitalFlow AI Server - v5.0: IA Hospitalar Futurística operacional.",
        "version": "5.0.0",
        "system_status": "ativo",
        "models_initialized": {
            "causal_ai": causal_ai.is_trained,
            "digital_twin": digital_twin.is_trained,
            "rl_optimizer": rl_optimizer.is_trained,
            "gnn_module": gnn_module.is_trained,
            "federated_learning_enabled": vitalflow_config.FEDERATED_LEARNING_ENABLED
        }
    }

@app.post("/predict/risk", response_model=RiskPredictionV5)
async def predict_patient_risk(patient: PatientDataV5):
    """
    Prediz o risco do paciente usando IA Causal, fornecendo Efeitos de Tratamento Individualizados (ITE)
    e explicações contrafactuais para maior interpretabilidade e acionabilidade.
    """
    if not causal_ai.is_trained:
        raise HTTPException(status_code=503, detail="Modelo de IA Causal ainda não treinado ou carregado.")

    try:
        # Prepara os dados do paciente para o modelo causal. Isso requer mapear PatientDataV5 para o
        # formato esperado do método causal_ai.estimate_ite (uma Series/DataFrame pandas).
        # Este é um passo crítico onde o mapeamento de dados reais precisa ser robusto.
        # Para demonstração, criamos um DataFrame dummy.
        patient_df = pd.DataFrame([{
            'age': patient.age,
            'gender': 1 if patient.gender.lower() == 'male' else 0,
            'heart_rate': patient.vital_signs.heart_rate if patient.vital_signs else 75,
            'blood_pressure_systolic': patient.vital_signs.blood_pressure_systolic if patient.vital_signs else 120,
            'blood_pressure_diastolic': patient.vital_signs.blood_pressure_diastolic if patient.vital_signs else 80,
            'temperature': patient.vital_signs.temperature if patient.vital_signs else 98.6,
            'respiratory_rate': patient.vital_signs.respiratory_rate if patient.vital_signs else 16,
            'oxygen_saturation': patient.vital_signs.oxygen_saturation if patient.vital_signs else 98,
            'glucose_level': patient.vital_signs.glucose_level if patient.vital_signs else 100,
            'white_blood_cell_count': patient.vital_signs.white_blood_cell_count if patient.vital_signs else 7,
            'creatinine': patient.vital_signs.creatinine if patient.vital_signs else 1.0,
            # Adiciona valores dummy para tratamento/resultado/confounders esperados por causal_ai.estimate_ite
            'intervention_A': 0, # Assume nenhuma intervenção para linha de base
            'intervention_B': 0,
            'patient_recovery_rate': 0.5, # Placeholder para o resultado atual
            'complication_risk': 0.2, # Placeholder para o resultado atual
            'comorbidity_score': 5.0 # Placeholder
        }])
        
        # Exemplo: Estima ITE para 'intervention_A' em 'complication_risk'
        # Em um sistema real, a escolha do tratamento/resultado seria dinâmica com base no contexto.
        causal_analysis_result: IndividualCausalEffect = causal_ai.estimate_ite(
            patient_df,
            treatment='intervention_A', # Tratamento de exemplo
            outcome='complication_risk' # Resultado de exemplo
        )

        # Deriva o risco geral e a recomendação da análise causal e outros fatores
        # Esta lógica seria mais sofisticada em um sistema real, combinando ITE com outros modelos de risco.
        overall_risk_score = (causal_analysis_result.point_estimate * 100).clip(0, 100) # Mapeamento simples
        risk_factors = [f"Efeito causal de {causal_analysis_result.treatment} em {causal_analysis_result.outcome} é {causal_analysis_result.point_estimate:.2f}"]
        
        recommendation = "Com base na análise causal, considere intervenções específicas para um resultado ótimo."
        if overall_risk_score > 70:
            recommendation = f"CRÍTICO: Alto risco indicado. A análise causal sugere que '{causal_analysis_result.treatment}' pode impactar significativamente '{causal_analysis_result.outcome}'. Ação imediata necessária."
        elif overall_risk_score > 40:
            recommendation = f"ALTO RISCO: A análise causal destaca o impacto potencial de '{causal_analysis_result.treatment}' em '{causal_analysis_result.outcome}'. Monitore de perto e considere a intervenção."

        # Confiança simulada (em um sistema real, isso viria da estimativa de incerteza do modelo)
        model_confidence = 0.92 - (abs(causal_analysis_result.point_estimate) * 0.1) # Placeholder

        return RiskPredictionV5(
            patient_id=patient.patient_id,
            overall_risk_score=float(overall_risk_score),
            risk_factors_identified=risk_factors,
            recommendation=recommendation,
            causal_analysis=IndividualCausalEffectResponse(**causal_analysis_result.__dict__), # Converte dataclass para modelo Pydantic
            model_confidence=model_confidence
        )
    except Exception as e:
        logger.error(f"Erro ao predizer risco com IA Causal: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro de predição: {str(e)}")

@app.get("/predict/flow", response_model=List[PatientFlowScenario])
async def predict_patient_flow(num_scenarios: int = 3):
    """
    Prediz o fluxo de pacientes para o próximo período gerando múltiplos cenários futuros plausíveis
    usando o Digital Twin Generativo.
    """
    if not digital_twin.is_trained:
        raise HTTPException(status_code=503, detail="Modelo Digital Twin ainda não treinado ou carregado.")

    try:
        # Para demonstração, usa uma sequência dummy do último estado conhecido.
        # Em um sistema real, isso viria de dados hospitalares em tempo real.
        dummy_last_known_sequence = np.random.rand(vitalflow_config.DT_SEQUENCE_LENGTH, vitalflow_config.DT_FEATURES)
        
        # Gera múltiplos cenários futuros
        synthetic_futures = digital_twin.generate_future_scenarios(dummy_last_known_sequence, num_scenarios)
        
        scenarios = []
        for i, future_sequence in enumerate(synthetic_futures):
            # Converte a sequência gerada em uma lista de predições horárias
            # Este mapeamento precisa ser preciso com base na estrutura do seu DT_FEATURES
            hourly_preds = []
            for hour_idx in range(future_sequence.shape[0]):
                # Assumindo que DT_FEATURES mapeiam para certas métricas (ex: admissões, altas, ocupação)
                # Este é um mapeamento conceitual.
                hourly_preds.append({
                    "hour": hour_idx,
                    "predicted_admissions": max(0, int(future_sequence[hour_idx, 0] * 10)), # Escala dados dummy
                    "predicted_discharges": max(0, int(future_sequence[hour_idx, 1] * 8)),
                    "predicted_occupancy_rate": float(future_sequence[hour_idx, 2] * 0.8 + 0.2).clip(0,1), # Escala para 0-1
                    "predicted_staff_workload": float(future_sequence[hour_idx, 3] * 0.5 + 0.5).clip(0,1)
                })
            
            # Plausibilidade e métricas chave simuladas
            plausibility = 0.7 + np.random.rand() * 0.3 # Plausibilidade aleatória
            key_metrics = {
                "peak_occupancy_rate": max([p['predicted_occupancy_rate'] for p in hourly_preds]),
                "total_admissions": sum([p['predicted_admissions'] for p in hourly_preds]),
                "total_discharges": sum([p['predicted_discharges'] for p in hourly_preds])
            }

            scenarios.append(PatientFlowScenario(
                scenario_id=f"scenario_{i+1}",
                hour_predictions=hourly_preds,
                plausibility_score=plausibility,
                key_metrics=key_metrics
            ))
        
        return scenarios
        
    except Exception as e:
        logger.error(f"Erro ao predizer fluxo com Digital Twin: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro de predição de fluxo: {str(e)}")

@app.post("/optimize/beds", response_model=List[BedOptimizationV5])
async def optimize_bed_allocation(patients: List[PatientDataV5]):
    """
    Otimiza a alocação de leitos usando o Otimizador de Aprendizado por Reforço,
    considerando restrições éticas e fornecendo recomendações acionáveis.
    """
    if not rl_optimizer.is_trained:
        raise HTTPException(status_code=503, detail="Otimizador de RL ainda não treinado ou carregado.")

    try:
        optimizations = []
        # Em um sistema real, `current_hospital_state` seria uma observação abrangente
        # do ambiente hospitalar, potencialmente incluindo dados agregados de pacientes,
        # disponibilidade de recursos, etc., formatados para o agente de RL.
        # Para esta demonstração, iteraremos pelos pacientes e simularemos a saída do RL.

        # Uma abordagem mais realista seria agregar todos os dados do paciente em um único estado
        # sobre o qual o agente de RL pode atuar, e o agente então produziria um conjunto de ações
        # para todos os pacientes/recursos.
        
        # Para simplicidade, vamos simular o agente de RL fornecendo recomendações por paciente
        # com base em algum estado simplificado derivado.
        for patient in patients:
            # Cria um 'current_hospital_state' simplificado para este paciente para o agente de RL
            # Isso precisa corresponder ao espaço de observação do HospitalEnvV5
            # Para demonstração, um estado dummy (ex: sinais vitais do paciente como parte do estado)
            current_patient_state_for_rl = np.array([
                patient.age / 100.0, # Normaliza
                1 if patient.gender.lower() == 'male' else 0,
                (patient.vital_signs.oxygen_saturation or 98) / 100.0,
                (patient.vital_signs.heart_rate or 75) / 150.0,
                (patient.vital_signs.temperature or 98.6) / 105.0,
                # ... outras features relevantes que o agente de RL observa
            ])
            # Preenche ou trunca para corresponder à forma esperada (vitalflow_config.DT_FEATURES)
            # Este é um placeholder para engenharia de estado adequada.
            padded_state = np.pad(current_patient_state_for_rl, (0, vitalflow_config.DT_FEATURES - len(current_patient_state_for_rl)), 'constant')
            
            # Obtém a recomendação do otimizador de RL
            # O método `optimize_bed_allocation` do otimizador de RL retorna um Dict
            rl_recommendation_output = rl_optimizer.optimize_bed_allocation(padded_state)

            optimizations.append(BedOptimizationV5(
                patient_id=patient.patient_id,
                recommended_action=rl_recommendation_output['recommended_action'],
                reasoning=rl_recommendation_output['reasoning'],
                priority=rl_recommendation_output['priority'],
                confidence=rl_recommendation_output['confidence'],
                expected_impact=rl_recommendation_output['expected_impact']
            ))
        
        return optimizations
        
    except Exception as e:
        logger.error(f"Erro ao otimizar leitos com Otimizador de RL: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro de otimização de leitos: {str(e)}")

@app.post("/digital_twin/predict_equipment_failure", response_model=EquipmentFailurePrediction)
async def predict_equipment_failure(equipment_data: EquipmentData):
    """
    Prediz a probabilidade de falha de equipamento usando o módulo de manutenção preditiva do Digital Twin.
    """
    if not digital_twin.config.DT_PREDICTIVE_MAINTENANCE_ENABLED:
        raise HTTPException(status_code=400, detail="A manutenção preditiva não está habilitada na configuração do VitalFlow.")
    if 'equipment' not in digital_twin.multi_scale_models:
        raise HTTPException(status_code=503, detail="Modelo Digital Twin de equipamento não treinado ou carregado.")

    try:
        # Garante que os dados do sensor estejam no formato correto para o modelo
        # O modelo espera (batch_size, sequence_length, num_features)
        # Assumimos que equipment_data.sensor_data já tem o número correto de features por passo de tempo
        num_features_per_step = len(equipment_data.sensor_data) // equipment_data.sequence_length
        sensor_data_np = np.array(equipment_data.sensor_data).reshape(1, equipment_data.sequence_length, num_features_per_step)
        
        failure_prob = digital_twin.predict_equipment_failure(sensor_data_np, equipment_data.equipment_id)
        
        recommendation = "Monitore o equipamento regularmente."
        if failure_prob > 0.7:
            recommendation = "CRÍTICO: Alta probabilidade de falha. Agende manutenção imediata."
        elif failure_prob > 0.4:
            recommendation = "ALTO: Risco aumentado de falha. Planeje a manutenção em breve."

        return EquipmentFailurePrediction(
            equipment_id=equipment_data.equipment_id,
            failure_probability=failure_prob,
            recommendation=recommendation,
            confidence=0.85 # Placeholder, deve ser derivado do modelo
        )
    except Exception as e:
        logger.error(f"Erro ao predizer falha de equipamento: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro de predição de falha de equipamento: {str(e)}")

@app.post("/gnn/analyze_network", response_model=NetworkAnalysisReportResponse)
async def analyze_hospital_network(request: NetworkAnalysisRequest):
    """
    Analisa a rede interna do hospital (ex: transferências de pacientes, interações da equipe)
    usando Redes Neurais Gráficas para identificar riscos como propagação de infecções ou gargalos de recursos.
    """
    if not gnn_module.is_trained:
        raise HTTPException(status_code=503, detail="Módulo GNN não treinado ou carregado.")

    try:
        # O `graph_snapshot` seria processado pelo módulo GNN.
        # Para demonstração, passamos diretamente e deixamos o módulo GNN retornar um relatório simulado.
        report_data = gnn_module.analyze_network_for_risks(request.graph_snapshot)
        
        # O `report_data` retornado por gnn_module.analyze_network_for_risks é um Dict
        # que já contém as chaves 'infection_spread_risk_nodes', 'resource_bottleneck_departments', 'recommendations'.
        # Precisamos concatenar as listas de riscos se elas vierem separadas.
        identified_risks_list = []
        if "infection_spread_risk_nodes" in report_data:
            identified_risks_list.extend(report_data["infection_spread_risk_nodes"])
        if "resource_bottleneck_departments" in report_data:
            identified_risks_list.extend(report_data["resource_bottleneck_departments"])

        return NetworkAnalysisReportResponse(
            analysis_type=request.analysis_type,
            identified_risks=identified_risks_list,
            recommendations=report_data.get("recommendations", []),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Erro ao analisar rede hospitalar com GNN: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro de análise de rede: {str(e)}")

@app.get("/rl/formal_verification_reports", response_model=List[FormalVerificationReportSummary])
async def get_formal_verification_reports():
    """
    Recupera os relatórios de verificação formal para a política de Aprendizado por Reforço,
    garantindo a adesão às propriedades de segurança e ética.
    """
    if not vitalflow_config.FORMAL_VERIFICATION_ENABLED:
        raise HTTPException(status_code=400, detail="A Verificação Formal não está habilitada na configuração do VitalFlow.")
    
    if not rl_optimizer.formal_verification_reports:
        raise HTTPException(status_code=404, detail="Nenhum relatório de verificação formal encontrado.")
    
    # Converte cada dataclass FormalVerificationReport para o modelo Pydantic FormalVerificationReportSummary
    return [FormalVerificationReportSummary(**report.__dict__) for report in rl_optimizer.formal_verification_reports]

@app.get("/federated_learning/status")
async def get_federated_learning_status():
    """
    Fornece o status atual do Orquestrador de Aprendizado Federado.
    """
    return {
        "federated_learning_enabled": vitalflow_config.FEDERATED_LEARNING_ENABLED,
        "orchestrator_active": federated_orchestrator.is_active,
        "last_round_info": "Não implementado nesta demonstração, mas mostraria detalhes da última rodada de FL."
    }

@app.get("/models/status")
async def get_model_status():
    """Obtém o status de todos os modelos VitalFlow AI v5.0."""
    return {
        "causal_ai_trained": causal_ai.is_trained,
        "digital_twin_trained": digital_twin.is_trained,
        "rl_optimizer_trained": rl_optimizer.is_trained,
        "gnn_module_trained": gnn_module.is_trained,
        "federated_learning_orchestrator_active": federated_orchestrator.is_active,
        "server_status": "ativo",
        "last_status_check": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Garanta que o arquivo `advanced_models.py` esteja nomeado corretamente e no mesmo diretório
    # ou acessível via PYTHONPATH.
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True, # Recarrega para desenvolvimento, desabilite em produção
        log_level="info"
    )