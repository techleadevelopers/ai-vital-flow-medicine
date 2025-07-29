# main.py - VitalFlow AI v6.0 Server
# Esta versão integra os modelos ultra-avançados do advanced_models.py (VitalFlow AI v6.0)

from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple, Union
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

# Importa os modelos avançados e a configuração da nova arquitetura V6
from advanced_models import (
    causal_ai, digital_twin, rl_optimizer, gnn_module, federated_orchestrator,
    initialize_and_train_all_models_v6, config as vitalflow_config,
    mkg_manager, rtdi_manager, cl_manager, hpc_accelerator, xai_manager, llm_assistant
)
# Importa dataclasses específicas para uso nos modelos Pydantic de resposta
from advanced_models import IndividualCausalEffect, FormalVerificationReport, PrescriptiveAction

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel('ERROR') # Suprime avisos do TensorFlow

app = FastAPI(
    title="VitalFlow AI Server - v6.0 (IA Hospitalar Futurística Ultra-Avançada)",
    description="API de IA em tempo real, Cognitiva, Ética, Explicável e Adaptativa para Operações Hospitalares, alimentada pelos modelos VitalFlow v6.0.",
    version="6.0.0"
)

# Middleware CORS para integração com frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite todas as origens para desenvolvimento. Ajuste para produção.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelos Pydantic para a API V6 ---

# Reutiliza IndividualCausalEffect do advanced_models, adaptando para resposta
class IndividualCausalEffectResponse(BaseModel):
    treatment: str
    outcome: str
    point_estimate: float
    confidence_interval: Tuple[float, float]
    patient_segment: Dict[str, Any]
    counterfactual_scenario: Optional[Dict[str, Any]] = None
    knowledge_based_explanation: Optional[str] = None

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

class PatientDataV6(BaseModel):
    patient_id: str
    age: int
    gender: str
    admission_date: str
    comorbidities: Optional[str] = None
    vital_signs: Optional[VitalSignData] = Field(default_factory=VitalSignData)

class RiskPredictionV6(BaseModel):
    patient_id: str
    overall_risk_score: float = Field(..., description="Pontuação de risco agregada de 0 a 100.")
    risk_factors_identified: List[str] = Field(..., description="Fatores chave que contribuem para o risco.")
    recommendation: str = Field(..., description="Recomendação acionável gerada pela IA.")
    causal_analysis: Optional[IndividualCausalEffectResponse] = Field(None, description="Insights causais detalhados e contrafactuais.")
    model_confidence: float = Field(..., description="Nível de confiança da predição (0-1).")
    xai_explanation_link: Optional[str] = Field(None, description="Link para uma explicação XAI mais detalhada.")

class PatientFlowScenario(BaseModel):
    scenario_id: str = Field(..., description="ID único para este cenário simulado.")
    hour_predictions: List[Dict[str, Any]] = Field(..., description="Lista de predições horárias para admissões, altas, ocupação, etc.")
    plausibility_score: float = Field(..., description="Pontuação indicando a plausibilidade deste cenário (0-1).")
    key_metrics: Dict[str, Any] = Field(..., description="Métricas de resumo para o cenário (ex: pico de ocupação, tempo médio de espera).")
    uncertainty_quantification: Optional[Dict[str, Any]] = Field(None, description="Quantificação da incerteza nas predições do cenário.")

class PrescriptiveActionResponse(BaseModel): # Mapeia a dataclass PrescriptiveAction
    action_id: str
    description: str
    target_entity_id: str
    action_type: str
    expected_impact: Dict[str, float]
    confidence: float
    reasoning: str
    knowledge_references: List[str]

class BedOptimizationV6(BaseModel):
    patient_id: str
    recommended_action: str = Field(..., description="Ação específica recomendada para alocação de leitos.")
    reasoning: str = Field(..., description="Explicação para a recomendação, derivada do agente de RL.")
    priority: int = Field(..., description="Nível de prioridade para a ação (1=mais alta).")
    confidence: float = Field(..., description="Confiança na recomendação (0-1).")
    expected_impact: Dict[str, float] = Field(..., description="Impacto esperado quantificado da ação (ex: {'reducao_mortalidade': 0.05}).")
    formal_verification_status: Optional[Dict[str, Any]] = Field(None, description="Status da verificação formal da política de RL que gerou a ação.")

class EquipmentData(BaseModel):
    equipment_id: str
    sensor_data: List[float] # Exemplo: [temperatura, pressão, vibração, tempo_de_uso, dias_desde_ultima_manutencao]
    sequence_length: int = Field(..., description="Comprimento da sequência de dados do sensor para predição.")

class EquipmentFailurePrediction(BaseModel):
    equipment_id: str
    failure_probability: float = Field(..., description="Probabilidade prevista de falha (0-1).")
    recommendation: str = Field(..., description="Recomendação acionável para manutenção.")
    confidence: float = Field(..., description="Confiança na predição (0-1).")
    prescriptive_action: Optional[PrescriptiveActionResponse] = Field(None, description="Ação prescritiva gerada pelo Digital Twin.")

class NetworkAnalysisRequest(BaseModel):
    graph_snapshot: Dict[str, Any] = Field(..., description="Instantâneo atual da rede hospitalar (nós e arestas).")
    analysis_type: str = Field(..., description="Tipo de análise solicitada (ex: 'infection_spread', 'resource_bottleneck').")

class NetworkAnalysisReportResponse(BaseModel):
    analysis_type: str
    identified_risks: List[str]
    recommendations: List[str]
    timestamp: str
    knowledge_references: List[str] = Field(default_factory=list, description="Referências a conhecimentos do MKG usados na análise.")

class FormalVerificationReportSummary(BaseModel): # Mapeia a dataclass FormalVerificationReport
    policy_name: str
    property_checked: str
    is_safe: bool
    details: str
    timestamp: str
    property_type: str

class RealTimeDataPoint(BaseModel):
    source_type: str = Field(..., description="Tipo da fonte de dados (ex: 'ehr', 'iot_sensors', 'medical_imaging', 'genomic_data').")
    data: Dict[str, Any] = Field(..., description="Ponto de dado bruto a ser ingerido.")
    entity_id: str = Field(..., description="ID da entidade a que o dado se refere (ex: patient_id, equipment_id).")

class RealTimeDataIngestionResponse(BaseModel):
    status: str
    harmonized_data_preview: Dict[str, Any]
    timestamp: str

class XAIExplanationResponse(BaseModel):
    model: str
    prediction: str
    data_point: Dict[str, Any]
    explanation_level: str
    details: List[Dict[str, Any]]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class LLMQueryRequest(BaseModel):
    query: str = Field(..., description="Consulta em linguagem natural para o assistente de IA.")
    patient_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class LLMQueryResponse(BaseModel):
    response_type: str
    content: Any
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class KnowledgeGraphQueryRequest(BaseModel):
    query: str = Field(..., description="Consulta em linguagem natural ou formato estruturado para o Grafo de Conhecimento Médico.")

class KnowledgeGraphQueryResponse(BaseModel):
    query: str
    results: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ContinualLearningStatus(BaseModel):
    model_name: str
    drift_status: str
    last_drift_score: Optional[float] = None
    last_adaptation_time: Optional[str] = None

class HPCAccelerationRequest(BaseModel):
    problem_description: Dict[str, Any]
    problem_type: str = Field(..., description="Tipo de problema (ex: 'optimization', 'simulation').")

class HPCAccelerationResponse(BaseModel):
    status: str
    optimized_solution: Optional[Dict[str, Any]] = None
    simulation_results: Optional[Dict[str, Any]] = None
    method: str
    speed_up_factor: Optional[float] = None

@app.on_event("startup")
async def startup_event():
    """Inicializa todos os modelos avançados da VitalFlow AI na inicialização."""
    logger.info("🚀 Iniciando VitalFlow AI Server v6.0...")
    
    try:
        await initialize_and_train_all_models_v6()
        logger.info("✅ Todos os modelos VitalFlow AI v6.0 inicializados e prontos!")
    except Exception as e:
        logger.error(f"❌ Falha ao inicializar os modelos VitalFlow AI v6.0: {e}", exc_info=True)
        raise RuntimeError(f"Erro crítico durante a inicialização do modelo: {e}")

@app.get("/")
async def root():
    return {
        "message": "VitalFlow AI Server - v6.0: IA Hospitalar Futurística Ultra-Avançada operacional.",
        "version": "6.0.0",
        "system_status": "ativo",
        "models_initialized": {
            "causal_ai": causal_ai.is_trained,
            "digital_twin": digital_twin.is_trained,
            "rl_optimizer": rl_optimizer.is_trained,
            "gnn_module": gnn_module.is_trained,
            "federated_learning_enabled": vitalflow_config.FEDERATED_LEARNING_ENABLED,
            "mkg_manager_loaded": bool(mkg_manager.knowledge_graph),
            "rtdi_manager_initialized": bool(rtdi_manager.data_streams),
            "cl_manager_initialized": True, # Always initialized, but models registered later
            "hpc_accelerator_initialized": True,
            "xai_manager_initialized": True,
            "llm_assistant_initialized": True
        }
    }

@app.post("/predict/risk", response_model=RiskPredictionV6)
async def predict_patient_risk(patient: PatientDataV6):
    """
    Prediz o risco do paciente usando IA Causal, fornecendo Efeitos de Tratamento Individualizados (ITE)
    e explicações contrafactuais, enriquecidas por conhecimento médico formal.
    """
    if not causal_ai.is_trained:
        raise HTTPException(status_code=503, detail="Modelo de IA Causal ainda não treinado ou carregado.")

    try:
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
            'intervention_A': 0,
            'intervention_B': 0,
            'patient_recovery_rate': 0.5,
            'complication_risk': 0.2,
            'comorbidity_score': 5.0
        }])
        
        causal_analysis_result: IndividualCausalEffect = causal_ai.estimate_ite(
            patient_df,
            treatment='intervention_A',
            outcome='complication_risk'
        )

        overall_risk_score = (causal_analysis_result.point_estimate * 100).clip(0, 100)
        risk_factors = [f"Efeito causal de {causal_analysis_result.treatment} em {causal_analysis_result.outcome} é {causal_analysis_result.point_estimate:.2f}"]
        
        recommendation = "Com base na análise causal, considere intervenções específicas para um resultado ótimo."
        if overall_risk_score > 70:
            recommendation = f"CRÍTICO: Alto risco indicado. A análise causal sugere que '{causal_analysis_result.treatment}' pode impactar significativamente '{causal_analysis_result.outcome}'. Ação imediata necessária."
        elif overall_risk_score > 40:
            recommendation = f"ALTO RISCO: A análise causal destaca o impacto potencial de '{causal_analysis_result.treatment}' em '{causal_analysis_result.outcome}'. Monitore de perto e considere a intervenção."

        model_confidence = 0.92 - (abs(causal_analysis_result.point_estimate) * 0.1)
        
        # Link para explicação XAI detalhada (conceitual)
        xai_link = f"/xai/explain_decision?model_name=causal_ai&patient_id={patient.patient_id}"

        return RiskPredictionV6(
            patient_id=patient.patient_id,
            overall_risk_score=float(overall_risk_score),
            risk_factors_identified=risk_factors,
            recommendation=recommendation,
            causal_analysis=IndividualCausalEffectResponse(**causal_analysis_result.__dict__),
            model_confidence=model_confidence,
            xai_explanation_link=xai_link
        )
    except Exception as e:
        logger.error(f"Erro ao predizer risco com IA Causal: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro de predição: {str(e)}")

@app.get("/predict/flow", response_model=List[PatientFlowScenario])
async def predict_patient_flow(num_scenarios: int = 3):
    """
    Prediz o fluxo de pacientes para o próximo período gerando múltiplos cenários futuros plausíveis
    usando o Digital Twin Generativo, com quantificação de incerteza.
    """
    if not digital_twin.is_trained:
        raise HTTPException(status_code=503, detail="Modelo Digital Twin ainda não treinado ou carregado.")

    try:
        dummy_last_known_sequence = np.random.rand(vitalflow_config.DT_SEQUENCE_LENGTH, vitalflow_config.DT_FEATURES)
        synthetic_futures = digital_twin.generate_future_scenarios(dummy_last_known_sequence, num_scenarios)
        
        scenarios = []
        for i, future_sequence in enumerate(synthetic_futures):
            hourly_preds = []
            for hour_idx in range(future_sequence.shape[0]):
                hourly_preds.append({
                    "hour": hour_idx,
                    "predicted_admissions": max(0, int(future_sequence[hour_idx, 0] * 10)),
                    "predicted_discharges": max(0, int(future_sequence[hour_idx, 1] * 8)),
                    "predicted_occupancy_rate": float(future_sequence[hour_idx, 2] * 0.8 + 0.2).clip(0,1),
                    "predicted_staff_workload": float(future_sequence[hour_idx, 3] * 0.5 + 0.5).clip(0,1)
                })
            
            plausibility = 0.7 + np.random.rand() * 0.3
            key_metrics = {
                "peak_occupancy_rate": max([p['predicted_occupancy_rate'] for p in hourly_preds]),
                "total_admissions": sum([p['predicted_admissions'] for p in hourly_preds]),
                "total_discharges": sum([p['predicted_discharges'] for p in hourly_preds])
            }
            
            # Quantificação de incerteza (conceitual)
            uncertainty = {
                "admissions_std_dev": np.random.uniform(0.5, 2.0),
                "occupancy_95_ci": (0.6, 0.9)
            }

            scenarios.append(PatientFlowScenario(
                scenario_id=f"scenario_{i+1}",
                hour_predictions=hourly_preds,
                plausibility_score=plausibility,
                key_metrics=key_metrics,
                uncertainty_quantification=uncertainty
            ))
        
        return scenarios
        
    except Exception as e:
        logger.error(f"Erro ao predizer fluxo com Digital Twin: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro de predição de fluxo: {str(e)}")

@app.post("/optimize/beds", response_model=List[BedOptimizationV6])
async def optimize_bed_allocation(patients: List[PatientDataV6]):
    """
    Otimiza a alocação de leitos usando o Otimizador de Aprendizado por Reforço,
    considerando restrições éticas e fornecendo recomendações acionáveis.
    """
    if not rl_optimizer.is_trained:
        raise HTTPException(status_code=503, detail="Otimizador de RL ainda não treinado ou carregado.")

    try:
        optimizations = []
        for patient in patients:
            current_patient_state_for_rl = np.array([
                patient.age / 100.0,
                1 if patient.gender.lower() == 'male' else 0,
                (patient.vital_signs.oxygen_saturation or 98) / 100.0,
                (patient.vital_signs.heart_rate or 75) / 150.0,
                (patient.vital_signs.temperature or 98.6) / 105.0,
            ])
            padded_state = np.pad(current_patient_state_for_rl, (0, vitalflow_config.DT_FEATURES - len(current_patient_state_for_rl)), 'constant')
            
            rl_recommendation_output = rl_optimizer.optimize_bed_allocation(padded_state)

            # Inclui o status da verificação formal mais recente para a política de RL
            formal_status = None
            if vitalflow_config.FORMAL_VERIFICATION_ENABLED and rl_optimizer.formal_verification_reports:
                # Pega o último relatório de segurança como exemplo
                safety_report = next((r for r in rl_optimizer.formal_verification_reports if r.property_type == "safety"), None)
                if safety_report:
                    formal_status = {"is_safe": safety_report.is_safe, "details": safety_report.details}

            optimizations.append(BedOptimizationV6(
                patient_id=patient.patient_id,
                recommended_action=rl_recommendation_output['recommended_action'],
                reasoning=rl_recommendation_output['reasoning'],
                priority=rl_recommendation_output['priority'],
                confidence=rl_recommendation_output['confidence'],
                expected_impact=rl_recommendation_output['expected_impact'],
                formal_verification_status=formal_status
            ))
        
        return optimizations
        
    except Exception as e:
        logger.error(f"Erro ao otimizar leitos com Otimizador de RL: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro de otimização de leitos: {str(e)}")

@app.post("/digital_twin/predict_equipment_failure", response_model=EquipmentFailurePrediction)
async def predict_equipment_failure(equipment_data: EquipmentData):
    """
    Prediz a probabilidade de falha de equipamento usando o módulo de manutenção preditiva do Digital Twin,
    e gera uma ação prescritiva.
    """
    if not digital_twin.config.DT_PREDICTIVE_MAINTENANCE_ENABLED:
        raise HTTPException(status_code=400, detail="A manutenção preditiva não está habilitada na configuração do VitalFlow.")
    if 'equipment' not in digital_twin.multi_scale_models:
        raise HTTPException(status_code=503, detail="Modelo Digital Twin de equipamento não treinado ou carregado.")

    try:
        num_features_per_step = len(equipment_data.sensor_data) // equipment_data.sequence_length
        sensor_data_np = np.array(equipment_data.sensor_data).reshape(1, equipment_data.sequence_length, num_features_per_step)
        
        failure_prob = digital_twin.predict_equipment_failure(sensor_data_np, equipment_data.equipment_id)
        
        recommendation = "Monitore o equipamento regularmente."
        if failure_prob > 0.7:
            recommendation = "CRÍTICO: Alta probabilidade de falha. Agende manutenção imediata."
        elif failure_prob > 0.4:
            recommendation = "ALTO: Risco aumentado de falha. Planeje a manutenção em breve."

        # APRIMORAMENTO V6: Gera ação prescritiva do Digital Twin
        prescriptive_action_output: PrescriptiveAction = digital_twin.prescribe_action_from_twin_state(
            current_twin_state=sensor_data_np, # Estado do twin para o DT (mock)
            target_entity_id=equipment_data.equipment_id
        )

        return EquipmentFailurePrediction(
            equipment_id=equipment_data.equipment_id,
            failure_probability=failure_prob,
            recommendation=recommendation,
            confidence=0.85, # Placeholder
            prescriptive_action=PrescriptiveActionResponse(**prescriptive_action_output.__dict__)
        )
    except Exception as e:
        logger.error(f"Erro ao predizer falha de equipamento: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro de predição de falha de equipamento: {str(e)}")

@app.post("/gnn/analyze_network", response_model=NetworkAnalysisReportResponse)
async def analyze_hospital_network(request: NetworkAnalysisRequest):
    """
    Analisa a rede interna do hospital (ex: transferências de pacientes, interações da equipe)
    usando Redes Neurais Gráficas para identificar riscos, com referências ao MKG.
    """
    if not gnn_module.is_trained:
        raise HTTPException(status_code=503, detail="Módulo GNN não treinado ou carregado.")

    try:
        report_data = gnn_module.analyze_network_for_risks(request.graph_snapshot, request.analysis_type)
        
        return NetworkAnalysisReportResponse(
            analysis_type=report_data.get("analysis_type", request.analysis_type),
            identified_risks=report_data.get("identified_risks", []),
            recommendations=report_data.get("recommendations", []),
            timestamp=datetime.now().isoformat(),
            knowledge_references=[] # O GNN Module já pode retornar isso no report_data
        )
    except Exception as e:
        logger.error(f"Erro ao analisar rede hospitalar com GNN: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro de análise de rede: {str(e)}")

@app.get("/rl/formal_verification_reports", response_model=List[FormalVerificationReportSummary])
async def get_formal_verification_reports():
    """
    Recupera os relatórios de verificação formal para a política de Aprendizado por Reforço,
    garantindo a adesão às propriedades de segurança, equidade e ética.
    """
    if not vitalflow_config.FORMAL_VERIFICATION_ENABLED:
        raise HTTPException(status_code=400, detail="A Verificação Formal não está habilitada na configuração do VitalFlow.")
    
    if not rl_optimizer.formal_verification_reports:
        raise HTTPException(status_code=404, detail="Nenhum relatório de verificação formal encontrado.")
    
    return [FormalVerificationReportSummary(**report.__dict__) for report in rl_optimizer.formal_verification_reports]

@app.get("/federated_learning/status")
async def get_federated_learning_status():
    """
    Fornece o status atual do Orquestrador de Aprendizado Federado, incluindo relatórios de privacidade.
    """
    privacy_reports = [FormalVerificationReportSummary(**r.__dict__) for r in federated_orchestrator.privacy_verification_reports]
    return {
        "federated_learning_enabled": vitalflow_config.FEDERATED_LEARNING_ENABLED,
        "orchestrator_active": federated_orchestrator.is_active,
        "last_round_info": "Não implementado nesta demonstração, mas mostraria detalhes da última rodada de FL.",
        "privacy_verification_reports": privacy_reports
    }

@app.post("/data/ingest_realtime", response_model=RealTimeDataIngestionResponse)
async def ingest_realtime_data(data_point: RealTimeDataPoint):
    """
    Ingere um ponto de dado multi-modal em tempo real, aplicando harmonização e pré-processamento.
    """
    try:
        harmonized_data = await rtdi_manager.ingest_data_point(data_point.source_type, data_point.data)
        
        # APRIMORAMENTO V6: Aciona adaptação contínua para Digital Twin com os novos dados
        # Isso é conceitual, pois o DT precisaria de um método 'adapt'
        # await digital_twin.update_real_time_data(harmonized_data, data_point.source_type)
        
        # APRIMORAMENTO V6: Aciona monitoramento de drift para modelos relevantes
        # cl_manager.monitor_and_adapt("digital_twin", pd.DataFrame([harmonized_data])) # Mock
        
        return RealTimeDataIngestionResponse(
            status="success",
            harmonized_data_preview=harmonized_data,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Erro na ingestão de dados em tempo real: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro na ingestão de dados: {str(e)}")

@app.post("/xai/explain_decision", response_model=XAIExplanationResponse)
async def explain_decision(model_name: str = Body(...), data_point: Dict[str, Any] = Body(...), prediction: Any = Body(...)):
    """
    Gera uma explicação multi-nível e interativa para uma decisão ou predição de um modelo específico.
    """
    try:
        explanation = xai_manager.explain_decision(model_name, data_point, prediction)
        return XAIExplanationResponse(**explanation)
    except Exception as e:
        logger.error(f"Erro ao gerar explicação XAI: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro XAI: {str(e)}")

@app.post("/llm/query", response_model=LLMQueryResponse)
async def llm_query(request: LLMQueryRequest):
    """
    Permite interagir com o assistente médico de IA via linguagem natural.
    """
    try:
        response = await llm_assistant.process_natural_language_query(request.query, {"patient_id": request.patient_id, **(request.context or {})})
        return LLMQueryResponse(**response)
    except Exception as e:
        logger.error(f"Erro na consulta LLM: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro LLM: {str(e)}")

@app.post("/knowledge_graph/query", response_model=KnowledgeGraphQueryResponse)
async def knowledge_graph_query(request: KnowledgeGraphQueryRequest):
    """
    Consulta o Grafo de Conhecimento Médico para obter informações ou validar conhecimento.
    """
    try:
        results = mkg_manager.query_knowledge(request.query)
        return KnowledgeGraphQueryResponse(query=request.query, results=results)
    except Exception as e:
        logger.error(f"Erro na consulta ao Grafo de Conhecimento: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro KG: {str(e)}")

@app.get("/continual_learning/status/{model_name}", response_model=ContinualLearningStatus)
async def get_continual_learning_status(model_name: str):
    """
    Obtém o status de aprendizado contínuo e detecção de drift para um modelo específico.
    """
    status = cl_manager.get_drift_status(model_name)
    return ContinualLearningStatus(
        model_name=model_name,
        drift_status=status.get("drift_status", "Not Monitored"),
        last_drift_score=status.get("last_drift_score"),
        last_adaptation_time=None # Placeholder
    )

@app.post("/hpc/accelerate", response_model=HPCAccelerationResponse)
async def accelerate_hpc_task(request: HPCAccelerationRequest):
    """
    Acelera uma tarefa de otimização ou simulação complexa usando HPC ou otimização inspirada em quantum.
    """
    try:
        if request.problem_type == "optimization":
            result = await hpc_accelerator.accelerate_optimization(request.problem_description)
            return HPCAccelerationResponse(
                status="success",
                optimized_solution=result.get("optimized_solution"),
                method=result.get("method", "HPC"),
                speed_up_factor=result.get("speed_up_factor")
            )
        elif request.problem_type == "simulation":
            result = await hpc_accelerator.run_large_scale_simulation(request.problem_description)
            return HPCAccelerationResponse(
                status="success",
                simulation_results=result.get("simulation_results"),
                method=result.get("method", "HPC"),
                speed_up_factor=result.get("speed_up_factor")
            )
        else:
            raise HTTPException(status_code=400, detail="Tipo de problema não suportado para aceleração HPC.")
    except Exception as e:
        logger.error(f"Erro na aceleração HPC: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro HPC: {str(e)}")

@app.get("/models/status")
async def get_model_status():
    """Obtém o status de todos os modelos VitalFlow AI v6.0 e módulos avançados."""
    return {
        "causal_ai_trained": causal_ai.is_trained,
        "digital_twin_trained": digital_twin.is_trained,
        "rl_optimizer_trained": rl_optimizer.is_trained,
        "gnn_module_trained": gnn_module.is_trained,
        "federated_learning_orchestrator_active": federated_orchestrator.is_active,
        "mkg_manager_loaded": bool(mkg_manager.knowledge_graph),
        "rtdi_manager_initialized": bool(rtdi_manager.data_streams),
        "cl_manager_initialized": True,
        "hpc_accelerator_initialized": True,
        "xai_manager_initialized": True,
        "llm_assistant_initialized": True,
        "server_status": "ativo",
        "last_status_check": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True, # Recarrega para desenvolvimento, desabilite em produção
        log_level="info"
    )