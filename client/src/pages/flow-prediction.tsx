import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  Workflow, 
  TrendingUp, 
  Users, 
  Clock,
  Activity,
  Brain,
  Target,
  AlertTriangle,
  RefreshCw,
  Calendar,
  BarChart3,
  LineChart
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import PatientFlowChart from "@/components/charts/patient-flow-chart";

export default function FlowPrediction() {
  const { data: patientFlow, isLoading: flowLoading } = useQuery({
    queryKey: ["/api/predictions/patient-flow"],
    queryFn: () => api.getPatientFlow(),
    refetchInterval: 30000 // Atualizar a cada 30 segundos
  });

  const hourlyPredictions = [
    { hour: "08:00", admissions: 8, discharges: 3, netFlow: 5, confidence: 92 },
    { hour: "09:00", admissions: 12, discharges: 5, netFlow: 7, confidence: 89 },
    { hour: "10:00", admissions: 15, discharges: 7, netFlow: 8, confidence: 94 },
    { hour: "11:00", admissions: 18, discharges: 12, netFlow: 6, confidence: 87 },
    { hour: "12:00", admissions: 22, discharges: 15, netFlow: 7, confidence: 91 },
    { hour: "13:00", admissions: 25, discharges: 18, netFlow: 7, confidence: 88 },
    { hour: "14:00", admissions: 20, discharges: 22, netFlow: -2, confidence: 93 },
    { hour: "15:00", admissions: 16, discharges: 25, netFlow: -9, confidence: 90 }
  ];

  const getFlowColor = (netFlow: number) => {
    if (netFlow > 5) return "bg-destructive/10 text-destructive border-destructive/30";
    if (netFlow > 0) return "bg-warning/10 text-warning border-warning/30";
    return "bg-medical-primary/10 text-medical-primary border-medical-primary/30";
  };

  const getFlowStatus = (netFlow: number) => {
    if (netFlow > 5) return "Crítico";
    if (netFlow > 0) return "Moderado";
    return "Normal";
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold medical-text">Predições de Fluxo LSTM</h1>
          <p className="text-muted-foreground">Previsões neurais avançadas para fluxo de pacientes</p>
        </div>
        
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-xs">
            <Brain className="h-3 w-3 mr-1" />
            LSTM Neural Network
          </Badge>
          <Badge variant="outline" className="text-xs">
            <Activity className="h-3 w-3 mr-1" />
            Tempo Real
          </Badge>
          <Button size="sm" variant="outline">
            <RefreshCw className="h-4 w-4 mr-2" />
            Atualizar
          </Button>
        </div>
      </div>

      {/* Métricas de Fluxo */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="medical-card border-medical-primary/30">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <Users className="h-5 w-5 text-medical-primary" />
              <Badge variant="outline" className="text-xs">Próximas 24h</Badge>
            </div>
            <p className="text-2xl font-bold text-medical-primary">187</p>
            <p className="text-sm text-muted-foreground">Admissões Previstas</p>
          </CardContent>
        </Card>

        <Card className="medical-card border-medical-secondary/30">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <TrendingUp className="h-5 w-5 text-medical-secondary" />
              <Badge variant="outline" className="text-xs">Próximas 24h</Badge>
            </div>
            <p className="text-2xl font-bold text-medical-secondary">142</p>
            <p className="text-sm text-muted-foreground">Altas Previstas</p>
          </CardContent>
        </Card>

        <Card className="medical-card border-warning/30">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <Workflow className="h-5 w-5 text-warning" />
              <Badge variant="secondary" className="text-xs">Net Flow</Badge>
            </div>
            <p className="text-2xl font-bold text-warning">+45</p>
            <p className="text-sm text-muted-foreground">Fluxo Líquido</p>
          </CardContent>
        </Card>

        <Card className="medical-card border-chart-4/30">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <Target className="h-5 w-5 text-chart-4" />
              <Badge variant="outline" className="text-xs">IA</Badge>
            </div>
            <p className="text-2xl font-bold text-chart-4">91%</p>
            <p className="text-sm text-muted-foreground">Precisão LSTM</p>
          </CardContent>
        </Card>
      </div>

      {/* Gráfico Principal */}
      <PatientFlowChart data={patientFlow} isLoading={flowLoading} />

      {/* Predições Horárias */}
      <Card className="medical-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            Predições Horárias Detalhadas
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {hourlyPredictions.map((prediction, index) => (
              <div key={index} className={`p-4 rounded-lg border-2 ${getFlowColor(prediction.netFlow)}`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="text-center">
                      <p className="text-sm font-medium">{prediction.hour}</p>
                      <p className="text-xs text-muted-foreground">Horário</p>
                    </div>
                    
                    <div className="text-center">
                      <p className="text-lg font-bold text-medical-primary">{prediction.admissions}</p>
                      <p className="text-xs text-muted-foreground">Admissões</p>
                    </div>
                    
                    <div className="text-center">
                      <p className="text-lg font-bold text-medical-secondary">{prediction.discharges}</p>
                      <p className="text-xs text-muted-foreground">Altas</p>
                    </div>
                    
                    <div className="text-center">
                      <p className={`text-lg font-bold ${prediction.netFlow > 0 ? 'text-warning' : 'text-medical-primary'}`}>
                        {prediction.netFlow > 0 ? '+' : ''}{prediction.netFlow}
                      </p>
                      <p className="text-xs text-muted-foreground">Fluxo</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-3">
                    <Badge variant={prediction.netFlow > 5 ? "destructive" : prediction.netFlow > 0 ? "secondary" : "outline"}>
                      {getFlowStatus(prediction.netFlow)}
                    </Badge>
                    <div className="text-right">
                      <p className="text-sm font-medium">{prediction.confidence}%</p>
                      <p className="text-xs text-muted-foreground">Confiança</p>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Insights e Recomendações */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="medical-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              Insights da IA LSTM
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="p-3 bg-warning/10 rounded-lg border border-warning/30">
                <div className="flex items-center gap-2 mb-2">
                  <AlertTriangle className="h-4 w-4 text-warning" />
                  <span className="font-medium">Pico de Demanda Detectado</span>
                </div>
                <p className="text-sm text-muted-foreground">
                  LSTM prevê aumento de 40% nas admissões entre 10h-13h
                </p>
              </div>
              
              <div className="p-3 bg-medical-primary/10 rounded-lg border border-medical-primary/30">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="h-4 w-4 text-medical-primary" />
                  <span className="font-medium">Padrão Sazonal</span>
                </div>
                <p className="text-sm text-muted-foreground">
                  Identificado padrão recorrente de segunda-feira
                </p>
              </div>
              
              <div className="p-3 bg-chart-4/10 rounded-lg border border-chart-4/30">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="h-4 w-4 text-chart-4" />
                  <span className="font-medium">Precisão Melhorada</span>
                </div>
                <p className="text-sm text-muted-foreground">
                  Modelo LSTM atingiu 91% de precisão nas últimas 48h
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="medical-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Workflow className="h-5 w-5" />
              Recomendações Automatizadas
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-destructive rounded-full mt-2" />
                <div>
                  <p className="font-medium text-sm">Escalar Equipe</p>
                  <p className="text-xs text-muted-foreground">
                    Aumentar staff médico em 25% durante pico (10h-13h)
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-warning rounded-full mt-2" />
                <div>
                  <p className="font-medium text-sm">Preparar Leitos</p>
                  <p className="text-xs text-muted-foreground">
                    Liberar 15 leitos adicionais para emergências
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-medical-primary rounded-full mt-2" />
                <div>
                  <p className="font-medium text-sm">Otimizar Fluxo</p>
                  <p className="text-xs text-muted-foreground">
                    Acelerar processo de altas no período da tarde
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-chart-4 rounded-full mt-2" />
                <div>
                  <p className="font-medium text-sm">Recursos Extras</p>
                  <p className="text-xs text-muted-foreground">
                    Ativar protocolo de contingência se necessário
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}