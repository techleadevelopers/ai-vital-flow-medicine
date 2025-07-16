import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { 
  Target, 
  AlertTriangle, 
  Brain, 
  Heart,
  Activity,
  TrendingUp,
  TrendingDown,
  Users,
  Clock,
  Shield,
  Zap
} from "lucide-react";

export default function RiskAnalysis() {
  const riskPatients = [
    {
      id: "P001",
      name: "Maria Santos",
      age: 67,
      riskScore: 89,
      category: "Crítico",
      factors: ["Diabetes", "Hipertensão", "Idade"],
      lastUpdate: "2 min atrás",
      trend: "up"
    },
    {
      id: "P002", 
      name: "João Silva",
      age: 54,
      riskScore: 76,
      category: "Alto",
      factors: ["Cardiopatia", "Obesidade"],
      lastUpdate: "5 min atrás",
      trend: "stable"
    },
    {
      id: "P003",
      name: "Ana Costa",
      age: 41,
      riskScore: 65,
      category: "Médio",
      factors: ["Histórico familiar"],
      lastUpdate: "1 min atrás",
      trend: "down"
    }
  ];

  const getRiskColor = (score: number) => {
    if (score >= 80) return "bg-destructive/10 text-destructive border-destructive/30";
    if (score >= 60) return "bg-warning/10 text-warning border-warning/30";
    return "bg-medical-primary/10 text-medical-primary border-medical-primary/30";
  };

  const getRiskLevel = (score: number) => {
    if (score >= 80) return "Crítico";
    if (score >= 60) return "Alto";
    return "Médio";
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold medical-text">Análise de Risco IA</h1>
          <p className="text-muted-foreground">Predições neurais avançadas para prevenção</p>
        </div>
        
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-xs">
            <Brain className="h-3 w-3 mr-1" />
            Neural Network
          </Badge>
          <Badge variant="outline" className="text-xs">
            <Zap className="h-3 w-3 mr-1" />
            Tempo Real
          </Badge>
        </div>
      </div>

      {/* Métricas de Risco */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="medical-card border-destructive/30">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <AlertTriangle className="h-5 w-5 text-destructive" />
              <Badge variant="destructive" className="text-xs">Crítico</Badge>
            </div>
            <p className="text-2xl font-bold text-destructive">23</p>
            <p className="text-sm text-muted-foreground">Pacientes Alto Risco</p>
          </CardContent>
        </Card>

        <Card className="medical-card border-warning/30">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <Target className="h-5 w-5 text-warning" />
              <Badge variant="secondary" className="text-xs">Médio</Badge>
            </div>
            <p className="text-2xl font-bold text-warning">45</p>
            <p className="text-sm text-muted-foreground">Risco Moderado</p>
          </CardContent>
        </Card>

        <Card className="medical-card border-medical-primary/30">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <Shield className="h-5 w-5 text-medical-primary" />
              <Badge variant="outline" className="text-xs">Baixo</Badge>
            </div>
            <p className="text-2xl font-bold text-medical-primary">189</p>
            <p className="text-sm text-muted-foreground">Risco Baixo</p>
          </CardContent>
        </Card>

        <Card className="medical-card border-chart-4/30">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <Brain className="h-5 w-5 text-chart-4" />
              <Badge variant="outline" className="text-xs">95%</Badge>
            </div>
            <p className="text-2xl font-bold text-chart-4">AI</p>
            <p className="text-sm text-muted-foreground">Precisão Modelo</p>
          </CardContent>
        </Card>
      </div>

      {/* Lista de Pacientes de Risco */}
      <Card className="medical-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Pacientes de Alto Risco
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {riskPatients.map((patient) => (
              <div key={patient.id} className={`p-4 rounded-lg border-2 ${getRiskColor(patient.riskScore)}`}>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-medical-primary/10 rounded-full flex items-center justify-center">
                      <Users className="h-5 w-5 text-medical-primary" />
                    </div>
                    <div>
                      <h3 className="font-semibold">{patient.name}</h3>
                      <p className="text-sm text-muted-foreground">{patient.id} • {patient.age} anos</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <Badge variant={patient.riskScore >= 80 ? "destructive" : patient.riskScore >= 60 ? "secondary" : "outline"}>
                      {getRiskLevel(patient.riskScore)}
                    </Badge>
                    {patient.trend === "up" && <TrendingUp className="h-4 w-4 text-destructive" />}
                    {patient.trend === "down" && <TrendingDown className="h-4 w-4 text-medical-primary" />}
                    {patient.trend === "stable" && <Activity className="h-4 w-4 text-muted-foreground" />}
                  </div>
                </div>

                <div className="space-y-3">
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-medium">Score de Risco</span>
                      <span className="text-sm font-bold">{patient.riskScore}%</span>
                    </div>
                    <Progress value={patient.riskScore} className="h-2" />
                  </div>

                  <div>
                    <p className="text-sm font-medium mb-1">Fatores de Risco:</p>
                    <div className="flex flex-wrap gap-1">
                      {patient.factors.map((factor, index) => (
                        <Badge key={index} variant="outline" className="text-xs">
                          {factor}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      {patient.lastUpdate}
                    </span>
                    <Button size="sm" variant="outline">
                      Ver Detalhes
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Insights da IA */}
      <Card className="medical-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Insights da IA Neural
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="font-semibold">Padrões Identificados:</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-destructive rounded-full" />
                  Aumento de 15% em pacientes diabéticos
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-warning rounded-full" />
                  Correlação entre idade e complicações
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-medical-primary rounded-full" />
                  Eficácia de 89% nas intervenções precoces
                </li>
              </ul>
            </div>
            
            <div className="space-y-4">
              <h4 className="font-semibold">Recomendações:</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-chart-4 rounded-full" />
                  Intensificar monitoramento de pacientes acima de 60 anos
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-chart-2 rounded-full" />
                  Implementar protocolos preventivos
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-chart-3 rounded-full" />
                  Otimizar alocação de recursos
                </li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}