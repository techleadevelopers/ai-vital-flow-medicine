import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { 
  Brain, 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  Target, 
  Activity, 
  Users,
  Heart,
  Zap,
  Eye,
  RefreshCw,
  CheckCircle,
  XCircle,
  Clock,
  BarChart3,
  LineChart,
  PieChart,
  Workflow
} from "lucide-react";
import { useState, useEffect } from "react";

interface PredictionData {
  id: string;
  type: 'risk' | 'flow' | 'resource' | 'anomaly';
  title: string;
  value: string;
  confidence: number;
  status: 'active' | 'warning' | 'critical' | 'resolved';
  trend: 'up' | 'down' | 'stable';
  description: string;
  lastUpdated: string;
  accuracy: number;
  recommendations: string[];
}

export default function AIPredictionsAdvanced() {
  const [predictions, setPredictions] = useState<PredictionData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedView, setSelectedView] = useState<'all' | 'risk' | 'flow' | 'resource' | 'anomaly'>('all');

  useEffect(() => {
    // Simular dados de predições da IA em tempo real
    const mockPredictions: PredictionData[] = [
      {
        id: '1',
        type: 'risk',
        title: 'Risco de Deterioração',
        value: '23 pacientes',
        confidence: 94,
        status: 'warning',
        trend: 'up',
        description: 'Modelo neural detectou aumento de risco em 23 pacientes',
        lastUpdated: '2 min atrás',
        accuracy: 94,
        recommendations: ['Aumentar monitoramento', 'Revisar medicação', 'Considerar transferência UTI']
      },
      {
        id: '2',
        type: 'flow',
        title: 'Fluxo de Admissões',
        value: '45 admissões',
        confidence: 87,
        status: 'active',
        trend: 'up',
        description: 'LSTM prevê pico de admissões nas próximas 6 horas',
        lastUpdated: '1 min atrás',
        accuracy: 87,
        recommendations: ['Preparar leitos extras', 'Escalar equipe', 'Otimizar fluxo']
      },
      {
        id: '3',
        type: 'resource',
        title: 'Otimização de Leitos',
        value: '12 realocações',
        confidence: 91,
        status: 'active',
        trend: 'stable',
        description: 'IA recomenda 12 realocações para otimizar ocupação',
        lastUpdated: '30 seg atrás',
        accuracy: 91,
        recommendations: ['Transferir pac. UTI→Geral', 'Liberar leitos cirúrgicos', 'Preparar alta']
      },
      {
        id: '4',
        type: 'anomaly',
        title: 'Anomalias Detectadas',
        value: '3 anomalias',
        confidence: 89,
        status: 'critical',
        trend: 'down',
        description: 'Padrões anômalos detectados em sinais vitais',
        lastUpdated: '45 seg atrás',
        accuracy: 89,
        recommendations: ['Investigar padrões', 'Verificar equipamentos', 'Alerta médico']
      },
      {
        id: '5',
        type: 'risk',
        title: 'Predição de Mortalidade',
        value: '2 casos críticos',
        confidence: 96,
        status: 'critical',
        trend: 'stable',
        description: 'Modelo ensemble identifica 2 casos de alto risco',
        lastUpdated: '1 min atrás',
        accuracy: 96,
        recommendations: ['Intervenção imediata', 'Equipe especializada', 'Monitoramento 24/7']
      },
      {
        id: '6',
        type: 'flow',
        title: 'Tempo de Espera',
        value: '18 min médio',
        confidence: 83,
        status: 'active',
        trend: 'down',
        description: 'IA prevê redução no tempo de espera',
        lastUpdated: '3 min atrás',
        accuracy: 83,
        recommendations: ['Otimizar triagem', 'Acelerar processos', 'Distribuir carga']
      }
    ];

    setTimeout(() => {
      setPredictions(mockPredictions);
      setIsLoading(false);
    }, 1000);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'critical': return 'bg-destructive/10 text-destructive border-destructive/30';
      case 'warning': return 'bg-warning/10 text-warning border-warning/30';
      case 'active': return 'bg-medical-primary/10 text-medical-primary border-medical-primary/30';
      case 'resolved': return 'bg-medical-secondary/10 text-medical-secondary border-medical-secondary/30';
      default: return 'bg-muted/10 text-muted-foreground border-muted/30';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'critical': return <XCircle className="h-4 w-4" />;
      case 'warning': return <AlertTriangle className="h-4 w-4" />;
      case 'active': return <CheckCircle className="h-4 w-4" />;
      case 'resolved': return <CheckCircle className="h-4 w-4" />;
      default: return <Clock className="h-4 w-4" />;
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'risk': return <Target className="h-4 w-4" />;
      case 'flow': return <Workflow className="h-4 w-4" />;
      case 'resource': return <BarChart3 className="h-4 w-4" />;
      case 'anomaly': return <Eye className="h-4 w-4" />;
      default: return <Activity className="h-4 w-4" />;
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <TrendingUp className="h-3 w-3 text-medical-secondary" />;
      case 'down': return <TrendingDown className="h-3 w-3 text-destructive" />;
      default: return <Activity className="h-3 w-3 text-muted-foreground" />;
    }
  };

  const filteredPredictions = selectedView === 'all' 
    ? predictions 
    : predictions.filter(p => p.type === selectedView);

  if (isLoading) {
    return (
      <Card className="medical-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-medical-primary" />
            Predições IA Avançadas
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="h-20 bg-muted/20 rounded-lg animate-pulse" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="medical-card">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="h-6 w-6 text-medical-primary" />
            <div>
              <CardTitle className="text-lg">Predições IA Avançadas</CardTitle>
              <p className="text-sm text-muted-foreground">
                Neural Networks + LSTM + Ensemble Learning
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs">
              <Zap className="h-3 w-3 mr-1" />
              Tempo Real
            </Badge>
            <Button size="sm" variant="ghost" onClick={() => window.location.reload()}>
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Filtros */}
        <div className="flex flex-wrap gap-2 mt-4">
          {[
            { key: 'all', label: 'Todas', icon: Activity },
            { key: 'risk', label: 'Risco', icon: Target },
            { key: 'flow', label: 'Fluxo', icon: Workflow },
            { key: 'resource', label: 'Recursos', icon: BarChart3 },
            { key: 'anomaly', label: 'Anomalias', icon: Eye }
          ].map(({ key, label, icon: Icon }) => (
            <Button
              key={key}
              size="sm"
              variant={selectedView === key ? "default" : "outline"}
              onClick={() => setSelectedView(key as any)}
              className="text-xs"
            >
              <Icon className="h-3 w-3 mr-1" />
              {label}
            </Button>
          ))}
        </div>
      </CardHeader>

      <CardContent>
        <div className="space-y-4">
          {filteredPredictions.map((prediction) => (
            <div key={prediction.id} className={`p-4 rounded-lg border-2 ${getStatusColor(prediction.status)}`}>
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2">
                    {getTypeIcon(prediction.type)}
                    <h3 className="font-semibold">{prediction.title}</h3>
                  </div>
                  <Badge variant="outline" className="text-xs">
                    {prediction.type.toUpperCase()}
                  </Badge>
                </div>
                
                <div className="flex items-center gap-2">
                  {getTrendIcon(prediction.trend)}
                  {getStatusIcon(prediction.status)}
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-2xl font-bold">{prediction.value}</span>
                    <span className="text-xs text-muted-foreground">{prediction.lastUpdated}</span>
                  </div>
                  <p className="text-sm text-muted-foreground">{prediction.description}</p>
                  
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span>Confiança</span>
                      <span className="font-medium">{prediction.confidence}%</span>
                    </div>
                    <Progress value={prediction.confidence} className="h-2" />
                  </div>
                </div>

                <div className="space-y-2">
                  <h4 className="text-sm font-medium">Recomendações IA:</h4>
                  <ul className="space-y-1">
                    {prediction.recommendations.map((rec, index) => (
                      <li key={index} className="text-xs text-muted-foreground flex items-center gap-2">
                        <div className="w-1 h-1 bg-medical-primary rounded-full" />
                        {rec}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Resumo das Predições */}
        <div className="mt-6 p-4 bg-muted/30 rounded-lg">
          <h4 className="font-semibold text-medical-primary mb-3 flex items-center gap-2">
            <Brain className="h-4 w-4" />
            Resumo de Performance da IA
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="text-center">
              <p className="text-lg font-bold text-medical-primary">
                {predictions.filter(p => p.status === 'active').length}
              </p>
              <p className="text-muted-foreground">Predições Ativas</p>
            </div>
            <div className="text-center">
              <p className="text-lg font-bold text-warning">
                {predictions.filter(p => p.status === 'warning').length}
              </p>
              <p className="text-muted-foreground">Alertas</p>
            </div>
            <div className="text-center">
              <p className="text-lg font-bold text-destructive">
                {predictions.filter(p => p.status === 'critical').length}
              </p>
              <p className="text-muted-foreground">Críticas</p>
            </div>
            <div className="text-center">
              <p className="text-lg font-bold text-chart-4">
                {Math.round(predictions.reduce((acc, p) => acc + p.accuracy, 0) / predictions.length)}%
              </p>
              <p className="text-muted-foreground">Precisão Média</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}