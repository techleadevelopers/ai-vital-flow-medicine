import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { 
  AlertTriangle, 
  Eye,
  Brain,
  Activity,
  TrendingUp,
  TrendingDown,
  Clock,
  Target,
  Zap,
  Shield,
  Users,
  Heart,
  Monitor,
  Search,
  Filter,
  RefreshCw,
  CheckCircle,
  XCircle,
  AlertCircle
} from "lucide-react";
import { useState } from "react";

export default function Anomalies() {
  const [selectedSeverity, setSelectedSeverity] = useState<'all' | 'critical' | 'high' | 'medium' | 'low'>('all');
  const [selectedType, setSelectedType] = useState<'all' | 'vital_signs' | 'lab_results' | 'behavior' | 'system'>('all');

  const anomalies = [
    {
      id: "A001",
      type: "vital_signs",
      severity: "critical",
      status: "active",
      patientId: "P2341",
      patientName: "Carlos Silva",
      title: "Frequência Cardíaca Irregular",
      description: "Padrão anômalo detectado na frequência cardíaca - possível arritmia",
      detectedAt: "2024-01-15T14:23:00Z",
      aiModel: "Neural Network",
      confidence: 96,
      currentValue: "FC: 156 bpm",
      normalRange: "60-100 bpm",
      duration: "15 min",
      actionTaken: "Médico notificado",
      recommendation: "Avaliação cardiológica imediata"
    },
    {
      id: "A002",
      type: "lab_results",
      severity: "high",
      status: "investigating",
      patientId: "P2342",
      patientName: "Maria Santos",
      title: "Valores Laboratoriais Anômalos",
      description: "Combinação incomum de marcadores inflamatórios",
      detectedAt: "2024-01-15T13:45:00Z",
      aiModel: "Ensemble Learning",
      confidence: 89,
      currentValue: "PCR: 45 mg/L",
      normalRange: "< 3 mg/L",
      duration: "2 horas",
      actionTaken: "Coleta repetida",
      recommendation: "Investigação adicional necessária"
    },
    {
      id: "A003",
      type: "behavior",
      severity: "medium",
      status: "monitored",
      patientId: "P2343",
      patientName: "João Oliveira",
      title: "Padrão de Movimento Incomum",
      description: "Atividade motora reduzida comparada ao baseline",
      detectedAt: "2024-01-15T12:30:00Z",
      aiModel: "LSTM",
      confidence: 78,
      currentValue: "Mobilidade: 15%",
      normalRange: "70-90%",
      duration: "4 horas",
      actionTaken: "Fisioterapeuta notificado",
      recommendation: "Avaliação neurológica"
    },
    {
      id: "A004",
      type: "system",
      severity: "low",
      status: "resolved",
      patientId: "P2344",
      patientName: "Ana Costa",
      title: "Falha no Monitoramento",
      description: "Perda intermitente de sinal do monitor",
      detectedAt: "2024-01-15T11:15:00Z",
      aiModel: "Anomaly Detection",
      confidence: 92,
      currentValue: "Sinal: 45%",
      normalRange: "95-100%",
      duration: "30 min",
      actionTaken: "Equipamento substituído",
      recommendation: "Manutenção preventiva"
    },
    {
      id: "A005",
      type: "vital_signs",
      severity: "high",
      status: "active",
      patientId: "P2345",
      patientName: "Pedro Ferreira",
      title: "Saturação de Oxigênio Baixa",
      description: "Queda progressiva da saturação de oxigênio",
      detectedAt: "2024-01-15T14:45:00Z",
      aiModel: "Deep Learning",
      confidence: 94,
      currentValue: "SpO2: 88%",
      normalRange: "95-100%",
      duration: "25 min",
      actionTaken: "Oxigenoterapia iniciada",
      recommendation: "Monitoramento contínuo"
    }
  ];

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-destructive text-destructive-foreground';
      case 'high': return 'bg-warning text-warning-foreground';
      case 'medium': return 'bg-medical-primary text-medical-primary-foreground';
      case 'low': return 'bg-medical-secondary text-medical-secondary-foreground';
      default: return 'bg-muted text-muted-foreground';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-destructive/10 text-destructive border-destructive/30';
      case 'investigating': return 'bg-warning/10 text-warning border-warning/30';
      case 'monitored': return 'bg-medical-primary/10 text-medical-primary border-medical-primary/30';
      case 'resolved': return 'bg-medical-secondary/10 text-medical-secondary border-medical-secondary/30';
      default: return 'bg-muted/10 text-muted-foreground border-muted/30';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'vital_signs': return <Heart className="h-4 w-4" />;
      case 'lab_results': return <Activity className="h-4 w-4" />;
      case 'behavior': return <Users className="h-4 w-4" />;
      case 'system': return <Monitor className="h-4 w-4" />;
      default: return <AlertTriangle className="h-4 w-4" />;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <AlertCircle className="h-4 w-4" />;
      case 'investigating': return <Eye className="h-4 w-4" />;
      case 'monitored': return <Shield className="h-4 w-4" />;
      case 'resolved': return <CheckCircle className="h-4 w-4" />;
      default: return <AlertTriangle className="h-4 w-4" />;
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return <AlertTriangle className="h-4 w-4" />;
      case 'high': return <TrendingUp className="h-4 w-4" />;
      case 'medium': return <Activity className="h-4 w-4" />;
      case 'low': return <TrendingDown className="h-4 w-4" />;
      default: return <Activity className="h-4 w-4" />;
    }
  };

  const filteredAnomalies = anomalies.filter(anomaly => {
    const severityMatch = selectedSeverity === 'all' || anomaly.severity === selectedSeverity;
    const typeMatch = selectedType === 'all' || anomaly.type === selectedType;
    return severityMatch && typeMatch;
  });

  const getTimeDifference = (dateString: string) => {
    const now = new Date();
    const date = new Date(dateString);
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 60) return `${diffMins} min atrás`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)} h atrás`;
    return `${Math.floor(diffMins / 1440)} dias atrás`;
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold medical-text">Detecção de Anomalias IA</h1>
          <p className="text-muted-foreground">Sistema inteligente de detecção de padrões anômalos</p>
        </div>
        
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-xs">
            <Brain className="h-3 w-3 mr-1" />
            AI Powered
          </Badge>
          <Badge variant="outline" className="text-xs">
            <Zap className="h-3 w-3 mr-1" />
            Tempo Real
          </Badge>
          <Button size="sm" variant="outline">
            <Search className="h-4 w-4 mr-2" />
            Buscar
          </Button>
          <Button size="sm" variant="outline">
            <Filter className="h-4 w-4 mr-2" />
            Filtros
          </Button>
          <Button size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Atualizar
          </Button>
        </div>
      </div>

      {/* Estatísticas */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <Card className="medical-card border-destructive/30">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <AlertTriangle className="h-5 w-5 text-destructive" />
              <Badge variant="destructive" className="text-xs">CRÍTICO</Badge>
            </div>
            <p className="text-2xl font-bold text-destructive">
              {anomalies.filter(a => a.severity === 'critical').length}
            </p>
            <p className="text-sm text-muted-foreground">Anomalias Críticas</p>
          </CardContent>
        </Card>

        <Card className="medical-card border-warning/30">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <TrendingUp className="h-5 w-5 text-warning" />
              <Badge variant="secondary" className="text-xs">ALTO</Badge>
            </div>
            <p className="text-2xl font-bold text-warning">
              {anomalies.filter(a => a.severity === 'high').length}
            </p>
            <p className="text-sm text-muted-foreground">Alta Prioridade</p>
          </CardContent>
        </Card>

        <Card className="medical-card border-medical-primary/30">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <Activity className="h-5 w-5 text-medical-primary" />
              <Badge variant="outline" className="text-xs">ATIVO</Badge>
            </div>
            <p className="text-2xl font-bold text-medical-primary">
              {anomalies.filter(a => a.status === 'active').length}
            </p>
            <p className="text-sm text-muted-foreground">Casos Ativos</p>
          </CardContent>
        </Card>

        <Card className="medical-card border-medical-secondary/30">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <CheckCircle className="h-5 w-5 text-medical-secondary" />
              <Badge variant="outline" className="text-xs">RESOLVIDO</Badge>
            </div>
            <p className="text-2xl font-bold text-medical-secondary">
              {anomalies.filter(a => a.status === 'resolved').length}
            </p>
            <p className="text-sm text-muted-foreground">Resolvidos</p>
          </CardContent>
        </Card>

        <Card className="medical-card border-chart-4/30">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <Target className="h-5 w-5 text-chart-4" />
              <Badge variant="outline" className="text-xs">PRECISÃO</Badge>
            </div>
            <p className="text-2xl font-bold text-chart-4">89%</p>
            <p className="text-sm text-muted-foreground">Precisão IA</p>
          </CardContent>
        </Card>
      </div>

      {/* Filtros */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="flex items-center gap-2 p-4 bg-muted/20 rounded-lg">
          <span className="text-sm font-medium">Severidade:</span>
          <div className="flex gap-2">
            {[
              { key: 'all', label: 'Todas' },
              { key: 'critical', label: 'Crítica' },
              { key: 'high', label: 'Alta' },
              { key: 'medium', label: 'Média' },
              { key: 'low', label: 'Baixa' }
            ].map(({ key, label }) => (
              <Button
                key={key}
                size="sm"
                variant={selectedSeverity === key ? "default" : "outline"}
                onClick={() => setSelectedSeverity(key as any)}
              >
                {label}
              </Button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2 p-4 bg-muted/20 rounded-lg">
          <span className="text-sm font-medium">Tipo:</span>
          <div className="flex gap-2">
            {[
              { key: 'all', label: 'Todos' },
              { key: 'vital_signs', label: 'Sinais Vitais' },
              { key: 'lab_results', label: 'Laboratório' },
              { key: 'behavior', label: 'Comportamento' },
              { key: 'system', label: 'Sistema' }
            ].map(({ key, label }) => (
              <Button
                key={key}
                size="sm"
                variant={selectedType === key ? "default" : "outline"}
                onClick={() => setSelectedType(key as any)}
              >
                {label}
              </Button>
            ))}
          </div>
        </div>
      </div>

      {/* Lista de Anomalias */}
      <Card className="medical-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Anomalias Detectadas pela IA
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {filteredAnomalies.map((anomaly) => (
              <div key={anomaly.id} className={`p-4 rounded-lg border-2 ${getStatusColor(anomaly.status)}`}>
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 bg-medical-primary/10 rounded-lg flex items-center justify-center">
                      {getTypeIcon(anomaly.type)}
                    </div>
                    
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="font-semibold text-lg">{anomaly.title}</h3>
                        <Badge variant="outline" className="text-xs">
                          {anomaly.patientId}
                        </Badge>
                        <Badge className={`text-xs ${getSeverityColor(anomaly.severity)}`}>
                          {getSeverityIcon(anomaly.severity)}
                          <span className="ml-1">{anomaly.severity.toUpperCase()}</span>
                        </Badge>
                      </div>
                      
                      <p className="text-sm text-muted-foreground mb-3">{anomaly.description}</p>
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <p className="text-muted-foreground">Paciente</p>
                          <p className="font-medium">{anomaly.patientName}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Detectado</p>
                          <p className="font-medium">{getTimeDifference(anomaly.detectedAt)}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Duração</p>
                          <p className="font-medium">{anomaly.duration}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Modelo IA</p>
                          <p className="font-medium">{anomaly.aiModel}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1">
                      {getStatusIcon(anomaly.status)}
                      <span className="text-sm font-medium capitalize">
                        {anomaly.status.replace('_', ' ')}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  <div className="space-y-2">
                    <h4 className="font-medium text-sm">Valores Anômalos</h4>
                    <div className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span>Atual:</span>
                        <span className="font-medium text-destructive">{anomaly.currentValue}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Normal:</span>
                        <span className="font-medium">{anomaly.normalRange}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <h4 className="font-medium text-sm">Confiança da IA</h4>
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-sm">
                        <span>Precisão</span>
                        <span className="font-medium">{anomaly.confidence}%</span>
                      </div>
                      <Progress value={anomaly.confidence} className="h-2" />
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <h4 className="font-medium text-sm">Ação Tomada</h4>
                    <div className="space-y-1">
                      <div className="flex items-center gap-2 text-sm">
                        <CheckCircle className="h-4 w-4 text-medical-primary" />
                        <span>{anomaly.actionTaken}</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="space-y-2">
                  <h4 className="font-medium text-sm">Recomendação da IA</h4>
                  <p className="text-sm text-muted-foreground">{anomaly.recommendation}</p>
                </div>

                <div className="flex items-center justify-between pt-4 border-t">
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                      <Clock className="h-4 w-4 text-medical-primary" />
                      <span className="text-sm font-medium">
                        Duração: {anomaly.duration}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Brain className="h-4 w-4 text-medical-primary" />
                      <span className="text-sm font-medium">
                        Confiança: {anomaly.confidence}%
                      </span>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <Button size="sm" variant="outline">
                      <Eye className="h-4 w-4 mr-1" />
                      Detalhes
                    </Button>
                    {anomaly.status === 'active' && (
                      <Button size="sm">
                        <Shield className="h-4 w-4 mr-1" />
                        Investigar
                      </Button>
                    )}
                    {anomaly.status === 'investigating' && (
                      <Button size="sm">
                        <CheckCircle className="h-4 w-4 mr-1" />
                        Resolver
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}