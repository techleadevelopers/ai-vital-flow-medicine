import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  Ambulance, 
  AlertTriangle,
  Clock,
  Heart,
  Activity,
  Users,
  MapPin,
  Phone,
  Stethoscope,
  Zap,
  Target,
  Eye,
  RefreshCw,
  Plus,
  Filter,
  CheckCircle,
  XCircle
} from "lucide-react";
import { useState, useEffect } from "react";

export default function Emergency() {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [selectedPriority, setSelectedPriority] = useState<'all' | 'critical' | 'high' | 'medium' | 'low'>('all');

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const emergencyCases = [
    {
      id: "E001",
      patientName: "Carlos Mendes",
      age: 58,
      gender: "M",
      priority: "critical",
      status: "active",
      triageTime: "14:23",
      waitTime: "00:07",
      symptoms: ["Dor no peito", "Sudorese", "Falta de ar"],
      vitalSigns: {
        bloodPressure: "180/110",
        heartRate: 98,
        temperature: "36.8°C",
        oxygenSaturation: "94%"
      },
      assignedDoctor: "Dr. Ana Silva",
      location: "Sala 1",
      estimatedTime: "IMEDIATO"
    },
    {
      id: "E002",
      patientName: "Maria Santos",
      age: 34,
      gender: "F",
      priority: "high",
      status: "waiting",
      triageTime: "14:15",
      waitTime: "00:15",
      symptoms: ["Fratura exposta", "Dor severa"],
      vitalSigns: {
        bloodPressure: "130/85",
        heartRate: 110,
        temperature: "37.2°C",
        oxygenSaturation: "98%"
      },
      assignedDoctor: "Dr. Pedro Costa",
      location: "Aguardando",
      estimatedTime: "10 min"
    },
    {
      id: "E003",
      patientName: "João Silva",
      age: 45,
      gender: "M",
      priority: "medium",
      status: "in_treatment",
      triageTime: "13:45",
      waitTime: "00:45",
      symptoms: ["Cefaleia intensa", "Náusea"],
      vitalSigns: {
        bloodPressure: "140/90",
        heartRate: 85,
        temperature: "36.5°C",
        oxygenSaturation: "99%"
      },
      assignedDoctor: "Dr. Laura Mendes",
      location: "Sala 3",
      estimatedTime: "20 min"
    },
    {
      id: "E004",
      patientName: "Ana Costa",
      age: 28,
      gender: "F",
      priority: "low",
      status: "completed",
      triageTime: "13:30",
      waitTime: "01:15",
      symptoms: ["Corte superficial", "Limpeza"],
      vitalSigns: {
        bloodPressure: "120/80",
        heartRate: 75,
        temperature: "36.4°C",
        oxygenSaturation: "99%"
      },
      assignedDoctor: "Dr. Roberto Lima",
      location: "Sala 2",
      estimatedTime: "CONCLUÍDO"
    }
  ];

  const getPriorityColor = (priority: string) => {
    switch (priority) {
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
      case 'waiting': return 'bg-warning/10 text-warning border-warning/30';
      case 'in_treatment': return 'bg-medical-primary/10 text-medical-primary border-medical-primary/30';
      case 'completed': return 'bg-medical-secondary/10 text-medical-secondary border-medical-secondary/30';
      default: return 'bg-muted/10 text-muted-foreground border-muted/30';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <Zap className="h-4 w-4" />;
      case 'waiting': return <Clock className="h-4 w-4" />;
      case 'in_treatment': return <Stethoscope className="h-4 w-4" />;
      case 'completed': return <CheckCircle className="h-4 w-4" />;
      default: return <AlertTriangle className="h-4 w-4" />;
    }
  };

  const getPriorityIcon = (priority: string) => {
    switch (priority) {
      case 'critical': return <Heart className="h-4 w-4" />;
      case 'high': return <AlertTriangle className="h-4 w-4" />;
      case 'medium': return <Activity className="h-4 w-4" />;
      case 'low': return <Target className="h-4 w-4" />;
      default: return <Activity className="h-4 w-4" />;
    }
  };

  const filteredCases = selectedPriority === 'all' 
    ? emergencyCases 
    : emergencyCases.filter(c => c.priority === selectedPriority);

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold medical-text">Emergências</h1>
          <p className="text-muted-foreground">
            Central de Emergências - {currentTime.toLocaleTimeString()}
          </p>
        </div>
        
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline">
            <Filter className="h-4 w-4 mr-2" />
            Filtros
          </Button>
          <Button size="sm" variant="outline">
            <RefreshCw className="h-4 w-4 mr-2" />
            Atualizar
          </Button>
          <Button size="sm" className="bg-destructive hover:bg-destructive/90">
            <Plus className="h-4 w-4 mr-2" />
            Nova Emergência
          </Button>
        </div>
      </div>

      {/* Status Geral */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <Card className="medical-card border-destructive/30">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <Heart className="h-5 w-5 text-destructive" />
              <Badge variant="destructive" className="text-xs">CRÍTICO</Badge>
            </div>
            <p className="text-2xl font-bold text-destructive">
              {emergencyCases.filter(c => c.priority === 'critical').length}
            </p>
            <p className="text-sm text-muted-foreground">Casos Críticos</p>
          </CardContent>
        </Card>

        <Card className="medical-card border-warning/30">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <AlertTriangle className="h-5 w-5 text-warning" />
              <Badge variant="secondary" className="text-xs">ALTO</Badge>
            </div>
            <p className="text-2xl font-bold text-warning">
              {emergencyCases.filter(c => c.priority === 'high').length}
            </p>
            <p className="text-sm text-muted-foreground">Alta Prioridade</p>
          </CardContent>
        </Card>

        <Card className="medical-card border-medical-primary/30">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <Activity className="h-5 w-5 text-medical-primary" />
              <Badge variant="outline" className="text-xs">MÉDIO</Badge>
            </div>
            <p className="text-2xl font-bold text-medical-primary">
              {emergencyCases.filter(c => c.priority === 'medium').length}
            </p>
            <p className="text-sm text-muted-foreground">Prioridade Média</p>
          </CardContent>
        </Card>

        <Card className="medical-card border-medical-secondary/30">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <Target className="h-5 w-5 text-medical-secondary" />
              <Badge variant="outline" className="text-xs">BAIXO</Badge>
            </div>
            <p className="text-2xl font-bold text-medical-secondary">
              {emergencyCases.filter(c => c.priority === 'low').length}
            </p>
            <p className="text-sm text-muted-foreground">Baixa Prioridade</p>
          </CardContent>
        </Card>

        <Card className="medical-card border-chart-4/30">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <Clock className="h-5 w-5 text-chart-4" />
              <Badge variant="outline" className="text-xs">MÉDIA</Badge>
            </div>
            <p className="text-2xl font-bold text-chart-4">18 min</p>
            <p className="text-sm text-muted-foreground">Tempo Espera</p>
          </CardContent>
        </Card>
      </div>

      {/* Filtros de Prioridade */}
      <div className="flex items-center gap-2 p-4 bg-muted/20 rounded-lg">
        <span className="text-sm font-medium">Prioridade:</span>
        <div className="flex gap-2">
          {[
            { key: 'all', label: 'Todos' },
            { key: 'critical', label: 'Crítico' },
            { key: 'high', label: 'Alto' },
            { key: 'medium', label: 'Médio' },
            { key: 'low', label: 'Baixo' }
          ].map(({ key, label }) => (
            <Button
              key={key}
              size="sm"
              variant={selectedPriority === key ? "default" : "outline"}
              onClick={() => setSelectedPriority(key as any)}
            >
              {label}
            </Button>
          ))}
        </div>
      </div>

      {/* Lista de Casos */}
      <Card className="medical-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Ambulance className="h-5 w-5" />
            Casos Ativos de Emergência
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {filteredCases.map((case_) => (
              <div key={case_.id} className={`p-4 rounded-lg border-2 ${getStatusColor(case_.status)}`}>
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-start gap-4">
                    <div className="w-16 h-16 bg-medical-primary/10 rounded-lg flex items-center justify-center">
                      <Ambulance className="h-8 w-8 text-medical-primary" />
                    </div>
                    
                    <div>
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="font-semibold text-lg">{case_.patientName}</h3>
                        <Badge variant="outline" className="text-xs">
                          {case_.id}
                        </Badge>
                        <Badge className={`text-xs ${getPriorityColor(case_.priority)}`}>
                          {getPriorityIcon(case_.priority)}
                          <span className="ml-1">{case_.priority.toUpperCase()}</span>
                        </Badge>
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <p className="text-muted-foreground">Idade/Gênero</p>
                          <p className="font-medium">{case_.age} anos, {case_.gender}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Triagem</p>
                          <p className="font-medium">{case_.triageTime}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Tempo Espera</p>
                          <p className="font-medium">{case_.waitTime}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Estimativa</p>
                          <p className="font-medium">{case_.estimatedTime}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1">
                      {getStatusIcon(case_.status)}
                      <span className="text-sm font-medium capitalize">
                        {case_.status.replace('_', ' ')}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  <div className="space-y-2">
                    <h4 className="font-medium text-sm">Sintomas</h4>
                    <div className="space-y-1">
                      {case_.symptoms.map((symptom, index) => (
                        <div key={index} className="flex items-center gap-2 text-sm">
                          <div className="w-2 h-2 bg-medical-primary rounded-full" />
                          <span>{symptom}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <h4 className="font-medium text-sm">Sinais Vitais</h4>
                    <div className="space-y-1 text-sm">
                      <div className="flex justify-between">
                        <span>PA:</span>
                        <span className="font-medium">{case_.vitalSigns.bloodPressure}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>FC:</span>
                        <span className="font-medium">{case_.vitalSigns.heartRate} bpm</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Temp:</span>
                        <span className="font-medium">{case_.vitalSigns.temperature}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>SpO2:</span>
                        <span className="font-medium">{case_.vitalSigns.oxygenSaturation}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <h4 className="font-medium text-sm">Atendimento</h4>
                    <div className="space-y-1">
                      <div className="flex items-center gap-2 text-sm">
                        <Stethoscope className="h-4 w-4 text-medical-primary" />
                        <span>{case_.assignedDoctor}</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <MapPin className="h-4 w-4 text-medical-primary" />
                        <span>{case_.location}</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-4 border-t">
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                      <Clock className="h-4 w-4 text-medical-primary" />
                      <span className="text-sm font-medium">
                        Tempo de espera: {case_.waitTime}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Activity className="h-4 w-4 text-medical-primary" />
                      <span className="text-sm font-medium">
                        Prioridade: {case_.priority}
                      </span>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <Button size="sm" variant="outline">
                      <Eye className="h-4 w-4 mr-1" />
                      Detalhes
                    </Button>
                    {case_.status === 'waiting' && (
                      <Button size="sm">
                        <Stethoscope className="h-4 w-4 mr-1" />
                        Atender
                      </Button>
                    )}
                    {case_.status === 'in_treatment' && (
                      <Button size="sm">
                        <CheckCircle className="h-4 w-4 mr-1" />
                        Finalizar
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