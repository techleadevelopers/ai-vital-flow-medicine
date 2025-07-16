import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  ClipboardList, 
  Calendar,
  Clock,
  MapPin,
  FileText,
  CheckCircle,
  AlertTriangle,
  Users,
  Heart,
  Home,
  Car,
  Stethoscope,
  Eye,
  Download,
  Filter,
  Search,
  RefreshCw,
  Plus
} from "lucide-react";
import { useState } from "react";

export default function Discharges() {
  const [selectedStatus, setSelectedStatus] = useState<'all' | 'pending' | 'ready' | 'completed'>('all');

  const discharges = [
    {
      id: "D001",
      patientName: "Ana Silva",
      age: 45,
      gender: "F",
      patientId: "P2341",
      department: "Cirurgia Geral",
      bedNumber: "B204",
      admissionDate: "2024-01-10",
      dischargeDate: "2024-01-15",
      lengthOfStay: "5 dias",
      status: "ready",
      dischargeType: "Home",
      physician: "Dr. Carlos Santos",
      instructions: "Repouso por 7 dias, retorno em 15 dias",
      medications: ["Dipirona 500mg", "Amoxicilina 875mg"],
      followUp: "Consulta em 15 dias",
      condition: "Stable",
      insurance: "Plano Saúde A"
    },
    {
      id: "D002",
      patientName: "João Oliveira",
      age: 67,
      gender: "M",
      patientId: "P2342",
      department: "Cardiologia",
      bedNumber: "UTI-05",
      admissionDate: "2024-01-12",
      dischargeDate: "2024-01-15",
      lengthOfStay: "3 dias",
      status: "pending",
      dischargeType: "Transfer",
      physician: "Dr. Maria Costa",
      instructions: "Transferência para hospital especializado",
      medications: ["Losartana 50mg", "AAS 100mg", "Sinvastatina 40mg"],
      followUp: "Acompanhamento cardiológico",
      condition: "Stable",
      insurance: "SUS"
    },
    {
      id: "D003",
      patientName: "Maria Santos",
      age: 34,
      gender: "F",
      patientId: "P2343",
      department: "Maternidade",
      bedNumber: "MAT-12",
      admissionDate: "2024-01-14",
      dischargeDate: "2024-01-15",
      lengthOfStay: "1 dia",
      status: "completed",
      dischargeType: "Home",
      physician: "Dr. Laura Mendes",
      instructions: "Cuidados com recém-nascido, amamentação",
      medications: ["Sulfato ferroso", "Ácido fólico"],
      followUp: "Consulta puerpério em 7 dias",
      condition: "Excellent",
      insurance: "Plano Saúde B"
    },
    {
      id: "D004",
      patientName: "Pedro Costa",
      age: 58,
      gender: "M",
      patientId: "P2344",
      department: "Ortopedia",
      bedNumber: "B301",
      admissionDate: "2024-01-11",
      dischargeDate: "2024-01-15",
      lengthOfStay: "4 dias",
      status: "pending",
      dischargeType: "Home",
      physician: "Dr. Roberto Lima",
      instructions: "Fisioterapia 3x/semana, uso de muletas",
      medications: ["Ibuprofeno 600mg", "Paracetamol 750mg"],
      followUp: "Retorno em 10 dias",
      condition: "Good",
      insurance: "SUS"
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'bg-warning/10 text-warning border-warning/30';
      case 'ready': return 'bg-medical-primary/10 text-medical-primary border-medical-primary/30';
      case 'completed': return 'bg-medical-secondary/10 text-medical-secondary border-medical-secondary/30';
      default: return 'bg-muted/10 text-muted-foreground border-muted/30';
    }
  };

  const getConditionColor = (condition: string) => {
    switch (condition) {
      case 'Excellent': return 'bg-medical-secondary text-medical-secondary-foreground';
      case 'Good': return 'bg-medical-primary text-medical-primary-foreground';
      case 'Stable': return 'bg-chart-4 text-chart-4-foreground';
      case 'Poor': return 'bg-warning text-warning-foreground';
      default: return 'bg-muted text-muted-foreground';
    }
  };

  const getDischargeTypeIcon = (type: string) => {
    switch (type) {
      case 'Home': return <Home className="h-4 w-4" />;
      case 'Transfer': return <Car className="h-4 w-4" />;
      case 'Against Medical Advice': return <AlertTriangle className="h-4 w-4" />;
      default: return <Home className="h-4 w-4" />;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending': return <Clock className="h-4 w-4" />;
      case 'ready': return <CheckCircle className="h-4 w-4" />;
      case 'completed': return <CheckCircle className="h-4 w-4" />;
      default: return <AlertTriangle className="h-4 w-4" />;
    }
  };

  const filteredDischarges = selectedStatus === 'all' 
    ? discharges 
    : discharges.filter(d => d.status === selectedStatus);

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold medical-text">Gestão de Altas</h1>
          <p className="text-muted-foreground">Controle completo do processo de alta hospitalar</p>
        </div>
        
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline">
            <Search className="h-4 w-4 mr-2" />
            Buscar
          </Button>
          <Button size="sm" variant="outline">
            <Filter className="h-4 w-4 mr-2" />
            Filtros
          </Button>
          <Button size="sm" variant="outline">
            <RefreshCw className="h-4 w-4 mr-2" />
            Atualizar
          </Button>
          <Button size="sm">
            <Plus className="h-4 w-4 mr-2" />
            Nova Alta
          </Button>
        </div>
      </div>

      {/* Estatísticas */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="medical-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <Clock className="h-5 w-5 text-warning" />
              <Badge variant="secondary" className="text-xs">Pendentes</Badge>
            </div>
            <p className="text-2xl font-bold text-warning">
              {discharges.filter(d => d.status === 'pending').length}
            </p>
            <p className="text-sm text-muted-foreground">Aguardando Processo</p>
          </CardContent>
        </Card>

        <Card className="medical-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <CheckCircle className="h-5 w-5 text-medical-primary" />
              <Badge variant="outline" className="text-xs">Prontas</Badge>
            </div>
            <p className="text-2xl font-bold text-medical-primary">
              {discharges.filter(d => d.status === 'ready').length}
            </p>
            <p className="text-sm text-muted-foreground">Prontas para Alta</p>
          </CardContent>
        </Card>

        <Card className="medical-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <ClipboardList className="h-5 w-5 text-medical-secondary" />
              <Badge variant="outline" className="text-xs">Concluídas</Badge>
            </div>
            <p className="text-2xl font-bold text-medical-secondary">
              {discharges.filter(d => d.status === 'completed').length}
            </p>
            <p className="text-sm text-muted-foreground">Altas Concluídas</p>
          </CardContent>
        </Card>

        <Card className="medical-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <Calendar className="h-5 w-5 text-chart-4" />
              <Badge variant="outline" className="text-xs">Média</Badge>
            </div>
            <p className="text-2xl font-bold text-chart-4">3.2</p>
            <p className="text-sm text-muted-foreground">Dias Internação</p>
          </CardContent>
        </Card>
      </div>

      {/* Filtros de Status */}
      <div className="flex items-center gap-2 p-4 bg-muted/20 rounded-lg">
        <span className="text-sm font-medium">Status:</span>
        <div className="flex gap-2">
          {[
            { key: 'all', label: 'Todos' },
            { key: 'pending', label: 'Pendentes' },
            { key: 'ready', label: 'Prontas' },
            { key: 'completed', label: 'Concluídas' }
          ].map(({ key, label }) => (
            <Button
              key={key}
              size="sm"
              variant={selectedStatus === key ? "default" : "outline"}
              onClick={() => setSelectedStatus(key as any)}
            >
              {label}
            </Button>
          ))}
        </div>
      </div>

      {/* Lista de Altas */}
      <Card className="medical-card">
        <CardHeader>
          <CardTitle>Altas Programadas - {new Date().toLocaleDateString()}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {filteredDischarges.map((discharge) => (
              <div key={discharge.id} className={`p-4 rounded-lg border-2 ${getStatusColor(discharge.status)}`}>
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 bg-medical-primary/10 rounded-full flex items-center justify-center">
                      <ClipboardList className="h-6 w-6 text-medical-primary" />
                    </div>
                    
                    <div>
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="font-semibold text-lg">{discharge.patientName}</h3>
                        <Badge variant="outline" className="text-xs">
                          {discharge.patientId}
                        </Badge>
                        <Badge className={`text-xs ${getConditionColor(discharge.condition)}`}>
                          {discharge.condition}
                        </Badge>
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <p className="text-muted-foreground">Idade/Gênero</p>
                          <p className="font-medium">{discharge.age} anos, {discharge.gender}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Departamento</p>
                          <p className="font-medium">{discharge.department}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Leito</p>
                          <p className="font-medium">{discharge.bedNumber}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Permanência</p>
                          <p className="font-medium">{discharge.lengthOfStay}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1">
                      {getStatusIcon(discharge.status)}
                      <span className="text-sm font-medium capitalize">{discharge.status}</span>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  <div className="space-y-2">
                    <h4 className="font-medium text-sm">Informações Médicas</h4>
                    <div className="space-y-1">
                      <div className="flex items-center gap-2 text-sm">
                        <Stethoscope className="h-4 w-4 text-medical-primary" />
                        <span>Dr: {discharge.physician}</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <Calendar className="h-4 w-4 text-medical-primary" />
                        <span>Admissão: {new Date(discharge.admissionDate).toLocaleDateString()}</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <MapPin className="h-4 w-4 text-medical-primary" />
                        <span>Alta: {new Date(discharge.dischargeDate).toLocaleDateString()}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <h4 className="font-medium text-sm">Tipo de Alta</h4>
                    <div className="space-y-1">
                      <div className="flex items-center gap-2 text-sm">
                        {getDischargeTypeIcon(discharge.dischargeType)}
                        <span>{discharge.dischargeType}</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <FileText className="h-4 w-4 text-medical-primary" />
                        <span>{discharge.insurance}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <h4 className="font-medium text-sm">Acompanhamento</h4>
                    <div className="space-y-1">
                      <div className="flex items-center gap-2 text-sm">
                        <Heart className="h-4 w-4 text-medical-primary" />
                        <span>{discharge.followUp}</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="space-y-3">
                  <div>
                    <h4 className="font-medium text-sm mb-2">Instruções de Alta</h4>
                    <p className="text-sm text-muted-foreground">{discharge.instructions}</p>
                  </div>
                  
                  <div>
                    <h4 className="font-medium text-sm mb-2">Medicamentos</h4>
                    <div className="flex flex-wrap gap-1">
                      {discharge.medications.map((med, index) => (
                        <Badge key={index} variant="outline" className="text-xs">
                          {med}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-4 border-t">
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                      <Calendar className="h-4 w-4 text-medical-primary" />
                      <span className="text-sm font-medium">
                        Permanência: {discharge.lengthOfStay}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      {getDischargeTypeIcon(discharge.dischargeType)}
                      <span className="text-sm font-medium">
                        Tipo: {discharge.dischargeType}
                      </span>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <Button size="sm" variant="outline">
                      <Eye className="h-4 w-4 mr-1" />
                      Visualizar
                    </Button>
                    <Button size="sm" variant="outline">
                      <Download className="h-4 w-4 mr-1" />
                      Relatório
                    </Button>
                    {discharge.status === 'pending' && (
                      <Button size="sm">
                        <CheckCircle className="h-4 w-4 mr-1" />
                        Aprovar Alta
                      </Button>
                    )}
                    {discharge.status === 'ready' && (
                      <Button size="sm">
                        <ClipboardList className="h-4 w-4 mr-1" />
                        Concluir Alta
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