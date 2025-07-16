import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  UserCheck, 
  Users,
  Calendar,
  Clock,
  MapPin,
  Phone,
  FileText,
  AlertCircle,
  CheckCircle,
  XCircle,
  Search,
  Filter,
  Plus,
  Eye,
  Edit,
  Stethoscope,
  Activity
} from "lucide-react";
import { useState } from "react";

export default function Admissions() {
  const [selectedStatus, setSelectedStatus] = useState<'all' | 'pending' | 'approved' | 'admitted'>('all');

  const admissions = [
    {
      id: "A001",
      patientName: "Carlos Oliveira",
      age: 45,
      gender: "M",
      admissionType: "Emergência",
      department: "Cardiologia",
      status: "pending",
      priority: "high",
      physician: "Dr. Ana Silva",
      bedAssigned: null,
      estimatedStay: "3-5 dias",
      admissionTime: "14:30",
      symptoms: ["Dor no peito", "Falta de ar"],
      insurance: "Plano Saúde A",
      contact: "(11) 98765-4321"
    },
    {
      id: "A002",
      patientName: "Maria Santos",
      age: 67,
      gender: "F",
      admissionType: "Programada",
      department: "Ortopedia",
      status: "approved",
      priority: "medium",
      physician: "Dr. Pedro Costa",
      bedAssigned: "B205",
      estimatedStay: "2-3 dias",
      admissionTime: "09:15",
      symptoms: ["Cirurgia do joelho"],
      insurance: "SUS",
      contact: "(11) 99876-5432"
    },
    {
      id: "A003",
      patientName: "João Silva",
      age: 34,
      gender: "M",
      admissionType: "Emergência",
      department: "Neurologia",
      status: "admitted",
      priority: "critical",
      physician: "Dr. Laura Mendes",
      bedAssigned: "UTI-03",
      estimatedStay: "7-10 dias",
      admissionTime: "02:45",
      symptoms: ["Convulsões", "Perda de consciência"],
      insurance: "Plano Saúde B",
      contact: "(11) 97654-3210"
    },
    {
      id: "A004",
      patientName: "Ana Costa",
      age: 28,
      gender: "F",
      admissionType: "Programada",
      department: "Obstetrícia",
      status: "approved",
      priority: "medium",
      physician: "Dr. Roberto Lima",
      bedAssigned: "MAT-15",
      estimatedStay: "2-3 dias",
      admissionTime: "16:20",
      symptoms: ["Parto cesariana"],
      insurance: "Plano Saúde C",
      contact: "(11) 96543-2109"
    },
    {
      id: "A005",
      patientName: "Roberto Ferreira",
      age: 58,
      gender: "M",
      admissionType: "Emergência",
      department: "Gastroenterologia",
      status: "pending",
      priority: "high",
      physician: "Dr. Fernanda Rocha",
      bedAssigned: null,
      estimatedStay: "4-6 dias",
      admissionTime: "11:10",
      symptoms: ["Dor abdominal severa", "Vômitos"],
      insurance: "SUS",
      contact: "(11) 95432-1098"
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'bg-warning/10 text-warning border-warning/30';
      case 'approved': return 'bg-medical-primary/10 text-medical-primary border-medical-primary/30';
      case 'admitted': return 'bg-medical-secondary/10 text-medical-secondary border-medical-secondary/30';
      case 'cancelled': return 'bg-destructive/10 text-destructive border-destructive/30';
      default: return 'bg-muted/10 text-muted-foreground border-muted/30';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'bg-destructive text-destructive-foreground';
      case 'high': return 'bg-warning text-warning-foreground';
      case 'medium': return 'bg-medical-primary text-medical-primary-foreground';
      case 'low': return 'bg-medical-secondary text-medical-secondary-foreground';
      default: return 'bg-muted text-muted-foreground';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending': return <Clock className="h-4 w-4" />;
      case 'approved': return <CheckCircle className="h-4 w-4" />;
      case 'admitted': return <UserCheck className="h-4 w-4" />;
      case 'cancelled': return <XCircle className="h-4 w-4" />;
      default: return <AlertCircle className="h-4 w-4" />;
    }
  };

  const filteredAdmissions = selectedStatus === 'all' 
    ? admissions 
    : admissions.filter(a => a.status === selectedStatus);

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold medical-text">Gestão de Admissões</h1>
          <p className="text-muted-foreground">Controle completo do processo de admissão</p>
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
          <Button size="sm">
            <Plus className="h-4 w-4 mr-2" />
            Nova Admissão
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
              {admissions.filter(a => a.status === 'pending').length}
            </p>
            <p className="text-sm text-muted-foreground">Aguardando Aprovação</p>
          </CardContent>
        </Card>

        <Card className="medical-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <CheckCircle className="h-5 w-5 text-medical-primary" />
              <Badge variant="outline" className="text-xs">Aprovadas</Badge>
            </div>
            <p className="text-2xl font-bold text-medical-primary">
              {admissions.filter(a => a.status === 'approved').length}
            </p>
            <p className="text-sm text-muted-foreground">Leito Designado</p>
          </CardContent>
        </Card>

        <Card className="medical-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <UserCheck className="h-5 w-5 text-medical-secondary" />
              <Badge variant="outline" className="text-xs">Admitidos</Badge>
            </div>
            <p className="text-2xl font-bold text-medical-secondary">
              {admissions.filter(a => a.status === 'admitted').length}
            </p>
            <p className="text-sm text-muted-foreground">Já Internados</p>
          </CardContent>
        </Card>

        <Card className="medical-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <Users className="h-5 w-5 text-chart-4" />
              <Badge variant="outline" className="text-xs">Total</Badge>
            </div>
            <p className="text-2xl font-bold text-chart-4">{admissions.length}</p>
            <p className="text-sm text-muted-foreground">Hoje</p>
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
            { key: 'approved', label: 'Aprovadas' },
            { key: 'admitted', label: 'Admitidos' }
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

      {/* Lista de Admissões */}
      <Card className="medical-card">
        <CardHeader>
          <CardTitle>Admissões - {new Date().toLocaleDateString()}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {filteredAdmissions.map((admission) => (
              <div key={admission.id} className={`p-4 rounded-lg border-2 ${getStatusColor(admission.status)}`}>
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 bg-medical-primary/10 rounded-full flex items-center justify-center">
                      <Users className="h-6 w-6 text-medical-primary" />
                    </div>
                    
                    <div>
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="font-semibold text-lg">{admission.patientName}</h3>
                        <Badge variant="outline" className="text-xs">
                          {admission.id}
                        </Badge>
                        <Badge className={`text-xs ${getPriorityColor(admission.priority)}`}>
                          {admission.priority.toUpperCase()}
                        </Badge>
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <p className="text-muted-foreground">Idade/Gênero</p>
                          <p className="font-medium">{admission.age} anos, {admission.gender}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Departamento</p>
                          <p className="font-medium">{admission.department}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Tipo</p>
                          <p className="font-medium">{admission.admissionType}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Médico</p>
                          <p className="font-medium">{admission.physician}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1">
                      {getStatusIcon(admission.status)}
                      <span className="text-sm font-medium capitalize">{admission.status}</span>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  <div className="space-y-2">
                    <h4 className="font-medium text-sm">Informações Clínicas</h4>
                    <div className="space-y-1">
                      <div className="flex items-center gap-2 text-sm">
                        <Activity className="h-4 w-4 text-medical-primary" />
                        <span>Sintomas: {admission.symptoms.join(', ')}</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <Calendar className="h-4 w-4 text-medical-primary" />
                        <span>Estadia: {admission.estimatedStay}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <h4 className="font-medium text-sm">Leito & Horário</h4>
                    <div className="space-y-1">
                      <div className="flex items-center gap-2 text-sm">
                        <MapPin className="h-4 w-4 text-medical-primary" />
                        <span>Leito: {admission.bedAssigned || 'Não designado'}</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <Clock className="h-4 w-4 text-medical-primary" />
                        <span>Horário: {admission.admissionTime}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <h4 className="font-medium text-sm">Contato & Seguro</h4>
                    <div className="space-y-1">
                      <div className="flex items-center gap-2 text-sm">
                        <Phone className="h-4 w-4 text-medical-primary" />
                        <span>{admission.contact}</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <FileText className="h-4 w-4 text-medical-primary" />
                        <span>{admission.insurance}</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-4 border-t">
                  <div className="flex items-center gap-2">
                    <Stethoscope className="h-4 w-4 text-medical-primary" />
                    <span className="text-sm font-medium">Prioridade: {admission.priority}</span>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <Button size="sm" variant="outline">
                      <Eye className="h-4 w-4 mr-1" />
                      Visualizar
                    </Button>
                    <Button size="sm" variant="outline">
                      <Edit className="h-4 w-4 mr-1" />
                      Editar
                    </Button>
                    {admission.status === 'pending' && (
                      <Button size="sm">
                        <CheckCircle className="h-4 w-4 mr-1" />
                        Aprovar
                      </Button>
                    )}
                    {admission.status === 'approved' && (
                      <Button size="sm">
                        <UserCheck className="h-4 w-4 mr-1" />
                        Admitir
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