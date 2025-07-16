import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  FileText, 
  Download, 
  Filter, 
  Calendar,
  BarChart3,
  TrendingUp,
  Users,
  Clock,
  FileSpreadsheet,
  FileImage,
  Search,
  RefreshCw,
  Eye,
  Share2
} from "lucide-react";
import { useState } from "react";

export default function Reports() {
  const [selectedTimeframe, setSelectedTimeframe] = useState<'today' | 'week' | 'month' | 'quarter'>('month');

  const reports = [
    {
      id: 1,
      title: "Relatório de Ocupação Hospitalar",
      description: "Análise completa da ocupação de leitos por departamento",
      type: "operational",
      format: "PDF",
      size: "2.4 MB",
      lastGenerated: "2 horas atrás",
      status: "ready",
      icon: BarChart3
    },
    {
      id: 2,
      title: "Análise de Performance de IA",
      description: "Métricas de precisão e eficiência dos modelos de ML",
      type: "analytics",
      format: "Excel",
      size: "890 KB",
      lastGenerated: "1 hora atrás",
      status: "ready",
      icon: TrendingUp
    },
    {
      id: 3,
      title: "Relatório de Satisfação do Paciente",
      description: "Pesquisa de satisfação e feedback dos pacientes",
      type: "quality",
      format: "PDF",
      size: "1.2 MB",
      lastGenerated: "30 min atrás",
      status: "ready",
      icon: Users
    },
    {
      id: 4,
      title: "Análise de Tempo de Espera",
      description: "Estatísticas de tempo de espera por departamento",
      type: "efficiency",
      format: "Excel",
      size: "650 KB",
      lastGenerated: "45 min atrás",
      status: "ready",
      icon: Clock
    },
    {
      id: 5,
      title: "Relatório Financeiro Mensal",
      description: "Análise financeira e custos operacionais",
      type: "financial",
      format: "PDF",
      size: "3.1 MB",
      lastGenerated: "Gerando...",
      status: "generating",
      icon: FileText
    },
    {
      id: 6,
      title: "Auditoria de Segurança",
      description: "Relatório de segurança e conformidade",
      type: "security",
      format: "PDF",
      size: "1.8 MB",
      lastGenerated: "6 horas atrás",
      status: "ready",
      icon: FileText
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ready': return 'bg-medical-primary/10 text-medical-primary border-medical-primary/30';
      case 'generating': return 'bg-warning/10 text-warning border-warning/30';
      case 'error': return 'bg-destructive/10 text-destructive border-destructive/30';
      default: return 'bg-muted/10 text-muted-foreground border-muted/30';
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'operational': return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'analytics': return 'bg-purple-100 text-purple-800 border-purple-200';
      case 'quality': return 'bg-green-100 text-green-800 border-green-200';
      case 'efficiency': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'financial': return 'bg-red-100 text-red-800 border-red-200';
      case 'security': return 'bg-gray-100 text-gray-800 border-gray-200';
      default: return 'bg-muted/10 text-muted-foreground border-muted/30';
    }
  };

  const getFormatIcon = (format: string) => {
    switch (format) {
      case 'PDF': return <FileText className="h-4 w-4 text-red-500" />;
      case 'Excel': return <FileSpreadsheet className="h-4 w-4 text-green-500" />;
      case 'Image': return <FileImage className="h-4 w-4 text-blue-500" />;
      default: return <FileText className="h-4 w-4" />;
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold medical-text">Relatórios Empresariais</h1>
          <p className="text-muted-foreground">Relatórios automáticos e análises detalhadas</p>
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
            <RefreshCw className="h-4 w-4 mr-2" />
            Atualizar
          </Button>
        </div>
      </div>

      {/* Filtros de Período */}
      <div className="flex items-center gap-4 p-4 bg-muted/20 rounded-lg">
        <div className="flex items-center gap-2">
          <Calendar className="h-4 w-4 text-medical-primary" />
          <span className="text-sm font-medium">Período:</span>
        </div>
        <div className="flex gap-2">
          {[
            { key: 'today', label: 'Hoje' },
            { key: 'week', label: 'Esta Semana' },
            { key: 'month', label: 'Este Mês' },
            { key: 'quarter', label: 'Trimestre' }
          ].map(({ key, label }) => (
            <Button
              key={key}
              size="sm"
              variant={selectedTimeframe === key ? "default" : "outline"}
              onClick={() => setSelectedTimeframe(key as any)}
            >
              {label}
            </Button>
          ))}
        </div>
      </div>

      {/* Estatísticas Rápidas */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="medical-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <FileText className="h-5 w-5 text-medical-primary" />
              <Badge variant="outline" className="text-xs">Total</Badge>
            </div>
            <p className="text-2xl font-bold">47</p>
            <p className="text-sm text-muted-foreground">Relatórios Gerados</p>
          </CardContent>
        </Card>

        <Card className="medical-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <Download className="h-5 w-5 text-medical-secondary" />
              <Badge variant="outline" className="text-xs">Hoje</Badge>
            </div>
            <p className="text-2xl font-bold">12</p>
            <p className="text-sm text-muted-foreground">Downloads</p>
          </CardContent>
        </Card>

        <Card className="medical-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <Clock className="h-5 w-5 text-warning" />
              <Badge variant="secondary" className="text-xs">Agendados</Badge>
            </div>
            <p className="text-2xl font-bold">8</p>
            <p className="text-sm text-muted-foreground">Automáticos</p>
          </CardContent>
        </Card>

        <Card className="medical-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <Share2 className="h-5 w-5 text-chart-4" />
              <Badge variant="outline" className="text-xs">Compartilhados</Badge>
            </div>
            <p className="text-2xl font-bold">23</p>
            <p className="text-sm text-muted-foreground">Este Mês</p>
          </CardContent>
        </Card>
      </div>

      {/* Lista de Relatórios */}
      <Card className="medical-card">
        <CardHeader>
          <CardTitle>Relatórios Disponíveis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {reports.map((report) => {
              const Icon = report.icon;
              
              return (
                <div key={report.id} className={`p-4 rounded-lg border-2 ${getStatusColor(report.status)}`}>
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-4">
                      <div className="w-12 h-12 bg-medical-primary/10 rounded-lg flex items-center justify-center">
                        <Icon className="h-6 w-6 text-medical-primary" />
                      </div>
                      
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <h3 className="font-semibold">{report.title}</h3>
                          <Badge variant="outline" className={`text-xs ${getTypeColor(report.type)}`}>
                            {report.type.toUpperCase()}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mb-2">{report.description}</p>
                        
                        <div className="flex items-center gap-4 text-xs text-muted-foreground">
                          <div className="flex items-center gap-1">
                            {getFormatIcon(report.format)}
                            <span>{report.format}</span>
                          </div>
                          <span>•</span>
                          <span>{report.size}</span>
                          <span>•</span>
                          <span>{report.lastGenerated}</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <Button size="sm" variant="outline" disabled={report.status === 'generating'}>
                        <Eye className="h-4 w-4 mr-1" />
                        Visualizar
                      </Button>
                      <Button size="sm" disabled={report.status === 'generating'}>
                        <Download className="h-4 w-4 mr-1" />
                        Download
                      </Button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}