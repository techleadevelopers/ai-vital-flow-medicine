import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  TrendingUp, 
  BarChart3, 
  PieChart, 
  LineChart, 
  Target,
  Users,
  Calendar,
  Download,
  Filter,
  RefreshCw
} from "lucide-react";

export default function Analytics() {
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold medical-text">Analytics Avançados</h1>
          <p className="text-muted-foreground">Análises profundas com IA e Machine Learning</p>
        </div>
        
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline">
            <Filter className="h-4 w-4 mr-2" />
            Filtros
          </Button>
          <Button size="sm" variant="outline">
            <Download className="h-4 w-4 mr-2" />
            Exportar
          </Button>
          <Button size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Atualizar
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { title: "Eficiência Operacional", value: "94.2%", trend: "+2.3%", icon: Target },
          { title: "Satisfação Paciente", value: "4.8/5", trend: "+0.2", icon: Users },
          { title: "Tempo Médio Espera", value: "12 min", trend: "-3 min", icon: Calendar },
          { title: "ROI IA", value: "247%", trend: "+15%", icon: TrendingUp }
        ].map((metric, index) => (
          <Card key={index} className="medical-card">
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-2">
                <metric.icon className="h-5 w-5 text-medical-primary" />
                <Badge variant="outline" className="text-xs">
                  {metric.trend}
                </Badge>
              </div>
              <p className="text-2xl font-bold">{metric.value}</p>
              <p className="text-sm text-muted-foreground">{metric.title}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="medical-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Análise de Tendências
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64 flex items-center justify-center text-muted-foreground">
              <p>Gráfico de tendências seria renderizado aqui</p>
            </div>
          </CardContent>
        </Card>

        <Card className="medical-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <PieChart className="h-5 w-5" />
              Distribuição de Recursos
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64 flex items-center justify-center text-muted-foreground">
              <p>Gráfico de pizza seria renderizado aqui</p>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="medical-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <LineChart className="h-5 w-5" />
            Performance Histórica
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-96 flex items-center justify-center text-muted-foreground">
            <p>Gráfico de linha histórico seria renderizado aqui</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}