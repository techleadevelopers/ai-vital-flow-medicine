import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api"; // Importa sua api mockada

// Componentes da UI (já existentes)
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

// Ícones (já existentes)
import { 
  TrendingUp, 
  BarChart3, 
  PieChart as PieChartIcon, // Renomeado para evitar conflito com o componente de gráfico
  LineChart as LineChartIcon, // Renomeado para evitar conflito
  Target,
  Users,
  Calendar,
  Download,
  Filter,
  RefreshCw,
  Loader2 // Ícone para o estado de carregamento
} from "lucide-react";

// Componentes de Gráfico da Recharts
import { 
  ResponsiveContainer, 
  LineChart, 
  BarChart,
  PieChart,
  Pie,
  Cell,
  CartesianGrid, 
  XAxis, 
  YAxis, 
  Tooltip, 
  Legend, 
  Line,
  Bar
} from 'recharts';

// Cores para o gráfico de pizza
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

// **CORREÇÃO 1: Definir a interface para os dados do analytics**
// Isso informa ao TypeScript qual é o formato do objeto 'analyticsData'
interface AnalyticsData {
  trends: { name: string; Admissões: number; Altas: number; }[];
  resources: { name: string; "Leitos Ocupados": number; }[];
  performance: { month: string; "Satisfação Paciente": number; "Tempo Médio Espera": number; }[];
}

export default function Analytics() {
  // **CORREÇÃO 2: Aplicar a interface ao hook useQuery**
  // Agora o TypeScript sabe que 'analyticsData' pode ter as propriedades 'trends', 'resources', etc.
  const { data: analyticsData, isLoading } = useQuery<AnalyticsData>({
    queryKey: ['analyticsData'],
    // @ts-ignore - Ignorando o erro aqui, pois a correção será feita no arquivo api.ts
    queryFn: api.getAnalyticsData, 
  });

  // Renderiza um estado de carregamento enquanto os dados não chegam
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Loader2 className="h-16 w-16 animate-spin text-medical-primary" />
        <p className="ml-4 text-lg">Carregando dados de análise...</p>
      </div>
    );
  }

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

      {/* Cards de KPI (permanecem os mesmos) */}
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
        {/* Card de Análise de Tendências com Gráfico de Barras */}
        <Card className="medical-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Análise de Tendências (Admissões vs. Altas)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={analyticsData?.trends}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="Admissões" fill="#8884d8" />
                <Bar dataKey="Altas" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Card de Distribuição de Recursos com Gráfico de Pizza */}
        <Card className="medical-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <PieChartIcon className="h-5 w-5" />
              Distribuição de Recursos (Leitos Ocupados)
            </CardTitle>
          </CardHeader>
          <CardContent>
             <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={analyticsData?.resources}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    outerRadius={110}
                    fill="#8884d8"
                    dataKey="Leitos Ocupados"
                    nameKey="name"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    {analyticsData?.resources.map((entry: any, index: number) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Card de Performance Histórica com Gráfico de Linha */}
      <Card className="medical-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <LineChartIcon className="h-5 w-5" />
            Performance Histórica (Satisfação vs. Tempo de Espera)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={analyticsData?.performance}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis yAxisId="left" label={{ value: 'Satisfação', angle: -90, position: 'insideLeft' }} />
              <YAxis yAxisId="right" orientation="right" label={{ value: 'Minutos', angle: -90, position: 'insideRight' }}/>
              <Tooltip />
              <Legend />
              <Line yAxisId="left" type="monotone" dataKey="Satisfação Paciente" stroke="#ffc658" strokeWidth={2} />
              <Line yAxisId="right" type="monotone" dataKey="Tempo Médio Espera" stroke="#ff8042" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
