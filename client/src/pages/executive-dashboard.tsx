import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  ArrowUp, 
  ArrowDown, 
  Brain, 
  Building, 
  DollarSign, 
  Heart, 
  PieChart, 
  TrendingUp, 
  Users, 
  Zap,
  Target,
  Award,
  Clock,
  Shield,
  Activity
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart as RechartsPieChart, Pie, Cell } from 'recharts';

interface ExecutiveMetric {
  id: string;
  title: string;
  value: string;
  change: number;
  changeType: 'positive' | 'negative' | 'neutral';
  target: string;
  category: 'clinical' | 'operational' | 'financial' | 'quality';
  description: string;
}

interface HospitalPerformance {
  hospital: string;
  beds: number;
  occupancy: number;
  efficiency: number;
  satisfaction: number;
  revenue: number;
  aiScore: number;
}

interface QualityIndicator {
  name: string;
  current: number;
  target: number;
  benchmark: number;
  trend: 'up' | 'down' | 'stable';
}

const executiveMetrics: ExecutiveMetric[] = [
  {
    id: 'mortality-rate',
    title: 'Taxa de Mortalidade',
    value: '2.3%',
    change: -18,
    changeType: 'positive',
    target: '2.0%',
    category: 'clinical',
    description: 'Redução significativa com implementação IA'
  },
  {
    id: 'bed-occupancy',
    title: 'Ocupação de Leitos',
    value: '87%',
    change: +12,
    changeType: 'positive',
    target: '85%',
    category: 'operational',
    description: 'Otimização via gêmeos digitais'
  },
  {
    id: 'cost-per-patient',
    title: 'Custo por Paciente',
    value: 'R$ 8.450',
    change: -15,
    changeType: 'positive',
    target: 'R$ 8.000',
    category: 'financial',
    description: 'Economia com automação IA'
  },
  {
    id: 'patient-satisfaction',
    title: 'Satisfação Paciente',
    value: '4.7/5',
    change: +8,
    changeType: 'positive',
    target: '4.5/5',
    category: 'quality',
    description: 'Melhoria no atendimento'
  },
  {
    id: 'readmission-rate',
    title: 'Taxa de Reinternação',
    value: '9.2%',
    change: -22,
    changeType: 'positive',
    target: '8.5%',
    category: 'clinical',
    description: 'Predição precoce eficaz'
  },
  {
    id: 'staff-efficiency',
    title: 'Eficiência Equipe',
    value: '92%',
    change: +16,
    changeType: 'positive',
    target: '90%',
    category: 'operational',
    description: 'Automação de processos'
  },
  {
    id: 'revenue-per-bed',
    title: 'Receita por Leito',
    value: 'R$ 145k',
    change: +24,
    changeType: 'positive',
    target: 'R$ 140k',
    category: 'financial',
    description: 'Otimização de recursos'
  },
  {
    id: 'safety-score',
    title: 'Índice de Segurança',
    value: '96%',
    change: +11,
    changeType: 'positive',
    target: '95%',
    category: 'quality',
    description: 'Sistema de alertas avançado'
  }
];

const hospitalPerformance: HospitalPerformance[] = [
  {
    hospital: 'Hospital Central',
    beds: 450,
    occupancy: 87,
    efficiency: 92,
    satisfaction: 4.7,
    revenue: 12500000,
    aiScore: 94
  },
  {
    hospital: 'Hospital Norte',
    beds: 280,
    occupancy: 82,
    efficiency: 89,
    satisfaction: 4.5,
    revenue: 8200000,
    aiScore: 91
  },
  {
    hospital: 'Hospital Sul',
    beds: 340,
    occupancy: 91,
    efficiency: 95,
    satisfaction: 4.8,
    revenue: 10800000,
    aiScore: 96
  }
];

const qualityIndicators: QualityIndicator[] = [
  { name: 'Tempo Resposta Emergência', current: 8.5, target: 10, benchmark: 12, trend: 'up' },
  { name: 'Taxa de Infecção', current: 1.2, target: 1.5, benchmark: 2.1, trend: 'down' },
  { name: 'Satisfação Médicos', current: 4.6, target: 4.3, benchmark: 4.1, trend: 'up' },
  { name: 'Eficiência Cirúrgica', current: 94, target: 90, benchmark: 87, trend: 'up' }
];

const aiImpactData = [
  { month: 'Jan', savings: 145000, efficiency: 78, quality: 85 },
  { month: 'Fev', savings: 189000, efficiency: 82, quality: 87 },
  { month: 'Mar', savings: 234000, efficiency: 85, quality: 89 },
  { month: 'Abr', savings: 267000, efficiency: 88, quality: 91 },
  { month: 'Mai', savings: 312000, efficiency: 92, quality: 94 },
  { month: 'Jun', savings: 356000, efficiency: 95, quality: 96 }
];

const costDistribution = [
  { name: 'Pessoal', value: 45, color: '#3b82f6' },
  { name: 'Medicamentos', value: 23, color: '#10b981' },
  { name: 'Equipamentos', value: 18, color: '#f59e0b' },
  { name: 'Infraestrutura', value: 14, color: '#ef4444' }
];

export default function ExecutiveDashboard() {
  const [selectedMetric, setSelectedMetric] = useState<ExecutiveMetric | null>(null);
  const [timeRange, setTimeRange] = useState('6M');
  const [realTimeData, setRealTimeData] = useState(aiImpactData);

  useEffect(() => {
    const interval = setInterval(() => {
      setRealTimeData(prev => prev.map(item => ({
        ...item,
        efficiency: Math.max(70, Math.min(100, item.efficiency + (Math.random() - 0.5) * 2)),
        quality: Math.max(80, Math.min(100, item.quality + (Math.random() - 0.5) * 2))
      })));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'clinical': return 'bg-red-100 text-red-800';
      case 'operational': return 'bg-blue-100 text-blue-800';
      case 'financial': return 'bg-green-100 text-green-800';
      case 'quality': return 'bg-purple-100 text-purple-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getChangeIcon = (changeType: string) => {
    switch (changeType) {
      case 'positive': return <ArrowUp className="h-4 w-4 text-green-600" />;
      case 'negative': return <ArrowDown className="h-4 w-4 text-red-600" />;
      default: return <ArrowUp className="h-4 w-4 text-gray-600" />;
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <TrendingUp className="h-4 w-4 text-green-500" />;
      case 'down': return <TrendingUp className="h-4 w-4 text-red-500 rotate-180" />;
      default: return <Activity className="h-4 w-4 text-gray-500" />;
    }
  };

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Dashboard Executivo</h1>
          <p className="text-muted-foreground">
            Visão estratégica e indicadores de performance hospitalar
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm">
            <Brain className="mr-2 h-4 w-4" />
            Relatório IA
          </Button>
          <Button>
            <PieChart className="mr-2 h-4 w-4" />
            Exportar
          </Button>
        </div>
      </div>

      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Visão Geral</TabsTrigger>
          <TabsTrigger value="hospitals">Hospitais</TabsTrigger>
          <TabsTrigger value="quality">Qualidade</TabsTrigger>
          <TabsTrigger value="ai-impact">Impacto IA</TabsTrigger>
          <TabsTrigger value="financial">Financeiro</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {executiveMetrics.map((metric) => (
              <Card 
                key={metric.id}
                className={`cursor-pointer transition-all hover:shadow-lg ${
                  selectedMetric?.id === metric.id ? 'ring-2 ring-blue-500' : ''
                }`}
                onClick={() => setSelectedMetric(metric)}
              >
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <Badge className={getCategoryColor(metric.category)}>
                      {metric.category}
                    </Badge>
                    <div className="flex items-center space-x-1">
                      {getChangeIcon(metric.changeType)}
                      <span className={`text-sm font-medium ${
                        metric.changeType === 'positive' ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {Math.abs(metric.change)}%
                      </span>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <h3 className="font-semibold text-lg">{metric.title}</h3>
                    <div className="flex items-baseline justify-between">
                      <span className="text-2xl font-bold">{metric.value}</span>
                      <span className="text-sm text-muted-foreground">
                        Meta: {metric.target}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {metric.description}
                    </p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {selectedMetric && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Target className="h-5 w-5" />
                  <span>Análise Detalhada: {selectedMetric.title}</span>
                </CardTitle>
                <CardDescription>
                  {selectedMetric.description}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <h4 className="font-semibold">Valor Atual</h4>
                    <div className="text-3xl font-bold">{selectedMetric.value}</div>
                    <div className="flex items-center space-x-2">
                      {getChangeIcon(selectedMetric.changeType)}
                      <span className={`text-sm ${
                        selectedMetric.changeType === 'positive' ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {Math.abs(selectedMetric.change)}% vs período anterior
                      </span>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <h4 className="font-semibold">Meta</h4>
                    <div className="text-2xl font-bold text-blue-600">{selectedMetric.target}</div>
                    <div className="text-sm text-muted-foreground">
                      Objetivo estratégico 2025
                    </div>
                  </div>
                  <div className="space-y-2">
                    <h4 className="font-semibold">Progresso</h4>
                    <Progress value={85} className="h-3" />
                    <div className="text-sm text-muted-foreground">
                      85% da meta alcançada
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="hospitals" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {hospitalPerformance.map((hospital) => (
              <Card key={hospital.hospital}>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Building className="h-5 w-5" />
                    <span>{hospital.hospital}</span>
                  </CardTitle>
                  <CardDescription>
                    {hospital.beds} leitos • Score IA: {hospital.aiScore}%
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <div className="text-sm text-muted-foreground mb-1">Ocupação</div>
                        <div className="flex items-center space-x-2">
                          <Progress value={hospital.occupancy} className="flex-1 h-2" />
                          <span className="text-sm font-medium">{hospital.occupancy}%</span>
                        </div>
                      </div>
                      <div>
                        <div className="text-sm text-muted-foreground mb-1">Eficiência</div>
                        <div className="flex items-center space-x-2">
                          <Progress value={hospital.efficiency} className="flex-1 h-2" />
                          <span className="text-sm font-medium">{hospital.efficiency}%</span>
                        </div>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <div className="text-muted-foreground">Satisfação</div>
                        <div className="font-medium">{hospital.satisfaction}/5</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Receita</div>
                        <div className="font-medium">R$ {(hospital.revenue / 1000000).toFixed(1)}M</div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="quality" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {qualityIndicators.map((indicator) => (
              <Card key={indicator.name}>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>{indicator.name}</span>
                    {getTrendIcon(indicator.trend)}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Atual</span>
                      <span className="text-lg font-bold">{indicator.current}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Meta</span>
                      <span className="text-lg font-medium text-blue-600">{indicator.target}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Benchmark</span>
                      <span className="text-lg font-medium text-gray-600">{indicator.benchmark}</span>
                    </div>
                    <div className="mt-2">
                      <div className="text-sm text-muted-foreground mb-1">Performance vs Meta</div>
                      <Progress 
                        value={Math.min(100, (indicator.current / indicator.target) * 100)} 
                        className="h-2" 
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="ai-impact" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Economia com IA</CardTitle>
                <CardDescription>
                  Impacto financeiro da implementação
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={realTimeData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="savings" fill="#10b981" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Eficiência & Qualidade</CardTitle>
                <CardDescription>
                  Métricas de performance com IA
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={realTimeData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="efficiency" stroke="#3b82f6" strokeWidth={2} />
                    <Line type="monotone" dataKey="quality" stroke="#10b981" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center space-x-2">
                  <Brain className="h-5 w-5" />
                  <span>Modelos Ativos</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">47</div>
                <p className="text-sm text-muted-foreground">Neural Networks & ML</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center space-x-2">
                  <Zap className="h-5 w-5" />
                  <span>Predições/Dia</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">12.4k</div>
                <p className="text-sm text-muted-foreground">Tempo real</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center space-x-2">
                  <Award className="h-5 w-5" />
                  <span>Precisão Média</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">94.7%</div>
                <p className="text-sm text-muted-foreground">Todos os modelos</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center space-x-2">
                  <Clock className="h-5 w-5" />
                  <span>Tempo Resposta</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">12ms</div>
                <p className="text-sm text-muted-foreground">Latência média</p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="financial" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Distribuição de Custos</CardTitle>
                <CardDescription>
                  Breakdown dos gastos hospitalares
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <RechartsPieChart>
                    <Pie
                      data={costDistribution}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, value }) => `${name} ${value}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {costDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </RechartsPieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Indicadores Financeiros</CardTitle>
                <CardDescription>
                  Resumo das métricas financeiras principais
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                    <div>
                      <div className="font-semibold">Receita Total</div>
                      <div className="text-sm text-muted-foreground">Mensal</div>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-green-600">R$ 31.5M</div>
                      <div className="text-sm text-green-600">+18% vs mês anterior</div>
                    </div>
                  </div>
                  
                  <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                    <div>
                      <div className="font-semibold">Custo Operacional</div>
                      <div className="text-sm text-muted-foreground">Mensal</div>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-red-600">R$ 24.8M</div>
                      <div className="text-sm text-red-600">-12% vs mês anterior</div>
                    </div>
                  </div>
                  
                  <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                    <div>
                      <div className="font-semibold">Margem Operacional</div>
                      <div className="text-sm text-muted-foreground">Líquida</div>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-blue-600">21.3%</div>
                      <div className="text-sm text-blue-600">+5.2pp vs mês anterior</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Alert>
            <DollarSign className="h-4 w-4" />
            <AlertDescription>
              <strong>ROI da IA:</strong> R$ 3.20 economizados para cada R$ 1 investido em IA na saúde. 
              Projeção de economia anual: R$ 4.2M com a implementação completa.
            </AlertDescription>
          </Alert>
        </TabsContent>
      </Tabs>
    </div>
  );
}