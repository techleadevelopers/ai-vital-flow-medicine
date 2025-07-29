import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Activity, 
  Brain, 
  Cpu, 
  Database, 
  GitBranch, 
  Layers, 
  MonitorSpeaker, 
  Network, 
  Radar, 
  Sparkles,
  TrendingUp,
  Zap
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

interface DigitalTwinModel {
  id: string;
  name: string;
  type: 'hospital' | 'department' | 'patient' | 'equipment';
  status: 'active' | 'processing' | 'offline';
  accuracy: number;
  lastUpdated: string;
  predictions: number;
  confidence: number;
}

interface SimulationScenario {
  id: string;
  name: string;
  type: 'capacity' | 'emergency' | 'resource' | 'workflow';
  status: 'running' | 'completed' | 'pending';
  duration: string;
  results: {
    efficiency: number;
    cost: number;
    satisfaction: number;
  };
}

const digitalTwinModels: DigitalTwinModel[] = [
  {
    id: 'hospital-main',
    name: 'Hospital Principal',
    type: 'hospital',
    status: 'active',
    accuracy: 94.2,
    lastUpdated: '2 min atrás',
    predictions: 1247,
    confidence: 92.5
  },
  {
    id: 'uti-model',
    name: 'UTI Adulto',
    type: 'department',
    status: 'processing',
    accuracy: 97.8,
    lastUpdated: '5 min atrás',
    predictions: 456,
    confidence: 95.1
  },
  {
    id: 'patient-flow',
    name: 'Fluxo de Pacientes',
    type: 'patient',
    status: 'active',
    accuracy: 89.3,
    lastUpdated: '1 min atrás',
    predictions: 2341,
    confidence: 88.7
  },
  {
    id: 'equipment-network',
    name: 'Rede de Equipamentos',
    type: 'equipment',
    status: 'active',
    accuracy: 96.1,
    lastUpdated: '3 min atrás',
    predictions: 789,
    confidence: 94.3
  }
];

const simulationScenarios: SimulationScenario[] = [
  {
    id: 'surge-capacity',
    name: 'Capacidade de Surto COVID',
    type: 'emergency',
    status: 'completed',
    duration: '2h 34m',
    results: { efficiency: 87, cost: 145000, satisfaction: 79 }
  },
  {
    id: 'bed-optimization',
    name: 'Otimização de Leitos',
    type: 'capacity',
    status: 'running',
    duration: '45m',
    results: { efficiency: 92, cost: 98000, satisfaction: 85 }
  },
  {
    id: 'staff-scheduling',
    name: 'Escala de Funcionários',
    type: 'resource',
    status: 'pending',
    duration: '1h 20m',
    results: { efficiency: 0, cost: 0, satisfaction: 0 }
  }
];

const performanceData = [
  { hour: '00:00', accuracy: 94, predictions: 45, confidence: 92 },
  { hour: '04:00', accuracy: 96, predictions: 52, confidence: 94 },
  { hour: '08:00', accuracy: 93, predictions: 78, confidence: 89 },
  { hour: '12:00', accuracy: 95, predictions: 89, confidence: 93 },
  { hour: '16:00', accuracy: 97, predictions: 76, confidence: 95 },
  { hour: '20:00', accuracy: 94, predictions: 65, confidence: 91 }
];

const resourceUtilization = [
  { name: 'CPU', value: 76, color: '#10b981' },
  { name: 'Memória', value: 68, color: '#3b82f6' },
  { name: 'Armazenamento', value: 45, color: '#f59e0b' },
  { name: 'Rede', value: 23, color: '#ef4444' }
];

export default function DigitalTwin() {
  const [selectedModel, setSelectedModel] = useState<DigitalTwinModel | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [realTimeData, setRealTimeData] = useState(performanceData);

  useEffect(() => {
    const interval = setInterval(() => {
      setRealTimeData(prev => prev.map(item => ({
        ...item,
        accuracy: Math.max(85, Math.min(100, item.accuracy + (Math.random() - 0.5) * 4)),
        predictions: Math.max(20, Math.min(120, item.predictions + (Math.random() - 0.5) * 10)),
        confidence: Math.max(80, Math.min(100, item.confidence + (Math.random() - 0.5) * 3))
      })));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const startSimulation = () => {
    setIsSimulating(true);
    setTimeout(() => setIsSimulating(false), 5000);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500';
      case 'processing': return 'bg-yellow-500';
      case 'offline': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'hospital': return <MonitorSpeaker className="h-5 w-5" />;
      case 'department': return <Layers className="h-5 w-5" />;
      case 'patient': return <Activity className="h-5 w-5" />;
      case 'equipment': return <Cpu className="h-5 w-5" />;
      default: return <Database className="h-5 w-5" />;
    }
  };

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Gêmeos Digitais</h1>
          <p className="text-muted-foreground">
            Simulação avançada e modelagem preditiva do ambiente hospitalar
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button onClick={startSimulation} disabled={isSimulating}>
            {isSimulating ? (
              <>
                <Radar className="mr-2 h-4 w-4 animate-spin" />
                Simulando...
              </>
            ) : (
              <>
                <Sparkles className="mr-2 h-4 w-4" />
                Nova Simulação
              </>
            )}
          </Button>
        </div>
      </div>

      <Tabs defaultValue="models" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="models">Modelos Digitais</TabsTrigger>
          <TabsTrigger value="simulation">Simulações</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
          <TabsTrigger value="infrastructure">Infraestrutura</TabsTrigger>
        </TabsList>

        <TabsContent value="models" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {digitalTwinModels.map((model) => (
              <Card 
                key={model.id} 
                className={`cursor-pointer transition-all hover:shadow-lg ${
                  selectedModel?.id === model.id ? 'ring-2 ring-green-500' : ''
                }`}
                onClick={() => setSelectedModel(model)}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      {getTypeIcon(model.type)}
                      <div className={`h-2 w-2 rounded-full ${getStatusColor(model.status)}`} />
                    </div>
                    <Badge variant="secondary">
                      {model.accuracy}% precisão
                    </Badge>
                  </div>
                  <CardTitle className="text-lg">{model.name}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Predições</span>
                      <span className="font-medium">{model.predictions.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Confiança</span>
                      <span className="font-medium">{model.confidence}%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Atualizado</span>
                      <span className="font-medium">{model.lastUpdated}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {selectedModel && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Brain className="h-5 w-5" />
                  <span>Detalhes do Modelo: {selectedModel.name}</span>
                </CardTitle>
                <CardDescription>
                  Análise detalhada do gêmeo digital selecionado
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="font-semibold mb-3">Métricas de Performance</h3>
                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Precisão</span>
                          <span>{selectedModel.accuracy}%</span>
                        </div>
                        <Progress value={selectedModel.accuracy} className="h-2" />
                      </div>
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Confiança</span>
                          <span>{selectedModel.confidence}%</span>
                        </div>
                        <Progress value={selectedModel.confidence} className="h-2" />
                      </div>
                    </div>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-3">Estatísticas</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Total de Predições</span>
                        <span className="font-medium">{selectedModel.predictions.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Taxa de Atualização</span>
                        <span className="font-medium">30 seg</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Última Sincronização</span>
                        <span className="font-medium">{selectedModel.lastUpdated}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="simulation" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {simulationScenarios.map((scenario) => (
              <Card key={scenario.id}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">{scenario.name}</CardTitle>
                    <Badge variant={scenario.status === 'completed' ? 'default' : 
                                  scenario.status === 'running' ? 'secondary' : 'outline'}>
                      {scenario.status}
                    </Badge>
                  </div>
                  <CardDescription>
                    Duração: {scenario.duration}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {scenario.status === 'completed' && (
                    <div className="space-y-3">
                      <div className="flex justify-between text-sm">
                        <span>Eficiência</span>
                        <span className="font-medium">{scenario.results.efficiency}%</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Custo</span>
                        <span className="font-medium">R$ {scenario.results.cost.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Satisfação</span>
                        <span className="font-medium">{scenario.results.satisfaction}%</span>
                      </div>
                    </div>
                  )}
                  {scenario.status === 'running' && (
                    <div className="space-y-3">
                      <div className="flex items-center space-x-2">
                        <Radar className="h-4 w-4 animate-spin" />
                        <span className="text-sm">Executando simulação...</span>
                      </div>
                      <Progress value={65} className="h-2" />
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Performance em Tempo Real</CardTitle>
                <CardDescription>
                  Métricas de precisão e confiança dos modelos
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={realTimeData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="hour" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={2} />
                    <Line type="monotone" dataKey="confidence" stroke="#3b82f6" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Utilização de Recursos</CardTitle>
                <CardDescription>
                  Consumo computacional dos gêmeos digitais
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {resourceUtilization.map((resource) => (
                    <div key={resource.name} className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>{resource.name}</span>
                        <span className="font-medium">{resource.value}%</span>
                      </div>
                      <Progress value={resource.value} className="h-2" />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="infrastructure" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Cpu className="h-5 w-5" />
                  <span>Processamento</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>CPU Clusters</span>
                    <span className="font-medium">8 ativos</span>
                  </div>
                  <div className="flex justify-between">
                    <span>GPU Aceleração</span>
                    <span className="font-medium">4 RTX 4090</span>
                  </div>
                  <div className="flex justify-between">
                    <span>TensorFlow Nodes</span>
                    <span className="font-medium">12 instâncias</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Database className="h-5 w-5" />
                  <span>Armazenamento</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Dados Históricos</span>
                    <span className="font-medium">2.4 TB</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Modelos Treinados</span>
                    <span className="font-medium">156 GB</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Cache em Tempo Real</span>
                    <span className="font-medium">32 GB</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Network className="h-5 w-5" />
                  <span>Conectividade</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>APIs Ativas</span>
                    <span className="font-medium">24</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Websockets</span>
                    <span className="font-medium">156 conexões</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Latência Média</span>
                    <span className="font-medium">12ms</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Alert>
            <Zap className="h-4 w-4" />
            <AlertDescription>
              Todos os sistemas estão operando com performance otimizada. 
              Próxima manutenção programada para 15/07/2025 às 02:00.
            </AlertDescription>
          </Alert>
        </TabsContent>
      </Tabs>
    </div>
  );
}